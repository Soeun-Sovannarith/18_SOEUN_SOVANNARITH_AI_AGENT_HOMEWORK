"""
Core Safe AI Agent Implementation for the Library System.
Implements the Reasoning and Acting (ReAct) loop with application-level
permission control, structured tool execution through SafetyHarness,
and multi-backend LLM integration (Ollama, OpenAI, and Heuristic Fallback).
"""

from __future__ import annotations
import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from schemas import (
    AgentConfig,
    AgentResponse,
    AgentStep,
    ChatMessage,
    FunctionCall,
    Role,
    ToolCall,
    ToolResult,
    UserContext,
    UserRole,
)
from tools import ToolRegistry, registry as default_registry


def build_system_prompt(user_context: UserContext) -> str:
    """Build dynamic system prompt with active user context and role."""
    canonical_role = UserRole.normalize(user_context.role)
    role_display = canonical_role.value.upper()
    return f"""You are a helpful, courteous, and safe Library Assistant AI Agent.
Active Patron: {user_context.username} | Role: {role_display}

You have access to 4 database-backed tools:
1. `get_all_books`: Get all books from the library catalog, browse inventory, or search by keyword / category / title / topic.
2. `get_book_details`: View detailed metadata, description, available copy counts, and shelf locations for a numeric book ID.
3. `loan_book`: Loan a book copy by numeric book ID, update database inventory, and record a new active loan.
4. `return_book`: Return a loaned book using its Loan ID (e.g. 'LOAN-1001') or Book Title and restore available copies to inventory.

Operational & Safety Guidelines:
1. MANDATORY DATABASE RETRIEVAL: You have NO prior knowledge of what books exist in the library inventory. You MUST ALWAYS query the database using `get_all_books` or `get_book_details` to retrieve actual book information. NEVER fabricate, assume, or guess book availability, IDs, authors, or loan confirmations without querying the database.
2. GET ALL BOOKS / BROWSING / STATUS: When the patron asks what books are available, asks to list all books, asks about loaning status, or searches for topics, invoke `get_all_books(search_query=...)`.
3. SEEING BOOK DETAILS:
   - If the patron gives a numeric book ID (e.g. 'details for book 1'), invoke `get_book_details(book_id=1)`.
   - If the patron asks for details of a book by TITLE or TOPIC, you DO NOT know the numeric book_id yet! STEP 1: invoke `get_all_books(search_query="...")` to query the database and find the `book_id`. STEP 2: invoke `get_book_details(book_id=<found_id>)`.
4. LOANING / BORROWING BOOKS:
   - If the patron specifies a numeric book ID (e.g., 'loan book 2' or 'borrow book 2'), invoke `loan_book(book_id=2, duration_days=14)`.
   - If the patron asks to loan/borrow by TITLE or TOPIC, you DO NOT know the numeric book_id yet! STEP 1: invoke `get_all_books(search_query="...")` to query the database and find the `book_id`. STEP 2: invoke `loan_book(book_id=<found_id>, duration_days=14)`.
   - If the patron just says 'I want to borrow book' without specifying a title, invoke `get_all_books(search_query="")` to retrieve the database catalog and ask which book they would like.
   - If `duration_days` is not specified, use 14 days by default.
5. RETURNING BOOKS: When the patron returns a book or loan, invoke `return_book(loan_id=...)` passing the Loan ID or the book title.
6. CONVERSATIONAL GREETINGS: For simple greetings without library inquiries (e.g., "Hello", "How are you?"), DO NOT invoke any tools. Respond directly in friendly text.
7. FINAL RESPONSES: Base all responses strictly on observed tool outputs from the database. Format a clean, polite, and well-structured response for the patron.
"""


class Agent:
    """Agent that runs the ReAct loop and calls tools via SafetyHarness."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        user_context: Optional[UserContext] = None,
        tool_registry: Optional[ToolRegistry] = None,
        safety_harness: Optional[Any] = None,
    ) -> None:
        from harness import SafetyHarness

        self.config = config or AgentConfig()
        self.user_context = user_context or UserContext(role=UserRole.MEMBER)
        self.registry = tool_registry or default_registry
        self.harness = safety_harness or SafetyHarness(
            user_context=self.user_context,
            tool_registry=self.registry,
            max_tool_calls_per_run=self.config.max_steps * 2,
        )
        self.history: List[ChatMessage] = []
        self._init_history()

    def _init_history(self) -> None:
        """Initialize message history with system prompt."""
        system_content = self.config.system_prompt or build_system_prompt(self.user_context)
        self.history = [ChatMessage(role=Role.SYSTEM, content=system_content)]

    def reset(self) -> None:
        """Reset conversation memory and harness counters."""
        self.harness.tool_calls_count = 0
        self._init_history()

    def set_role(self, role: UserRole) -> None:
        """Switch user role dynamically."""
        canonical = UserRole.normalize(role)
        self.user_context.role = canonical
        self.harness.user_context.role = canonical
        self._init_history()

    def _log(self, message: str) -> None:
        """Print trace message if verbose is enabled."""
        if self.config.verbose:
            print(message)

    # LLM Drivers

    def _call_ollama(
        self, messages: List[ChatMessage], tools: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
        """Query local Ollama server using REST API."""
        url = os.getenv("OLLAMA_HOST", "http://localhost:11434/api/chat")
        payload = {
            "model": self.config.model_name,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }

        # Avoid tool call on greeting queries for the latest user turn
        last_user_text = ""
        for m in reversed(messages):
            if m.role == Role.USER and m.content:
                last_user_text = m.content.strip().lower()
                break

        is_greeting = last_user_text in (
            "hello", "hi", "hey", "hello!", "hi!", "good morning", "good afternoon",
            "what can you help me with?", "help", "who are you", "what are you", "how are you"
        )

        if tools and not is_greeting:
            payload["tools"] = tools

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            msg = res_data.get("message", {})
            content = msg.get("content")
            raw_tool_calls = msg.get("tool_calls", [])

            tool_calls: List[ToolCall] = []
            for idx, tc in enumerate(raw_tool_calls):
                fn = tc.get("function", {})
                tool_name = (fn.get("name") or "").strip()
                if not tool_name:
                    continue

                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                # If LLM passed null book_id for detail or loan, redirect to search
                if tool_name in ("get_book_details", "loan_book") and (args.get("book_id") in (None, "null", "", 0)):
                    clean_search = re.sub(r"^(?:tell\s+me\s+)?(?:about\s+)?(?:the\s+)?(?:book\s+)?(?:details?\s+)?(?:of\s+)?", "", last_user_text)
                    clean_search = re.sub(r"\s+book(?:\s+detail)?$", "", clean_search).strip()
                    tool_calls.append(
                        ToolCall(
                            id=f"call_ollama_{int(time.time())}_{idx}",
                            type="function",
                            function=FunctionCall(
                                name="get_all_books",
                                arguments={"search_query": clean_search or last_user_text},
                            ),
                        )
                    )
                    continue

                # If LLM passed null / none loan_id for return_book, extract title or loan_id from user text
                if tool_name == "return_book" and (args.get("loan_id") in (None, "null", "None", "")):
                    clean_ret = re.sub(r"^(?:now\s+)?(?:i\s+want\s+to\s+|please\s+)?(?:return|give\s+back)\s+(?:the\s+)?(?:book\s+)?", "", last_user_text).strip().rstrip("!?.").strip()
                    clean_ret = re.sub(r"\s+book$", "", clean_ret).strip()
                    args["loan_id"] = clean_ret or last_user_text

                tool_calls.append(
                    ToolCall(
                        id=f"call_ollama_{int(time.time())}_{idx}",
                        type="function",
                        function=FunctionCall(
                            name=tool_name,
                            arguments=args,
                        ),
                    )
                )

            # Fallback parsing if model returned tool call inside content
            if not tool_calls and content and ("<tool_call>" in content or '"name":' in content):
                tc_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
                if not tc_blocks and content.strip().startswith("{") and content.strip().endswith("}"):
                    tc_blocks = [content.strip()]
                for idx, block in enumerate(tc_blocks):
                    try:
                        parsed_tc = json.loads(block.strip())
                        tname = parsed_tc.get("name") or parsed_tc.get("function", {}).get("name")
                        targs = parsed_tc.get("arguments") or parsed_tc.get("parameters") or parsed_tc.get("function", {}).get("arguments", {})
                        if tname:
                            tool_calls.append(
                                ToolCall(
                                    id=f"call_ollama_parsed_{int(time.time())}_{idx}",
                                    type="function",
                                    function=FunctionCall(name=tname, arguments=targs),
                                )
                            )
                    except Exception:
                        pass

            if content and content.strip().startswith("{") and ('"name": ""' in content or '"name":null' in content):
                content = (
                    f"Hello {self.user_context.username}! I am your AI Library Assistant. "
                    f"I can help you browse all books in the catalog, check book details, loan books, and return books. "
                    f"How can I assist you today?"
                )

            # Preserve explicit negative book ID if user tested negative input
            if tool_calls and last_user_text:
                neg_match = re.search(r"(-\d+)", last_user_text)
                if neg_match:
                    neg_val = int(neg_match.group(1))
                    for tc in tool_calls:
                        if isinstance(tc.function.arguments, dict) and "book_id" in tc.function.arguments:
                            tc.function.arguments["book_id"] = neg_val

            return content, (tool_calls if tool_calls else None)

    def _call_openai(
        self, messages: List[ChatMessage], tools: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
        """Query OpenAI API or compatible endpoint."""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [m.to_dict() for m in messages],
            "temperature": self.config.temperature,
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            choice = res_data["choices"][0]["message"]
            content = choice.get("content")
            raw_tool_calls = choice.get("tool_calls", [])

            tool_calls: List[ToolCall] = []
            for idx, tc in enumerate(raw_tool_calls):
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_oa_{idx}"),
                        type="function",
                        function=FunctionCall(name=fn.get("name"), arguments=args),
                    )
                )

            return content, (tool_calls if tool_calls else None)

    def _fallback_heuristic_step(
        self, messages: List[ChatMessage]
    ) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
        """
        Deterministic heuristic reasoning engine for offline tests, benchmarks,
        and fallback execution when remote LLMs are unavailable.
        """
        last_user_msg = ""
        last_user_idx = -1

        for idx, m in enumerate(messages):
            if m.role == Role.USER and m.content:
                last_user_idx = idx
                last_user_msg = m.content

        # Only collect tool observations that happened during the current turn
        tool_observations: List[Tuple[str, str]] = []
        if last_user_idx != -1:
            for m in messages[last_user_idx + 1:]:
                if m.role == Role.TOOL and m.content:
                    tool_observations.append((m.name or "tool", m.content))

        q = last_user_msg.lower().strip()

        # Handle post-observation transitions
        if tool_observations:
            last_tool_name, last_obs = tool_observations[-1]

            # 1. If we just searched catalog for a loan request
            if last_tool_name in ("get_all_books", "search_books") and any(k in q for k in ("borrow", "checkout", "take out", "loan")):
                # Check if original query was purely generic (e.g. 'I want to borrow book')
                clean_q = re.sub(r"^(?:now\s+)?(?:i\s+want\s+to\s+|please\s+)?(?:loan|borrow|checkout|take\s+out)\s+", "", q).strip().rstrip("!?.").strip()
                is_generic = clean_q in ("", "book", "a book", "books", "the book", "some books")
                if is_generic:
                    return (
                        f"{last_obs}\n\n"
                        f"Which book from the catalog would you like to borrow? Please let me know the book title or Book ID."
                    ), None

                id_match = re.search(r"(?:Book\s+ID|ID):\s*(\d+)", last_obs, re.IGNORECASE)
                if id_match and not any(t[0] in ("loan_book", "borrow_book") for t in tool_observations):
                    bid = int(id_match.group(1))
                    days_match = re.search(r"(\d+)\s+days?", q) or re.search(r"duration\s+(\d+)", q)
                    days = int(days_match.group(1)) if days_match else 14
                    return f"Found matching book in catalog (Book ID {bid}). Proceeding to loan book.", [
                        ToolCall(
                            id="call_auto_loan_book",
                            type="function",
                            function=FunctionCall(name="loan_book", arguments={"book_id": bid, "duration_days": days}),
                        )
                    ]

            # 2. If we just searched catalog for details
            if last_tool_name in ("get_all_books", "search_books") and any(k in q for k in ("detail", "metadata", "info", "shelf", "copies")):
                id_match = re.search(r"(?:Book\s+ID|ID):\s*(\d+)", last_obs, re.IGNORECASE)
                if id_match and not any(t[0] == "get_book_details" for t in tool_observations):
                    bid = int(id_match.group(1))
                    return f"Found matching book. Now retrieving details for book ID {bid}.", [
                        ToolCall(
                            id="call_auto_book_details",
                            type="function",
                            function=FunctionCall(name="get_book_details", arguments={"book_id": bid}),
                        )
                    ]

            if "BOOK_DETAILS" in last_obs:
                return f"{last_obs}", None

            if "PERMISSION_DENIED" in last_obs:
                role_val = self.user_context.role.value.upper()
                return (
                    f"Action was blocked by the security harness:\n{last_obs}\n"
                    f"Your current role is '{role_val}'."
                ), None

            if "INPUT_VALIDATION_ERROR" in last_obs:
                return f"Input validation failed: {last_obs}", None

            if "LOAN_SUCCESS" in last_obs or "BORROW_SUCCESS" in last_obs:
                return f"{last_obs}", None

            if "UNAVAILABLE" in last_obs or "ALL_COPIES_BORROWED" in last_obs:
                return f"Notice: {last_obs}", None

            if "RETURN_SUCCESS" in last_obs or "ALREADY_RETURNED" in last_obs or "LOAN_NOT_FOUND" in last_obs:
                return f"{last_obs}", None

            return f"{last_obs}", None

        # Direct conversational greetings (no tools needed)
        clean_q = q.strip().rstrip("!?.").strip()
        if clean_q in (
            "hello", "hi", "hey", "hello there", "hey there", "greetings",
            "good morning", "good afternoon", "good evening", "how are you",
            "what can you help me with", "help", "who are you", "what are you"
        ):
            return (
                f"Hello {self.user_context.username}! I am your AI Library Assistant.\n"
                f"I can help you browse all books in the catalog, see book details, loan books, and return books.\n"
                f"How can I assist you today?"
            ), None

        # 1. Book Return Intent
        if "return" in q or "give back" in q:
            loan_match = re.search(r"(LOAN-\d+)", last_user_msg.upper())
            if loan_match:
                target_loan = loan_match.group(1)
            else:
                # Extract book title from user query (e.g. 'Now I want to return clean code book' -> 'clean code')
                target = re.sub(r"^(?:now\s+)?(?:i\s+want\s+to\s+|please\s+)?(?:return|give\s+back)\s+(?:the\s+)?(?:book\s+)?", "", q).strip().rstrip("!?.").strip()
                target = re.sub(r"\s+book$", "", target).strip()
                target_loan = target if target else "LOAN-1001"

            return f"Processing return for '{target_loan}'.", [
                ToolCall(
                    id="call_return_book",
                    type="function",
                    function=FunctionCall(name="return_book", arguments={"loan_id": target_loan}),
                )
            ]

        # 2. Book Details Intent
        if any(kw in q for kw in ["detail", "metadata", "full info", "description", "see detail", "see details"]):
            # Check for negative or explicit numeric ID
            id_match = re.search(r"id\s*[:=]?\s*(-?\d+)", q) or re.search(r"book\s+(-?\d+)", q)
            if id_match:
                bid = int(id_match.group(1))
                return f"Reading book details for book ID {bid} from database.", [
                    ToolCall(
                        id="call_book_details",
                        type="function",
                        function=FunctionCall(name="get_book_details", arguments={"book_id": bid}),
                    )
                ]
            else:
                # User asked details by title (e.g. 'Tell me the book detail of system design book' or 'introction to algorithm book detail')
                clean_title = re.sub(r"^(?:now\s+)?(?:tell\s+me\s+)?(?:about\s+)?(?:the\s+)?(?:book\s+)?(?:details?\s+)?(?:of\s+)?", "", q).strip().rstrip("!?.").strip()
                clean_title = re.sub(r"\s+book(?:\s+details?)?$", "", clean_title).strip()
                clean_title = re.sub(r"details?$", "", clean_title).strip()
                if not clean_title:
                    clean_title = "algorithms" if "algorithm" in q else q
                return f"Searching catalog for '{clean_title}' to find the book ID.", [
                    ToolCall(
                        id="call_search_for_details",
                        type="function",
                        function=FunctionCall(name="get_all_books", arguments={"search_query": clean_title}),
                    )
                ]

        # 3. Loaning / Borrowing Intent
        if "loan" in q or "borrow" in q or "checkout" in q or "take out" in q:
            # Check for explicit numeric ID
            id_match = re.search(r"book\s+(?:id\s+)?(\d+)", q) or re.search(r"id\s+(\d+)", q) or re.search(r"\b(\d+)\b", q)
            days_match = re.search(r"(\d+)\s+days?", q) or re.search(r"duration\s+(\d+)", q)
            days = int(days_match.group(1)) if days_match else 14

            if id_match:
                bid = int(id_match.group(1))
                return f"Loaning book ID {bid} for {days} days.", [
                    ToolCall(
                        id="call_loan_book",
                        type="function",
                        function=FunctionCall(name="loan_book", arguments={"book_id": bid, "duration_days": days}),
                    )
                ]
            else:
                # Check if it's generic ("I want to borrow book")
                clean_title = re.sub(r"^(?:now\s+)?(?:i\s+want\s+to\s+|please\s+)?(?:loan|borrow|checkout|take\s+out)\s+(?:the\s+)?(?:book\s+)?", "", q).strip().rstrip("!?.").strip()
                clean_title = re.sub(r"\s+book$", "", clean_title).strip()
                is_generic = clean_title in ("", "book", "a book", "books", "the book", "some books")

                search_term = "" if is_generic else clean_title
                return f"Searching catalog for '{search_term or 'all books'}' to find the book ID.", [
                    ToolCall(
                        id="call_search_for_loan",
                        type="function",
                        function=FunctionCall(name="get_all_books", arguments={"search_query": search_term}),
                    )
                ]

        # 4. Loaning Status / Catalog Listing / Browse Requests
        if any(phrase in q for phrase in [
            "loaning status", "loan status", "loans status", "all status", "status", "inventory",
            "get all", "all book", "all books", "what book", "what books", "which book", "list book", "list all", "show book", "show all",
            "have we", "we have", "have u", "u have", "you have", "in library", "available books", "catalog", "browse"
        ]):
            # Check if there are specific non-stopword tokens in query
            tokens = [t for t in re.findall(r"[a-z0-9]+", q) if t not in STOP_WORDS and len(t) >= 2]
            if not tokens:
                return "Getting all books and availability status from the library catalog.", [
                    ToolCall(
                        id="call_get_all_books",
                        type="function",
                        function=FunctionCall(name="get_all_books", arguments={"search_query": ""}),
                    )
                ]
            else:
                search_term = " ".join(tokens)
                return f"Searching library catalog for '{search_term}'.", [
                    ToolCall(
                        id="call_get_all_books_filtered",
                        type="function",
                        function=FunctionCall(name="get_all_books", arguments={"search_query": search_term}),
                    )
                ]

        # 5. Filtered Search / Topics
        if any(kw in q for kw in ["search", "find", "list", "show", "catalog", "books", "book", "author", "topic"]):
            tokens = [t for t in re.findall(r"[a-z0-9]+", q) if t not in STOP_WORDS and len(t) >= 2]
            search_term = " ".join(tokens) if tokens else last_user_msg
            return f"Searching library catalog for '{search_term}'.", [
                ToolCall(
                    id="call_get_all_books_filtered",
                    type="function",
                    function=FunctionCall(name="get_all_books", arguments={"search_query": search_term}),
                )
            ]

        return (
            f"Hello {self.user_context.username}! I am your AI Library Assistant.\n"
            f"I have 4 tools: get all books, see book details, loan books, and return books.\n"
            f"How can I assist you today?"
        ), None

    def _query_model(
        self, messages: List[ChatMessage]
    ) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
        """Dispatch query to configured backend or fallback safely."""
        tools_schema = self.registry.get_openai_tools()

        if self.config.provider == "openai" or (self.config.provider == "auto" and os.getenv("OPENAI_API_KEY")):
            try:
                return self._call_openai(messages, tools_schema)
            except Exception as e:
                self._log(f"[WARN] OpenAI backend failed ({e}). Falling back...")

        if self.config.provider in ("ollama", "auto"):
            try:
                return self._call_ollama(messages, tools_schema)
            except Exception:
                pass

        return self._fallback_heuristic_step(messages)

    # ReAct Execution Loop

    def run(self, query: str) -> AgentResponse:
        """
        Execute the agent on a user query through the ReAct loop:
        User Request -> LLM -> Tool Proposed -> Safety Harness -> Observation -> LLM -> Final Answer
        """
        start_time = time.time()
        steps: List[AgentStep] = []
        tool_calls_count = 0

        # Reset harness tool call quota per user request turn
        self.harness.tool_calls_count = 0

        self.history.append(ChatMessage(role=Role.USER, content=query))

        self._log(f"\n[USER REQUEST]: '{query}'")
        self._log(f"[USER ROLE]: {self.user_context.role.value.upper()}")
        self._log("-" * 65)

        final_answer = ""
        success = True
        error_msg = None

        for step_idx in range(1, self.config.max_steps + 1):
            self._log(f"\n[Step {step_idx}/{self.config.max_steps}]")

            # 1. Reasoning and tool selection
            thought_or_content, tool_calls = self._query_model(self.history)
            step = AgentStep(step_number=step_idx)

            if thought_or_content:
                step.thought = thought_or_content.strip()
                if tool_calls:
                    self._log(f"[Thought]: {step.thought}")

            # 2. Tool call execution
            if tool_calls:
                self.history.append(
                    ChatMessage(
                        role=Role.ASSISTANT,
                        content=thought_or_content,
                        tool_calls=tool_calls,
                    )
                )

                for tc in tool_calls:
                    tool_calls_count += 1
                    step.action = tc.function.name
                    step.action_input = (
                        tc.function.arguments
                        if isinstance(tc.function.arguments, dict)
                        else {"raw": tc.function.arguments}
                    )

                    self._log(f"[Action Proposed]: {step.action}({step.action_input})")

                    # 3. Intercept via Safety Harness
                    tool_result: ToolResult = self.harness.intercept_and_execute(tc)
                    step.observation = tool_result.content
                    step.permission_allowed = tool_result.success or (tool_result.error_code != "PERMISSION_DENIED")

                    self._log(f"[Observation]:\n{step.observation}")

                    # 4. Record tool observation
                    self.history.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=tool_result.content,
                            name=tool_result.name,
                            tool_call_id=tool_result.tool_call_id,
                        )
                    )

                steps.append(step)
                continue

            else:
                final_answer = thought_or_content or "Execution completed."
                self.history.append(
                    ChatMessage(role=Role.ASSISTANT, content=final_answer)
                )
                steps.append(step)
                # Display [Final Answer] in red so the user clearly distinguishes the conclusion
                self._log(f"\n\033[1;31m[Final Answer]:\033[0m\n {final_answer}")
                break
        else:
            final_answer = "Maximum reasoning loop iterations reached without final answer."
            success = False
            error_msg = "MaxStepsExceeded"
            self._log(f"\n\033[1;31m[Final Answer]:\033[0m\n\033[91m{final_answer}\033[0m")

        total_time = time.time() - start_time
        self._log(f"\nExecution finished in {total_time:.2f}s across {len(steps)} step(s).")
        self._log("=" * 65)

        return AgentResponse(
            query=query,
            final_answer=final_answer,
            steps=steps,
            total_steps=len(steps),
            tool_calls_count=tool_calls_count,
            execution_time_seconds=total_time,
            success=success,
            error=error_msg,
        )

    def chat(self, user_input: str) -> str:
        """Convenience method for interactive chat sessions."""
        res = self.run(user_input)
        return res.final_answer

