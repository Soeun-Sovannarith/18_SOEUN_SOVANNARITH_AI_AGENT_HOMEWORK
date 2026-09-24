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

You have access to database-backed tools for searching catalog books, reading detailed book metadata, checking copy availability and shelf locations, borrowing books, returning loans, and checking active loan status.
Librarians can also permanently delete/remove books from the database catalog.

Operational & Formatting Guidelines:
1. When the patron asks what books are available or asks to browse the catalog (e.g., 'what books do you have?', 'what book that u have', 'list all books', 'show catalog'), call search_books with query='' to list all available books.
2. When searching for specific topics or titles (e.g. 'algorithms', 'clean code'), call search_books with the specific keyword.
3. Present final answers cleanly and neatly formatted with proper spacing and structure. Avoid dumping raw unformatted text or repeating boilerplate phrases.
4. For conversational greetings without inquiries about library inventory (e.g., "Hello", "How are you?"), DO NOT invoke any tools. Respond directly in friendly text.
5. If a regular member/patron asks to delete a book or perform unauthorized administrative actions, call the tool so the application security harness can enforce permissions.
6. When observing tool results, provide a clear, polite, and well-organized final answer to the patron.
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

        # Avoid tool call on greeting queries
        is_greeting = False
        if len(messages) <= 2:
            last_text = (messages[-1].content or "").strip().lower()
            if last_text in (
                "hello", "hi", "hey", "hello!", "hi!",
                "what can you help me with?", "help"
            ):
                is_greeting = True

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

                tool_calls.append(
                    ToolCall(
                        id=f"call_ollama_{int(time.time())}_{idx}",
                        type="function",
                        function=FunctionCall(
                            name=tool_name,
                            arguments=fn.get("arguments", {}),
                        ),
                    )
                )

            if content and content.strip().startswith("{") and ('"name": ""' in content or '"name":null' in content):
                content = (
                    f"Hello {self.user_context.username}! I am your AI Library Assistant. "
                    f"I can help you search the catalog for books, check copy availability, borrow titles, and manage loans. "
                    f"How can I assist you today?"
                )

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
        tool_observations: List[Tuple[str, str]] = []

        for m in messages:
            if m.role == Role.USER and m.content:
                last_user_msg = m.content
            elif m.role == Role.TOOL and m.content:
                tool_observations.append((m.name or "tool", m.content))

        q = last_user_msg.lower()

        # Handle post-observation transitions
        if tool_observations:
            last_tool_name, last_obs = tool_observations[-1]

            if last_tool_name == "search_books" and any(k in q for k in ("check", "available", "availability", "copies", "shelf", "stock")):
                id_match = re.search(r"(?:Book\s+ID|ID):\s*(\d+)", last_obs, re.IGNORECASE)
                if id_match and not any(t[0] == "check_availability" for t in tool_observations):
                    bid = int(id_match.group(1))
                    return f"Found matching books. Now checking copy availability and location for book ID {bid}.", [
                        ToolCall(
                            id="call_auto_check_avail",
                            type="function",
                            function=FunctionCall(name="check_availability", arguments={"book_id": bid}),
                        )
                    ]

            if "BOOK_DETAILS" in last_obs:
                return f"{last_obs}", None

            if "PERMISSION_DENIED" in last_obs:
                role_val = self.user_context.role.value.upper()
                return (
                    f"Action was blocked by the security harness:\n{last_obs}\n"
                    f"Your current role is '{role_val}'. "
                    f"This operation requires librarian / administrative privileges."
                ), None

            if "INPUT_VALIDATION_ERROR" in last_obs:
                return f"Input validation failed: {last_obs}", None

            if "BORROW_SUCCESS" in last_obs:
                return f"{last_obs}", None

            if "UNAVAILABLE" in last_obs or "ALL_COPIES_BORROWED" in last_obs:
                return f"Notice: {last_obs}", None

            if "RETURN_SUCCESS" in last_obs:
                return f"{last_obs}", None

            if "DELETE_SUCCESS" in last_obs:
                return f"Administrative action complete: {last_obs}", None

            return f"{last_obs}", None

        # Intent routing for step 1
        if "delete" in q or "remove" in q:
            id_match = re.search(r"(-?\d+)", last_user_msg)
            bid = int(id_match.group(1)) if id_match else 1
            return f"Attempting to delete book ID {bid}.", [
                ToolCall(
                    id="call_delete_book",
                    type="function",
                    function=FunctionCall(name="delete_book", arguments={"book_id": bid}),
                )
            ]

        if "return" in q and ("loan" in q or "book" in q or "loan-" in q):
            loan_match = re.search(r"(LOAN-\d+)", last_user_msg.upper())
            loan_id = loan_match.group(1) if loan_match else "LOAN-1001"
            return f"Processing return for {loan_id}.", [
                ToolCall(
                    id="call_return_book",
                    type="function",
                    function=FunctionCall(name="return_book", arguments={"loan_id": loan_id}),
                )
            ]

        if "loan" in q and ("status" in q or "due" in q or "check" in q):
            loan_match = re.search(r"(LOAN-\d+)", last_user_msg.upper())
            loan_id = loan_match.group(1) if loan_match else "LOAN-1001"
            return f"Looking up loan status for {loan_id}.", [
                ToolCall(
                    id="call_loan_status",
                    type="function",
                    function=FunctionCall(name="get_loan_status", arguments={"loan_id": loan_id}),
                )
            ]

        if any(kw in q for kw in ["detail", "metadata", "full info", "description", "get details"]) and re.search(r"id\s*[:=]?\s*(\d+)", q):
            id_match = re.search(r"id\s*[:=]?\s*(\d+)", q) or re.search(r"(\d+)", q)
            bid = int(id_match.group(1)) if id_match else 1
            return f"Reading book details for book ID {bid} from database.", [
                ToolCall(
                    id="call_book_details",
                    type="function",
                    function=FunctionCall(name="get_book_details", arguments={"book_id": bid}),
                )
            ]

        if "borrow" in q or "checkout" in q or "take out" in q:
            id_match = re.search(r"book\s+(?:id\s+)?(\d+)", q) or re.search(r"id\s+(\d+)", q) or re.search(r"(\d+)", q)
            days_match = re.search(r"(\d+)\s+days?", q) or re.search(r"duration\s+(\d+)", q)

            bid = int(id_match.group(1)) if id_match else 2
            days = int(days_match.group(1)) if days_match else 14

            return f"Borrowing book ID {bid} for {days} days.", [
                ToolCall(
                    id="call_borrow_book",
                    type="function",
                    function=FunctionCall(name="borrow_book", arguments={"book_id": bid, "duration_days": days}),
                )
            ]

        if ("availability" in q or "available" in q or "copies" in q or "shelf" in q or "location" in q) and re.search(r"id\s*[:=]?\s*(-?\d+)", q):
            id_match = re.search(r"id\s*[:=]?\s*(-?\d+)", q)
            bid = int(id_match.group(1)) if id_match else 1
            return f"Checking availability for book ID {bid}.", [
                ToolCall(
                    id="call_check_avail",
                    type="function",
                    function=FunctionCall(name="check_availability", arguments={"book_id": bid}),
                )
            ]

        # Broad catalog listing / browse requests (e.g. 'what book that we have ?', 'list books', 'show all books')
        if any(phrase in q for phrase in [
            "what book", "what books", "which book", "list book", "list all", "show book", "show all",
            "have we", "we have", "have u", "u have", "you have", "in library", "available books", "catalog", "browse"
        ]) and not any(specific in q for specific in ["clean code", "algorithm", "design pattern", "ai", "artificial intelligence", "data"]):
            return "Searching library catalog for all available books.", [
                ToolCall(
                    id="call_search_books_all",
                    type="function",
                    function=FunctionCall(name="search_books", arguments={"query": ""}),
                )
            ]

        if any(kw in q for kw in ["search", "find", "list", "show", "catalog", "books", "book", "author", "topic"]):
            if "clean code" in q:
                kw = "clean code"
            elif "algorithm" in q:
                kw = "algorithms"
            elif "design pattern" in q:
                kw = "design patterns"
            elif "ai" in q or "artificial intelligence" in q:
                kw = "artificial intelligence"
            elif "data" in q or "distributed" in q:
                kw = "designing data-intensive"
            else:
                kw = last_user_msg

            return f"Searching library catalog for '{kw}'.", [
                ToolCall(
                    id="call_search_books",
                    type="function",
                    function=FunctionCall(name="search_books", arguments={"query": kw}),
                )
            ]

        if "availability" in q or "available" in q:
            return f"Searching catalog for books.", [
                ToolCall(
                    id="call_search_avail",
                    type="function",
                    function=FunctionCall(name="search_books", arguments={"query": "algorithms"}),
                )
            ]

        return (
            f"Hello {self.user_context.username}! I am your AI Library Assistant (Role: {self.user_context.role.value.upper()}). "
            f"I can help you search the catalog for books, read book details from the database, check copy availability, borrow books, return loans, and check due dates. "
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
                self._log(f"\n[Final Answer]:\n{final_answer}")
                break
        else:
            final_answer = "Maximum reasoning loop iterations reached without final answer."
            success = False
            error_msg = "MaxStepsExceeded"

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
