"""
Tool implementations and PostgreSQL database access for the Library AI Agent.
Provides direct database connectivity, atomic loan transactions, tool registry,
and explicit Pydantic schema validation for PostgreSQL.
"""

from __future__ import annotations
import inspect
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import psycopg2
import psycopg2.extras
from pydantic import BaseModel, ValidationError

from schemas import (
    GetAllBooksInput,
    GetBookDetailsInput,
    LoanBookInput,
    ReturnBookInput,
    RiskLevel,
    ToolDefinition,
)

# Load environment variables if dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# PostgreSQL Database Manager

class DatabaseManager:
    """Direct PostgreSQL connection and query manager."""

    def __init__(self) -> None:
        self.db_url = os.getenv("DATABASE_URL")
        self.pg_host = os.getenv("DB_HOST", "localhost")
        self.pg_port = os.getenv("DB_PORT", "5432")
        self.pg_name = os.getenv("DB_NAME", "library_db")
        self.pg_user = os.getenv("DB_USER", "postgres")
        self.pg_password = os.getenv("DB_PASSWORD", "")

    def get_connection(self) -> psycopg2.extensions.connection:
        """Establish and return a connection to the PostgreSQL database."""
        if self.db_url:
            return psycopg2.connect(self.db_url)
        return psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            dbname=self.pg_name,
            user=self.pg_user,
            password=self.pg_password,
        )

    def query_all(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        """Execute SELECT query and return all matching rows as dictionaries."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        """Execute SELECT query and return a single row dictionary or None."""
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    def execute_write(self, sql: str, params: Tuple[Any, ...] = ()) -> int:
        """Execute INSERT / UPDATE / DELETE statement and return modified row count."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                conn.commit()
                return cur.rowcount

    def _load_sql_file(self, filename: str = "db.sql") -> str:
        """Load SQL script from project directory."""
        sql_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(sql_path):
            sql_path = filename
        if os.path.exists(sql_path):
            with open(sql_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def init_database(self) -> None:
        """Initialize PostgreSQL tables and seed data if not present."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.books')")
                res = cur.fetchone()
                table_exists = res and res[0] is not None
                if not table_exists:
                    sql_content = self._load_sql_file("db.sql")
                    if sql_content:
                        cur.execute(sql_content)
                        conn.commit()

    def reset_database(self) -> None:
        """Reset PostgreSQL tables and reload seed data from db.sql."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                sql_content = self._load_sql_file("db.sql")
                if sql_content:
                    cur.execute(sql_content)
                    conn.commit()


# Global database instance
db = DatabaseManager()
db.init_database()


def reset_db() -> None:
    """Reset PostgreSQL database state to default seeds."""
    db.reset_database()


# Tool Registry

class ToolRegistry:
    """Registry managing tool functions, Pydantic schemas, and risk tiers."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Type[BaseModel]] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    ALIAS_MAP = {
        "search_books": "get_all_books",
        "borrow_book": "loan_book",
        "get_all_book": "get_all_books",
        "see_book_detail": "get_book_details",
        "see_book_details": "get_book_details",
        "loan_books": "loan_book",
    }

    def register(
        self,
        schema: Type[BaseModel],
        risk_level: RiskLevel = RiskLevel.GREEN,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a tool function with a Pydantic input schema."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            doc = description or inspect.getdoc(fn) or f"Execute {tool_name}"
            json_schema = schema.model_json_schema()
            parameters = {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            }
            tool_def = ToolDefinition(
                name=tool_name,
                description=doc.strip().split("\n")[0],
                parameters=parameters,
                risk_level=risk_level,
            )
            self._tools[tool_name] = fn
            self._schemas[tool_name] = schema
            self._definitions[tool_name] = tool_def
            return fn

        return decorator

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        resolved = self.ALIAS_MAP.get(name, name)
        return self._tools.get(resolved)

    def get_schema(self, name: str) -> Optional[Type[BaseModel]]:
        resolved = self.ALIAS_MAP.get(name, name)
        return self._schemas.get(resolved)

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        resolved = self.ALIAS_MAP.get(name, name)
        return self._definitions.get(resolved)

    def get_definitions(self) -> List[ToolDefinition]:
        return list(self._definitions.values())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        return [td.to_openai_dict() for td in self._definitions.values()]

    def validate_inputs(self, name: str, raw_args: Union[Dict[str, Any], str]) -> Tuple[Optional[BaseModel], Optional[str]]:
        """Validate raw arguments against the registered Pydantic input schema."""
        resolved = self.ALIAS_MAP.get(name, name)
        schema = self._schemas.get(resolved)
        if not schema:
            return None, f"No schema registered for tool '{name}'"

        parsed_dict = raw_args
        if isinstance(raw_args, str):
            try:
                parsed_dict = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as err:
                return None, f"JSONDecodeError: Invalid JSON payload ({str(err)})"

        if not isinstance(parsed_dict, dict):
            parsed_dict = {}

        cleaned_dict = dict(parsed_dict)
        if "duration_days" in cleaned_dict and cleaned_dict["duration_days"] in (None, "None", "null", ""):
            cleaned_dict["duration_days"] = 14
        if "query" in cleaned_dict and "search_query" not in cleaned_dict:
            cleaned_dict["search_query"] = cleaned_dict.pop("query")
        for num_field in ("book_id", "duration_days"):
            if num_field in cleaned_dict and isinstance(cleaned_dict[num_field], str):
                num_match = re.search(r"(-?\d+)", cleaned_dict[num_field].strip())
                if num_match:
                    try:
                        cleaned_dict[num_field] = int(num_match.group(1))
                    except ValueError:
                        pass
        parsed_dict = cleaned_dict

        try:
            validated = schema(**parsed_dict)
            return validated, None
        except ValidationError as val_err:
            error_details = []
            for err in val_err.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                msg = err["msg"]
                error_details.append(f"Field '{field}': {msg}")
            return None, f"InputValidationError: {'; '.join(error_details)}"


# Global registry instance
registry = ToolRegistry()


STOP_WORDS = {
    "what", "which", "who", "where", "when", "how", "why",
    "book", "books", "that", "u", "you", "your", "have", "do", "did", "does",
    "is", "are", "was", "were", "be", "been", "a", "an", "the", "in", "on", "at",
    "to", "for", "of", "with", "from", "by", "about", "into", "through",
    "me", "us", "him", "her", "them", "my", "our", "their",
    "show", "list", "get", "find", "search", "give", "tell", "display",
    "all", "any", "some", "can", "could", "would", "should", "please",
    "library", "catalog", "available", "store", "database", "there"
}


# 1. Get All Books Tool
@registry.register(
    schema=GetAllBooksInput,
    risk_level=RiskLevel.GREEN,
    name="get_all_books",
    description="Get all books from library catalog, browse inventory, or search titles/authors/topics."
)
def get_all_books(search_query: Optional[str] = "", category: Optional[str] = None) -> str:
    """Retrieve and list books in PostgreSQL database with optional search keyword or category."""
    raw_q = search_query or ""
    q = raw_q.lower().strip()
    if category and category.lower() in ("null", "none", ""):
        category = None

    cleaned_tokens = re.findall(r"[a-z0-9]+", q)
    meaningful_keywords = [t for t in cleaned_tokens if t not in STOP_WORDS and len(t) >= 2]

    if not meaningful_keywords or q in ("", "all", "books", "catalog", "null", "none", "loaning status", "status"):
        if category:
            sql = """
                SELECT id, title, author, category, total_copies, available_copies, location, description
                FROM books
                WHERE LOWER(category) LIKE %s
                ORDER BY id ASC
            """
            rows = db.query_all(sql, (f"%{category.lower()}%",))
        else:
            sql = """
                SELECT id, title, author, category, total_copies, available_copies, location, description
                FROM books
                ORDER BY id ASC
            """
            rows = db.query_all(sql)
    else:
        where_clauses = []
        params: List[Any] = []
        for kw in meaningful_keywords:
            where_clauses.append("(LOWER(title) LIKE %s OR LOWER(author) LIKE %s OR LOWER(description) LIKE %s OR LOWER(category) LIKE %s)")
            params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%"])

        sql_where = " AND ".join(where_clauses)
        if category:
            sql_where += " AND LOWER(category) LIKE %s"
            params.append(f"%{category.lower()}%")

        sql = f"""
            SELECT id, title, author, category, total_copies, available_copies, location, description
            FROM books
            WHERE {sql_where}
            ORDER BY id ASC
        """
        rows = db.query_all(sql, tuple(params))

        if not rows and len(meaningful_keywords) > 1:
            sql_or_where = " OR ".join(where_clauses)
            if category:
                sql_or_where = f"({sql_or_where}) AND LOWER(category) LIKE %s"
                params_or = list(params) + [f"%{category.lower()}%"]
            else:
                params_or = list(params)
            sql_fallback = f"""
                SELECT id, title, author, category, total_copies, available_copies, location, description
                FROM books
                WHERE {sql_or_where}
                ORDER BY id ASC
            """
            rows = db.query_all(sql_fallback, tuple(params_or))

        if not rows:
            sql_phrase = """
                SELECT id, title, author, category, total_copies, available_copies, location, description
                FROM books
                WHERE (LOWER(title) LIKE %s OR LOWER(author) LIKE %s OR LOWER(description) LIKE %s OR LOWER(category) LIKE %s)
                ORDER BY id ASC
            """
            rows = db.query_all(sql_phrase, (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"))

    if not rows:
        return f"No books found matching query '{search_query}' in database."

    header = "Library Catalog (All Books):" if not meaningful_keywords else f"Found {len(rows)} book(s) in catalog:"
    output = [header, ""]
    for idx, b in enumerate(rows, 1):
        avail = b["available_copies"]
        total = b["total_copies"]
        status = f"{avail}/{total} available" if avail > 0 else "0 available (ALL COPIES BORROWED)"
        output.append(f"{idx}. {b['title']}")
        output.append(f"   - Book ID:  {b['id']}")
        output.append(f"   - Author:   {b['author']}")
        output.append(f"   - Category: {b['category']}")
        output.append(f"   - Status:   {status}")
        output.append(f"   - Location: {b['location']}")
        output.append("")

    return "\n".join(output).rstrip()


# 2. Get Book Details Tool
@registry.register(
    schema=GetBookDetailsInput,
    risk_level=RiskLevel.GREEN,
    name="get_book_details",
    description="See full book details, metadata, available copies, and shelf location by Book ID."
)
def get_book_details(book_id: int) -> str:
    """Read full book details and metadata from PostgreSQL database by book ID."""
    sql = """
        SELECT id, title, author, category, isbn, total_copies, available_copies, location, description
        FROM books
        WHERE id = %s
    """
    b = db.query_one(sql, (book_id,))
    if not b:
        return f"BOOK_NOT_FOUND: Book ID {book_id} was not found in the database."

    avail = b["available_copies"]
    total = b["total_copies"]
    status = f"{avail}/{total} available" if avail > 0 else "0 available (ALL COPIES BORROWED)"

    return (
        f"[BOOK_DETAILS]\n"
        f"Title:       {b['title']}\n"
        f"Book ID:     {b['id']}\n"
        f"Author:      {b['author']}\n"
        f"Category:    {b['category']}\n"
        f"ISBN:        {b['isbn']}\n"
        f"Status:      {status}\n"
        f"Location:    {b['location']}\n"
        f"Description: {b['description']}"
    )


# 3. Loan Book Tool
@registry.register(
    schema=LoanBookInput,
    risk_level=RiskLevel.YELLOW,
    name="loan_book",
    description="Loan a book copy, update inventory in database, and record a new active loan entry."
)
def loan_book(book_id: int, duration_days: int = 14) -> str:
    """Loan a book copy in PostgreSQL, update inventory, and record loan entry."""
    book = db.query_one("SELECT id, title, available_copies, total_copies FROM books WHERE id = %s", (book_id,))
    if not book:
        return f"BOOK_NOT_FOUND: Book ID {book_id} does not exist in database."

    if book["available_copies"] <= 0:
        return (
            f"ALL_COPIES_BORROWED: Cannot loan '{book['title']}' (ID: {book_id}). "
            f"All {book['total_copies']} copies are currently checked out by other patrons."
        )

    # Decrement available copies
    db.execute_write("UPDATE books SET available_copies = available_copies - 1 WHERE id = %s", (book_id,))

    # Create new sequential loan ID
    existing_loans = db.query_all("SELECT loan_id FROM loans")
    loan_numbers = []
    for l_row in existing_loans:
        lid = str(l_row.get("loan_id", ""))
        parts = lid.split("-")
        if len(parts) > 1 and parts[1].isdigit():
            loan_numbers.append(int(parts[1]))
    next_num = max(loan_numbers, default=1000) + 1
    loan_id = f"LOAN-{next_num}"
    due_date = (datetime.now() + timedelta(days=duration_days)).strftime("%Y-%m-%d")

    db.execute_write("""
        INSERT INTO loans (loan_id, book_id, book_title, user_id, duration_days, due_date, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (loan_id, book_id, book["title"], "user_001", duration_days, due_date, "Active"))

    updated_book = db.query_one("SELECT available_copies FROM books WHERE id = %s", (book_id,))
    remaining = updated_book["available_copies"] if updated_book else 0

    return (
        f"[LOAN_SUCCESS]\n"
        f"Book:        '{book['title']}' (Book ID: {book_id})\n"
        f"Loan ID:     {loan_id}\n"
        f"Due Date:    {due_date} ({duration_days} days)\n"
        f"Status:      Active\n"
        f"Remaining:   {remaining} copy/copies available on shelf."
    )


# 4. Return Book Tool
@registry.register(
    schema=ReturnBookInput,
    risk_level=RiskLevel.YELLOW,
    name="return_book",
    description="Return a loaned book by Loan ID, Book Title, or Book ID, and restore available copies in database."
)
def return_book(loan_id: str) -> str:
    """Return a borrowed/loaned book in PostgreSQL by Loan ID, Book Title, or Book ID."""
    clean_target = (loan_id or "").strip()
    clean_id = clean_target.upper()

    # 1. Try finding loan by direct loan_id (e.g. LOAN-1001)
    loan = db.query_one("SELECT loan_id, book_id, book_title, status FROM loans WHERE UPPER(loan_id) = %s", (clean_id,))

    # 2. If loan_id is empty or null, check if single active loan exists
    if not loan and clean_target.lower() in ("none", "null", ""):
        active_loans = db.query_all("SELECT loan_id, book_id, book_title, status FROM loans WHERE status = 'Active' ORDER BY due_date ASC")
        if len(active_loans) == 1:
            loan = active_loans[0]

    # 3. If not found by exact loan_id, search for active loan by book ID or title
    if not loan:
        if clean_target.isdigit():
            loan = db.query_one(
                "SELECT loan_id, book_id, book_title, status FROM loans WHERE book_id = %s AND status = 'Active' ORDER BY due_date ASC",
                (int(clean_target),)
            )
        else:
            words = [w for w in re.findall(r"[a-z0-9]+", clean_target.lower()) if w not in STOP_WORDS and len(w) >= 2]
            if words:
                where_clauses = ["LOWER(book_title) LIKE %s" for _ in words]
                params = [f"%{w}%" for w in words]
                sql = f"SELECT loan_id, book_id, book_title, status FROM loans WHERE status = 'Active' AND ({' AND '.join(where_clauses)}) ORDER BY due_date ASC"
                loan = db.query_one(sql, tuple(params))
                if not loan and len(words) > 1:
                    sql_or = f"SELECT loan_id, book_id, book_title, status FROM loans WHERE status = 'Active' AND ({' OR '.join(where_clauses)}) ORDER BY due_date ASC"
                    loan = db.query_one(sql_or, tuple(params))

    # 4. If no active loan found, check if already returned
    if not loan:
        if clean_target.isdigit():
            returned_loan = db.query_one(
                "SELECT loan_id, book_id, book_title, status FROM loans WHERE book_id = %s ORDER BY due_date DESC",
                (int(clean_target),)
            )
        else:
            words = [w for w in re.findall(r"[a-z0-9]+", clean_target.lower()) if w not in STOP_WORDS and len(w) >= 2]
            if words:
                where_clauses = ["LOWER(book_title) LIKE %s" for _ in words]
                params = [f"%{w}%" for w in words]
                sql = f"SELECT loan_id, book_id, book_title, status FROM loans WHERE ({' AND '.join(where_clauses)}) ORDER BY due_date DESC"
                returned_loan = db.query_one(sql, tuple(params))
            else:
                returned_loan = None

        if returned_loan:
            return f"ALREADY_RETURNED: Book for Loan {returned_loan['loan_id']} ('{returned_loan['book_title']}') has already been returned."

        return f"LOAN_NOT_FOUND: No active loan record found matching '{loan_id}' in the database."

    if loan["status"] == "Returned":
        return f"ALREADY_RETURNED: Book for Loan {loan['loan_id']} ('{loan['book_title']}') has already been returned."

    db.execute_write("UPDATE loans SET status = 'Returned' WHERE loan_id = %s", (loan["loan_id"],))
    db.execute_write("UPDATE books SET available_copies = available_copies + 1 WHERE id = %s", (loan["book_id"],))

    return (
        f"[RETURN_SUCCESS]\n"
        f"Loan ID:     {loan['loan_id']}\n"
        f"Book Title:  '{loan['book_title']}'\n"
        f"Status:      Returned\n"
        f"Message:     Book successfully returned to the library inventory."
    )
