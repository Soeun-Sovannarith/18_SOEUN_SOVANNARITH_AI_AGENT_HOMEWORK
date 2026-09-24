"""
Tool implementations and database access for the Library AI Agent.
Supports PostgreSQL database (via DATABASE_URL or db.sql) with fallback to SQLite,
atomic loan transactions, tool registry, and explicit Pydantic schema mappings.
"""

from __future__ import annotations
import inspect
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, ValidationError

from schemas import (
    BorrowBookInput,
    CheckAvailabilityInput,
    DeleteBookInput,
    GetBookDetailsInput,
    GetLoanStatusInput,
    ReturnBookInput,
    RiskLevel,
    SearchBooksInput,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

# Load environment variables if dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for psycopg2 availability
try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

import sqlite3


# Database Adapter Layer (PostgreSQL with SQLite fallback)

class DatabaseManager:
    """Unified database connection manager supporting PostgreSQL and SQLite."""

    def __init__(self) -> None:
        self.db_url = os.getenv("DATABASE_URL")
        self.pg_host = os.getenv("DB_HOST")
        self.pg_name = os.getenv("DB_NAME")
        self.pg_user = os.getenv("DB_USER")
        self.pg_password = os.getenv("DB_PASSWORD")
        self.pg_port = os.getenv("DB_PORT", "5432")
        self.sqlite_path = os.getenv("LIBRARY_DB_PATH", "library.db")

    @property
    def is_postgres_configured(self) -> bool:
        """Check if PostgreSQL credentials or URL are provided and driver is available."""
        if not PSYCOPG2_AVAILABLE:
            return False
        return bool(self.db_url or (self.pg_host and self.pg_name and self.pg_user))

    def get_connection(self) -> Any:
        """Return active database connection."""
        if self.is_postgres_configured:
            try:
                if self.db_url:
                    conn = psycopg2.connect(self.db_url)
                else:
                    conn = psycopg2.connect(
                        host=self.pg_host,
                        port=self.pg_port,
                        dbname=self.pg_name,
                        user=self.pg_user,
                        password=self.pg_password or "",
                    )
                return conn
            except Exception:
                # Fall back to SQLite if PostgreSQL connection fails
                pass

        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def query_all(self, sql_pg: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return list of dictionaries."""
        conn = self.get_connection()
        try:
            is_pg = PSYCOPG2_AVAILABLE and isinstance(conn, psycopg2.extensions.connection)
            if is_pg:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql_pg, params)
                    return [dict(row) for row in cur.fetchall()]
            else:
                sql_sqlite = sql_pg.replace("%s", "?")
                cur = conn.cursor()
                cur.execute(sql_sqlite, params)
                return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def query_one(self, sql_pg: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return single row dictionary or None."""
        rows = self.query_all(sql_pg, params)
        return rows[0] if rows else None

    def execute_write(self, sql_pg: str, params: Tuple[Any, ...] = ()) -> int:
        """Execute INSERT / UPDATE / DELETE statement and return row count."""
        conn = self.get_connection()
        try:
            is_pg = PSYCOPG2_AVAILABLE and isinstance(conn, psycopg2.extensions.connection)
            if is_pg:
                with conn.cursor() as cur:
                    cur.execute(sql_pg, params)
                    conn.commit()
                    return cur.rowcount
            else:
                sql_sqlite = sql_pg.replace("%s", "?")
                cur = conn.cursor()
                cur.execute(sql_sqlite, params)
                conn.commit()
                return cur.rowcount
        finally:
            conn.close()

    def init_database(self) -> None:
        """Initialize schema and seed initial data if tables are empty."""
        conn = self.get_connection()
        try:
            is_pg = PSYCOPG2_AVAILABLE and isinstance(conn, psycopg2.extensions.connection)
            if is_pg:
                # Run db.sql script if available for PostgreSQL
                sql_file = os.path.join(os.path.dirname(__file__), "db.sql")
                if os.path.exists(sql_file):
                    with open(sql_file, "r", encoding="utf-8") as f:
                        sql_content = f.read()
                    with conn.cursor() as cur:
                        cur.execute(sql_content)
                        conn.commit()
            else:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS books (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        author TEXT NOT NULL,
                        category TEXT NOT NULL,
                        isbn TEXT UNIQUE,
                        total_copies INTEGER NOT NULL,
                        available_copies INTEGER NOT NULL,
                        location TEXT NOT NULL,
                        description TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS loans (
                        loan_id TEXT PRIMARY KEY,
                        book_id INTEGER NOT NULL,
                        book_title TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        duration_days INTEGER NOT NULL,
                        due_date TEXT NOT NULL,
                        status TEXT NOT NULL,
                        FOREIGN KEY (book_id) REFERENCES books (id)
                    )
                """)
                cur.execute("SELECT COUNT(*) as cnt FROM books")
                if cur.fetchone()["cnt"] == 0:
                    self._seed_sqlite(cur)
                conn.commit()
        finally:
            conn.close()

    def _seed_sqlite(self, cur: Any) -> None:
        """Seed SQLite tables."""
        initial_books = [
            (1, "Clean Code: A Handbook of Agile Software Craftsmanship", "Robert C. Martin", "Software Engineering", "978-0132350884", 3, 3, "Shelf A-12", "Essential principles and best practices for writing clean, maintainable code."),
            (2, "Design Patterns: Elements of Reusable Object-Oriented Software", "Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides", "Computer Science", "978-0201633610", 2, 2, "Shelf B-04", "The classic guide to 23 foundational software design patterns."),
            (3, "Introduction to Algorithms (CLRS)", "Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein", "Algorithms", "978-0262033848", 4, 0, "Shelf C-01", "Comprehensive textbook covering modern algorithm analysis and data structures."),
            (4, "Artificial Intelligence: A Modern Approach", "Stuart Russell, Peter Norvig", "Artificial Intelligence", "978-0136042594", 5, 5, "Shelf A-03", "The authoritative textbook on AI algorithms, intelligent agents, and machine learning."),
            (5, "The Pragmatic Programmer: Your Journey To Mastery", "David Thomas, Andrew Hunt", "Software Engineering", "978-0135957059", 2, 1, "Shelf B-09", "Practical advice covering software craftsmanship and testing."),
            (6, "Designing Data-Intensive Applications", "Martin Kleppmann", "Distributed Systems", "978-1449373320", 4, 4, "Shelf D-15", "In-depth architecture guide exploring databases, replication, and distributed systems.")
        ]
        cur.executemany("""
            INSERT INTO books (id, title, author, category, isbn, total_copies, available_copies, location, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, initial_books)

        cur.execute("""
            INSERT INTO loans (loan_id, book_id, book_title, user_id, duration_days, due_date, status)
            VALUES ('LOAN-1001', 3, 'Introduction to Algorithms (CLRS)', 'user_001', 14, '2026-10-08', 'Active')
        """)

    def reset_database(self) -> None:
        """Reset database to fresh state."""
        conn = self.get_connection()
        try:
            is_pg = PSYCOPG2_AVAILABLE and isinstance(conn, psycopg2.extensions.connection)
            if is_pg:
                sql_file = os.path.join(os.path.dirname(__file__), "db.sql")
                if os.path.exists(sql_file):
                    with open(sql_file, "r", encoding="utf-8") as f:
                        sql_content = f.read()
                    with conn.cursor() as cur:
                        cur.execute(sql_content)
                        conn.commit()
            else:
                cur = conn.cursor()
                cur.execute("DROP TABLE IF EXISTS loans")
                cur.execute("DROP TABLE IF EXISTS books")
                conn.commit()
                self.init_database()
        finally:
            conn.close()


# Global database manager instance
db = DatabaseManager()
db.init_database()


def reset_db() -> None:
    """Module-level reset function."""
    db.reset_database()


# Tool Registry

class ToolRegistry:
    """Registry managing tool functions, Pydantic schemas, and risk tiers."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Type[BaseModel]] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

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
        return self._tools.get(name)

    def get_schema(self, name: str) -> Optional[Type[BaseModel]]:
        return self._schemas.get(name)

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        return self._definitions.get(name)

    def get_definitions(self) -> List[ToolDefinition]:
        return list(self._definitions.values())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        return [td.to_openai_dict() for td in self._definitions.values()]

    def validate_inputs(self, name: str, raw_args: Union[Dict[str, Any], str]) -> Tuple[Optional[BaseModel], Optional[str]]:
        """Validate raw arguments against the registered Pydantic input schema."""
        schema = self._schemas.get(name)
        if not schema:
            return None, f"No schema registered for tool '{name}'"

        parsed_dict = raw_args
        if isinstance(raw_args, str):
            try:
                parsed_dict = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as err:
                return None, f"JSONDecodeError: Invalid JSON payload ({str(err)})"

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


# Database-Backed Tool Implementations

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


@registry.register(schema=SearchBooksInput, risk_level=RiskLevel.GREEN)
def search_books(query: str = "", category: Optional[str] = None) -> str:
    """Search books in the database catalog by title, author, keyword, or category."""
    raw_q = query or ""
    q = raw_q.lower().strip()
    if category and category.lower() in ("null", "none", ""):
        category = None

    # Clean punctuation and tokenize query words
    cleaned_tokens = re.findall(r"[a-z0-9]+", q)
    meaningful_keywords = [t for t in cleaned_tokens if t not in STOP_WORDS]

    # If query is empty or contains only generic browse words
    if not meaningful_keywords or q in ("", "all", "books", "catalog", "null", "none"):
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
        # Match any meaningful keyword across title, author, category, description
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

        # Fallback to phrase search if strict conjunction returned nothing
        if not rows:
            sql_fallback = """
                SELECT id, title, author, category, total_copies, available_copies, location, description
                FROM books
                WHERE (LOWER(title) LIKE %s OR LOWER(author) LIKE %s OR LOWER(description) LIKE %s OR LOWER(category) LIKE %s)
                ORDER BY id ASC
            """
            params_fallback = (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%")
            rows = db.query_all(sql_fallback, params_fallback)

    if not rows:
        return f"No books found matching query '{query}' in database."

    header = "Available Books in Library Catalog:" if not meaningful_keywords else f"Found {len(rows)} book(s) matching your search:"
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


@registry.register(schema=CheckAvailabilityInput, risk_level=RiskLevel.GREEN)
def check_availability(book_id: int) -> str:
    """Read book availability, remaining copies, and shelf location from the database."""
    sql = "SELECT id, title, available_copies, total_copies, location FROM books WHERE id = %s"
    book = db.query_one(sql, (book_id,))

    if not book:
        return f"BOOK_NOT_FOUND: Book ID {book_id} does not exist in database."

    avail = book["available_copies"]
    total = book["total_copies"]
    if avail == 0:
        return (
            f"UNAVAILABLE: Book '{book['title']}' (ID: {book_id}) has 0 copies available "
            f"(all {total} copies currently borrowed). Location: {book['location']}."
        )

    return (
        f"AVAILABLE: Book '{book['title']}' (ID: {book_id}) has {avail} of {total} copy/copies "
        f"available for checkout. Location: {book['location']}."
    )


@registry.register(schema=GetBookDetailsInput, risk_level=RiskLevel.GREEN)
def get_book_details(book_id: int) -> str:
    """Read full book details and metadata from database by book ID."""
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


@registry.register(schema=BorrowBookInput, risk_level=RiskLevel.YELLOW)
def borrow_book(book_id: int, duration_days: int = 14) -> str:
    """Borrow a book copy, update inventory in database, and record loan entry."""
    book = db.query_one("SELECT id, title, available_copies, total_copies FROM books WHERE id = %s", (book_id,))
    if not book:
        return f"BOOK_NOT_FOUND: Book ID {book_id} does not exist in database."

    if book["available_copies"] <= 0:
        return (
            f"ALL_COPIES_BORROWED: Cannot borrow '{book['title']}' (ID: {book_id}). "
            f"All {book['total_copies']} copies are currently checked out by other patrons."
        )

    # Decrement available copies
    db.execute_write("UPDATE books SET available_copies = available_copies - 1 WHERE id = %s", (book_id,))

    # Create new loan ID
    loan_count_row = db.query_one("SELECT COUNT(*) as cnt FROM loans")
    loan_count = loan_count_row["cnt"] if loan_count_row else 0
    loan_id = f"LOAN-{1000 + loan_count + 1}"
    due_date = (datetime.now() + timedelta(days=duration_days)).strftime("%Y-%m-%d")

    db.execute_write("""
        INSERT INTO loans (loan_id, book_id, book_title, user_id, duration_days, due_date, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (loan_id, book_id, book["title"], "user_001", duration_days, due_date, "Active"))

    updated_book = db.query_one("SELECT available_copies FROM books WHERE id = %s", (book_id,))
    remaining = updated_book["available_copies"] if updated_book else 0

    return (
        f"[BORROW_SUCCESS]\n"
        f"Book:        '{book['title']}' (Book ID: {book_id})\n"
        f"Loan ID:     {loan_id}\n"
        f"Due Date:    {due_date} ({duration_days} days)\n"
        f"Status:      Active\n"
        f"Remaining:   {remaining} copy/copies available on shelf."
    )


@registry.register(schema=ReturnBookInput, risk_level=RiskLevel.YELLOW)
def return_book(loan_id: str) -> str:
    """Return a borrowed book by Loan ID and restore available copies in database."""
    clean_id = loan_id.strip().upper()
    loan = db.query_one("SELECT loan_id, book_id, book_title, status FROM loans WHERE loan_id = %s", (clean_id,))

    if not loan:
        return f"LOAN_NOT_FOUND: Loan '{loan_id}' was not found in database records."

    if loan["status"] == "Returned":
        return f"ALREADY_RETURNED: Book for Loan {clean_id} has already been returned."

    db.execute_write("UPDATE loans SET status = 'Returned' WHERE loan_id = %s", (clean_id,))
    db.execute_write("UPDATE books SET available_copies = available_copies + 1 WHERE id = %s", (loan["book_id"],))

    return (
        f"[RETURN_SUCCESS]\n"
        f"Loan ID:     {clean_id}\n"
        f"Book Title:  '{loan['book_title']}'\n"
        f"Status:      Returned\n"
        f"Message:     Book successfully returned to the library inventory."
    )


@registry.register(schema=GetLoanStatusInput, risk_level=RiskLevel.GREEN)
def get_loan_status(loan_id: str) -> str:
    """Read loan details and return due date from database by Loan ID."""
    clean_id = loan_id.strip().upper()
    loan = db.query_one(
        "SELECT loan_id, book_id, book_title, status, due_date, user_id FROM loans WHERE loan_id = %s",
        (clean_id,)
    )

    if not loan:
        return f"LOAN_NOT_FOUND: Loan '{loan_id}' was not found in database records."

    return (
        f"[LOAN_FOUND]\n"
        f"Loan ID:     {loan['loan_id']}\n"
        f"Book Title:  '{loan['book_title']}' (Book ID: {loan['book_id']})\n"
        f"Borrower:    {loan['user_id']}\n"
        f"Due Date:    {loan['due_date']}\n"
        f"Status:      {loan['status']}"
    )


@registry.register(schema=DeleteBookInput, risk_level=RiskLevel.RED)
def delete_book(book_id: int) -> str:
    """Permanently delete a book from the database catalog (Librarian only)."""
    book = db.query_one("SELECT id, title FROM books WHERE id = %s", (book_id,))
    if not book:
        return f"BOOK_NOT_FOUND: Book ID {book_id} does not exist in database."

    deleted_title = book["title"]
    db.execute_write("DELETE FROM books WHERE id = %s", (book_id,))

    return f"[DELETE_SUCCESS] Book '{deleted_title}' (ID: {book_id}) was permanently removed from the library database."
