# Safe AI Library Agent (Topic 07: Autonomous Agents & Tool Integration)

An autonomous AI agent with application-level permission control (RBAC), schema validation, risk-tiered safety boundaries, a ReAct reasoning loop, and PostgreSQL database support (with `db.sql` schema and SQLite fallback).

---

## 1. Project Overview

This project implements a Safe Library Assistant Agent capable of receiving natural language user requests, selecting and calling tools with structured arguments, reading and modifying records in a relational database, observing tool outputs, and iteratively reasoning until a final answer is formulated.

### Core Principle
> **"The model proposes an action. The application decides whether that action is allowed to execute."**

The application enforces security, permissions, and input validation in Python application code rather than relying on LLM prompt compliance.

---

## 2. Available Tools & Schemas

The agent has access to 7 database-backed tools with explicit Pydantic input schemas and risk tier classifications:

| Tool Name | Risk Tier | Input Schema | Description |
| :--- | :--- | :--- | :--- |
| `search_books` | GREEN | `query: str`, `category: Optional[str]` | Searches library database catalog by title, author, keyword, or genre. |
| `get_book_details` | GREEN | `book_id: int` (`> 0`) | Reads complete book record, metadata, and description from the database by ID. |
| `check_availability` | GREEN | `book_id: int` (`> 0`) | Reads available copy count, total inventory, and shelf location from the database. |
| `borrow_book` | YELLOW | `book_id: int` (`> 0`), `duration_days: int` (`1-30`, default: `14`) | Borrows a book copy, decrements available shelf copies in database, and creates an active loan receipt with due date. |
| `return_book` | YELLOW | `loan_id: str` (`min_length=3`) | Returns a borrowed book, marks loan as returned in database, and restores available copy count. |
| `get_loan_status` | GREEN | `loan_id: str` (`min_length=3`) | Reads active loan information, book title, and return due date from database. |
| `delete_book` | RED | `book_id: int` (`> 0`) | Permanently deletes a book from the database catalog (Librarian only). |

---

## 3. Database Schema (`db.sql`)

The database structure is defined in [`db.sql`](file:///Users/sovannarith/Desktop/18_SOEUN_SOVANNARITH_AI_AGENT_HOMEWORK/db.sql) for PostgreSQL:

```sql
-- Books Table
CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    isbn VARCHAR(50) UNIQUE,
    total_copies INTEGER NOT NULL CHECK (total_copies >= 0),
    available_copies INTEGER NOT NULL CHECK (available_copies >= 0 AND available_copies <= total_copies),
    location VARCHAR(100) NOT NULL,
    description TEXT
);

-- Loans Table
CREATE TABLE loans (
    loan_id VARCHAR(50) PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    book_title VARCHAR(255) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    duration_days INTEGER NOT NULL CHECK (duration_days > 0),
    due_date DATE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'Active'
);
```

### Initializing PostgreSQL Database:
```bash
# Create database and import schema + seed data:
psql -U postgres -c "CREATE DATABASE library_db;"
psql -U postgres -d library_db -f db.sql
```

---

## 4. Agent Loop Architecture

The agent executes an iterative ReAct (Reasoning and Acting) loop:

```mermaid
flowchart TD
    A[User Request] --> B[Agent / LLM]
    B --> C{Tool Needed?}
    C -- No --> H[Final Synthesized Answer]
    C -- Yes --> D[Propose Tool Call & Arguments]
    D --> E[Safety Harness Interception]
    E -->|Check 1: Loop Limit| F1[Limit Guard]
    E -->|Check 2: RBAC Permission| F2[Permission Guard]
    E -->|Check 3: Pydantic Schema| F3[Validation Guard]
    F1 & F2 & F3 --> G[Execute Tool & Query Database]
    G --> I[Tool Observation / Error Result]
    I --> B
```

### Step-by-Step Flow:
1. **Receive User Request**: Captured via CLI REPL, single-query mode, or automated benchmark harness.
2. **Reasoning and Tool Selection**: Model generates internal thought and proposes tool invocation with JSON arguments.
3. **Safety Harness Interception**: Application intercepts the proposal before execution:
   - Validates active user role permissions (Member vs Librarian).
   - Validates input types and bounds via Pydantic (`book_id > 0`, `duration_days: 1-30`).
   - Checks session tool limit bounds to prevent endless loops.
4. **Tool Execution**: If authorized and valid, executes database queries in `tools.py`.
5. **Observation Feedback**: Returns structured observation or controlled error code (`PERMISSION_DENIED`, `UNAVAILABLE`, `ALL_COPIES_BORROWED`, `INPUT_VALIDATION_ERROR`, `BOOK_NOT_FOUND`) back to LLM context.
6. **Decide Again**: Model inspects observation and determines whether an additional action is needed.
7. **Final Answer**: Formulates a clear response to the user.

---

## 5. Permission Rule (Role-Based Access Control)

Permissions are enforced in `harness.py` at the application layer:

| Action / Tool | Member / Patron | Librarian / Admin | Enforcement Mechanism |
| :--- | :---: | :---: | :--- |
| `search_books` | Allowed | Allowed | Application RBAC Check |
| `get_book_details` | Allowed | Allowed | Application RBAC Check |
| `check_availability` | Allowed | Allowed | Application RBAC Check |
| `borrow_book` | Allowed | Allowed | Application RBAC Check |
| `return_book` | Allowed | Allowed | Application RBAC Check |
| `get_loan_status` | Allowed | Allowed | Application RBAC Check |
| `delete_book` | Blocked | Allowed | Safety Harness (Returns `PERMISSION_DENIED`) |

When a `Member` attempts an administrative action (such as `delete_book`), the execution is intercepted and rejected by `SafetyHarness.check_permission()`.

---

## 6. Safety Controls

1. **Input Validation**:
   - Pydantic models validate all inputs. Negative IDs (`book_id = -5`) or invalid loan durations trigger a controlled `INPUT_VALIDATION_ERROR` instead of unhandled exceptions.
2. **Controlled Failure Boundaries**:
   - Structured domain errors (`UNAVAILABLE`, `ALL_COPIES_BORROWED`, `PERMISSION_DENIED`, `BOOK_NOT_FOUND`, `LOAN_NOT_FOUND`) allow the agent to recover and explain the situation to the patron.
3. **Loop Limits**:
   - Configurable `max_steps` (default: 6) and `max_tool_calls` prevents runaway recursion or infinite loops.
4. **Risk Classification**:
   - Tools are categorized as Green (read-only), Yellow (state-modifying), and Red (destructive).

---

## 7. Example Runs

### Example 1: Multi-Step Member Search and Availability Chaining
```bash
$ python3 main.py --role member --query "Find algorithms books in the catalog and check its availability"
```
```text
[USER REQUEST]: 'Find algorithms books in the catalog and check its availability'
[USER ROLE]: MEMBER
-----------------------------------------------------------------

[Step 1/6]
[Thought]: Searching library catalog for 'algorithms'.
[Action Proposed]: search_books({'query': 'algorithms'})
[Observation]:
Found 2 book(s) matching your search:

1. Introduction to Algorithms (CLRS)
   - Book ID:  3
   - Author:   Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein
   - Category: Algorithms
   - Status:   0 available (ALL COPIES BORROWED)
   - Location: Shelf C-01

2. Artificial Intelligence: A Modern Approach
   - Book ID:  4
   - Author:   Stuart Russell, Peter Norvig
   - Category: Artificial Intelligence
   - Status:   5/5 available
   - Location: Shelf A-03

[Step 2/6]
[Thought]: Found matching books. Now checking copy availability and location for book ID 3.
[Action Proposed]: check_availability({'book_id': 3})
[Observation]:
UNAVAILABLE: Book 'Introduction to Algorithms (CLRS)' (ID: 3) has 0 copies available (all 4 copies currently borrowed). Location: Shelf C-01.

[Step 3/6]
[Final Answer]:
Notice: UNAVAILABLE: Book 'Introduction to Algorithms (CLRS)' (ID: 3) has 0 copies available (all 4 copies currently borrowed). Location: Shelf C-01.
```

---

### Example 2: Read Book Details from Database
```bash
$ python3 main.py --role member --query "Get details for book ID 1 from the database"
```
```text
[USER REQUEST]: 'Get details for book ID 1 from the database'
[USER ROLE]: MEMBER
-----------------------------------------------------------------

[Step 1/6]
[Action Proposed]: get_book_details({'book_id': '1'})
[Observation]:
[BOOK_DETAILS]
Title:       Clean Code: A Handbook of Agile Software Craftsmanship
Book ID:     1
Author:      Robert C. Martin
Category:    Software Engineering
ISBN:        978-0132350884
Status:      3/3 available
Location:    Shelf A-12
Description: Essential principles and best practices for writing clean, maintainable code.

[Step 2/6]
[Final Answer]:
[BOOK_DETAILS]
Title:       Clean Code: A Handbook of Agile Software Craftsmanship
Book ID:     1
Author:      Robert C. Martin
Category:    Software Engineering
ISBN:        978-0132350884
Status:      3/3 available
Location:    Shelf A-12
Description: Essential principles and best practices for writing clean, maintainable code.
```

---

### Example 3: Borrowing a Book (Creates Database Loan Record)
```bash
$ python3 main.py --role member --query "Borrow book ID 2 for 14 days"
```
```text
[USER REQUEST]: 'Borrow book ID 2 for 14 days'
[USER ROLE]: MEMBER
-----------------------------------------------------------------

[Step 1/6]
[Action Proposed]: borrow_book({'duration_days': '14', 'book_id': '2'})
[Observation]:
[BORROW_SUCCESS]
Book:        'Design Patterns: Elements of Reusable Object-Oriented Software' (Book ID: 2)
Loan ID:     LOAN-1002
Due Date:    2026-10-08 (14 days)
Status:      Active
Remaining:   1 copy/copies available on shelf.

[Step 2/6]
[Final Answer]:
[BORROW_SUCCESS]
Book:        'Design Patterns: Elements of Reusable Object-Oriented Software' (Book ID: 2)
Loan ID:     LOAN-1002
Due Date:    2026-10-08 (14 days)
Status:      Active
Remaining:   1 copy/copies available on shelf.
```

---

### Example 4: Member Attempts Forbidden Action (Blocked by Harness)
```bash
$ python3 main.py --role member --query "Delete book ID 1 from the catalog"
```
```text
[USER REQUEST]: 'Delete book ID 1 from the catalog'
[USER ROLE]: MEMBER
-----------------------------------------------------------------

[Step 1/6]
[Thought]: Attempting to delete book ID 1.
[Action Proposed]: delete_book({'book_id': 1})
[Observation]:
PERMISSION_DENIED: Role 'MEMBER' is not authorized to execute action 'delete_book'.

[Step 2/6]
[Final Answer]:
Action was blocked by the security harness:
PERMISSION_DENIED: Role 'MEMBER' is not authorized to execute action 'delete_book'.
Your current role is 'MEMBER'. This operation requires librarian / administrative privileges.
```

---

### Example 5: Librarian Deletes Book (Authorized)
```bash
$ python3 main.py --role librarian --query "Delete book ID 1 from the catalog"
```
```text
[USER REQUEST]: 'Delete book ID 1 from the catalog'
[USER ROLE]: LIBRARIAN
-----------------------------------------------------------------

[Step 1/6]
[Thought]: Attempting to delete book ID 1.
[Action Proposed]: delete_book({'book_id': 1})
[Observation]:
[DELETE_SUCCESS] Book 'Clean Code: A Handbook of Agile Software Craftsmanship' (ID: 1) was permanently removed from the library database.

[Step 2/6]
[Final Answer]:
Administrative action complete: [DELETE_SUCCESS] Book 'Clean Code: A Handbook of Agile Software Craftsmanship' (ID: 1) was permanently removed from the library database.
```

---

## 8. Project Structure

```
.
├── README.md                                    # Project documentation
├── main.py                                      # Application CLI, interactive REPL, and role switcher
├── agent.py                                     # Agent loop with LLM drivers & fallback
├── tools.py                                     # Database access manager (Postgres/SQLite) and schemas
├── schemas.py                                   # Data models for messages, tools, roles, and traces
├── harness.py                                   # Permission guard & automated test suite
├── db.sql                                       # PostgreSQL schema & initial seed data script
├── library.db                                   # SQLite database file (fallback)
├── harness_results.json                         # Benchmark test evaluation results
├── requirements.txt                             # Python dependencies
├── .env.example                                 # Environment configuration template
└── .gitignore                                   # Git ignore rules
```

---

## 9. How to Run This Project

### 9.1 Prerequisites
- Python 3.9+
- Optional: PostgreSQL database server
- Optional: Ollama with `llama3.2` installed, or an OpenAI API key.

### 9.2 Installation
```bash
# 1. Clone repository and navigate into the project directory:
cd 18_SOEUN_SOVANNARITH_AI_AGENT_HOMEWORK

# 2. (Optional) Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies:
pip install -r requirements.txt
```

---

### 9.3 Database Setup (PostgreSQL)

1. Create a PostgreSQL database and import the schema script:
   ```bash
   psql -U postgres -c "CREATE DATABASE library_db;"
   psql -U postgres -d library_db -f db.sql
   ```

2. Configure connection settings in `.env` (or set `DATABASE_URL`):
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/library_db
   ```

*(Note: If PostgreSQL is not configured, the application automatically uses the built-in SQLite database so everything runs out of the box without setup).*

---

### 9.4 Running the Interactive Agent (REPL Mode)

Start an interactive session to chat with the agent, test database operations, and switch roles:

```bash
# Start as a Member (default patron role):
python3 main.py --role member

# Start as a Librarian (admin role):
python3 main.py --role librarian
```

#### Interactive Commands within REPL:
- `/role member` — Switch role to Member (blocks deletion actions).
- `/role librarian` — Switch role to Librarian (allows administrative deletion).
- `/tools` — Display all registered tools, risk tiers, and current permissions.
- `/status` — View active user, role, model, and step limits.
- `/reset` — Reset conversation memory and re-initialize the database tables.
- `/test` — Run the automated test suite directly from the REPL.
- `/help` — Display command menu.
- `/exit` or `q` — Quit the application.

---

### 9.5 Running Single-Shot Queries

Execute one-off queries directly from the command line:

```bash
# Search for books in the database:
python3 main.py --role member --query "Search for clean code in the catalog"

# Browse all books:
python3 main.py --role member --query "what book that we have ?"

# Read book record from database:
python3 main.py --role member --query "Get details for book ID 1 from the database"

# Borrow a book (updates database inventory):
python3 main.py --role member --query "Borrow book ID 2 for 14 days"

# Check availability:
python3 main.py --role member --query "Check availability for book ID 4"

# Attempt unauthorized deletion as Member (Blocked):
python3 main.py --role member --query "Delete book ID 1 from the catalog"

# Perform authorized deletion as Librarian:
python3 main.py --role librarian --query "Delete book ID 1 from the catalog"
```

---

### 9.6 Running the Automated Safety & Benchmark Suite

To evaluate and verify the agent across all 8 benchmark test cases:

```bash
python3 harness.py
```
or via main CLI:
```bash
python3 main.py --test
```

The benchmark runs all test cases and outputs a structured results table, saving detailed output to `harness_results.json`.

---

### 9.7 Optional: Configuring Remote LLMs (Ollama / OpenAI)

By default, the agent runs in deterministic heuristic mode, which works offline without external API keys or services.

To connect with a local or cloud LLM:

1. **Local Ollama**:
   ```bash
   ollama pull llama3.2:latest
   python3 main.py --provider ollama --model llama3.2:latest
   ```

2. **OpenAI API**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   python3 main.py --provider openai --model gpt-4o-mini
   ```

---

## 10. Submission Checklist Verification

- [x] **At least 2 tools available**: 7 database-backed tools implemented (`search_books`, `get_book_details`, `check_availability`, `borrow_book`, `return_book`, `get_loan_status`, `delete_book`).
- [x] **Model chooses and calls a tool**: Structured function calling loop.
- [x] **Tool result returned to agent**: Observations fed back to context.
- [x] **Agent continues loop after observing result**: Multi-step chaining verified (search -> availability -> answer).
- [x] **Tool inputs have defined schema**: Strict Pydantic models for all tool inputs.
- [x] **Permission rule enforced in application code**: RBAC matrix checked in `harness.py`.
- [x] **Basic validation implemented**: Positive ID (`gt=0`), borrow duration (`1-30`), string lengths.
- [x] **Basic error handling implemented**: Controlled error messages (`PERMISSION_DENIED`, `ALL_COPIES_BORROWED`, `UNAVAILABLE`, etc.).
- [x] **Maximum iteration / tool-call limit**: `max_steps` and `max_tool_calls` enforced.
- [x] **PostgreSQL Database (`db.sql`)**: PostgreSQL DDL script with tables, constraints, foreign keys, indexes, and seed data provided.
- [x] **README & run instructions included**: Complete overview with architecture diagram and run instructions.
