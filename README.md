# Safe Library AI Agent

An autonomous Library Assistant AI Agent that interacts with a PostgreSQL database to query catalog books, view book details, loan books, and return books with safety checks.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- **Python 3.9+**
- **PostgreSQL**
- **Ollama** (running `llama3.2:latest` or your preferred local LLM)

---

### 2. Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables (`.env`):**
   Copy `.env.example` to `.env` or set your PostgreSQL connection details:
   ```bash
   cp .env.example .env
   ```
   *Example `.env`:*
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/library_db
   OLLAMA_HOST=http://localhost:11434/api/chat
   OLLAMA_MODEL=llama3.2:latest
   ```

3. **Initialize Database:**
   Import the schema and initial seed data into PostgreSQL:
   ```bash
   psql -U postgres -c "CREATE DATABASE library_db;"
   psql -U postgres -d library_db -f db.sql
   ```

---

## 💻 How to Run

### 1. Interactive Terminal Mode (REPL)
Start the interactive session:
```bash
python3 main.py
```

### 2. Single-Shot Query Mode
Execute a single query directly from the command line:
```bash
python3 main.py --query "List all books in the catalog"
```
```bash
python3 main.py --query "Show details for Clean Code"
```
```bash
python3 main.py --query "I want to borrow Design Patterns for 14 days"
```
