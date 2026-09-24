-- Library Database Schema for PostgreSQL

-- Drop existing tables
DROP TABLE IF EXISTS loans CASCADE;
DROP TABLE IF EXISTS books CASCADE;

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

-- Indexes for performance
CREATE INDEX idx_books_title ON books(title);
CREATE INDEX idx_books_author ON books(author);
CREATE INDEX idx_books_category ON books(category);
CREATE INDEX idx_loans_user_id ON loans(user_id);
CREATE INDEX idx_loans_status ON loans(status);

-- Seed Initial Books
INSERT INTO books (id, title, author, category, isbn, total_copies, available_copies, location, description) VALUES
(1, 'Clean Code: A Handbook of Agile Software Craftsmanship', 'Robert C. Martin', 'Software Engineering', '978-0132350884', 3, 3, 'Shelf A-12', 'Essential principles and best practices for writing clean, maintainable code.'),
(2, 'Design Patterns: Elements of Reusable Object-Oriented Software', 'Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides', 'Computer Science', '978-0201633610', 2, 2, 'Shelf B-04', 'The classic guide to 23 foundational software design patterns.'),
(3, 'Introduction to Algorithms (CLRS)', 'Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein', 'Algorithms', '978-0262033848', 4, 0, 'Shelf C-01', 'Comprehensive textbook covering modern algorithm analysis and data structures.'),
(4, 'Artificial Intelligence: A Modern Approach', 'Stuart Russell, Peter Norvig', 'Artificial Intelligence', '978-0136042594', 5, 5, 'Shelf A-03', 'The authoritative textbook on AI algorithms, intelligent agents, and machine learning.'),
(5, 'The Pragmatic Programmer: Your Journey To Mastery', 'David Thomas, Andrew Hunt', 'Software Engineering', '978-0135957059', 2, 1, 'Shelf B-09', 'Practical advice covering software craftsmanship and testing.'),
(6, 'Designing Data-Intensive Applications', 'Martin Kleppmann', 'Distributed Systems', '978-1449373320', 4, 4, 'Shelf D-15', 'In-depth architecture guide exploring databases, replication, and distributed systems.');

-- Reset sequence for id auto-increment
SELECT setval('books_id_seq', (SELECT MAX(id) FROM books));

-- Seed Initial Loans
INSERT INTO loans (loan_id, book_id, book_title, user_id, duration_days, due_date, status) VALUES
('LOAN-1001', 3, 'Introduction to Algorithms (CLRS)', 'user_001', 14, '2026-10-08', 'Active');
