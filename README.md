 Here's a comprehensive `README.md` for your project, showcasing FastAPI, Sentence-Transformers (SBERT), PostgreSQL with `pgvector` for similarity search.

---

## 🚀 FastAPI Similarity Search with SBERT and pgvector

This project is a **FastAPI-based** application that leverages **Sentence-Transformers (SBERT)** and **PostgreSQL with pgvector** to perform semantic similarity searches. It's ideal for applications like FAQ systems, document retrieval, and knowledge base searching. I made this repo for learning & testing how to develop & integrate it for POC.

The system is designed to:
- Efficiently **insert and store questions with embeddings**.
- Perform **semantic similarity searches** directly within PostgreSQL using the `pgvector` extension.
- Be lightweight, fast, and scalable for both development and production environments.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI
- **Embedding Model**: Sentence-Transformers (`paraphrase-MiniLM-L6-v2`)
- **Database**: PostgreSQL 15+ with `pgvector` extension
- **Containerization**: Docker, Docker Compose
- **Deployment**: PM2 (for process management)
- **Logging**: Python Logging with Timed Rotating File Handlers

---

## 📂 Project Structure

```
fastapi-similarity-search/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models and database schemas
│   ├── db.py                   # PostgreSQL connection and pooling
│   ├── logging_config.py       # Logging setup
│   ├── routes.py               # API routes
│   └── services.py             # Embedding generation and similarity logic
├── config/
│   ├── dev.env                 # Environment variables for development
│   └── prod.env                # Environment variables for production
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── README.md
```

---

## 🎯 Features
- **Semantic Search**: Perform similarity searches directly in PostgreSQL using SBERT embeddings.
- **Batch Insert**: Efficiently insert large sets of questions into the database.
- **Concurrent Requests**: Supports multiple users with PostgreSQL connection pooling.
- **Dynamic Filtering**: Filter searches by subject, academic year, and threshold.
- **Production Ready**: Deploy with Docker and PM2, ensuring high availability and scalability.

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.8+
- Docker & Docker Compose
- PostgreSQL 15+ (with `pgvector` extension)
- Node.js (for PM2)

---

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/fastapi-similarity-search.git
cd fastapi-similarity-search
```

---

### 3. Environment Configuration
Create `.env` files for development and production in the `config/` directory.

**config/dev.env**:
```env
DB_NAME=similarity_db
DB_USER=postgres
DB_PASSWORD=root
DB_HOST=postgres
DB_PORT=5432
APP_ENV=development
DEBUG=True
```

**config/prod.env**:
```env
DB_NAME=similarity_db
DB_USER=postgres
DB_PASSWORD=root
DB_HOST=postgres
DB_PORT=5432
APP_ENV=production
DEBUG=False
```

---

### 4. Docker Setup
#### Build and Run (Development):
```bash
ENV=dev docker-compose up --build
```

#### Build and Run (Production):
```bash
ENV=prod docker-compose up --build
```

This will:
- Spin up PostgreSQL with `pgvector` extension.
- Launch the FastAPI app on port `8000`.

---

## 🛠️ Database Setup (PostgreSQL + pgvector)

1. Access the running PostgreSQL container:
```bash
docker exec -it postgres psql -U postgres
```

2. Enable `pgvector` extension:
```sql
CREATE EXTENSION vector;
```

3. Create Tables for Questions and Embeddings:
```sql
CREATE TABLE questions (
    question_id SERIAL PRIMARY KEY,
    subject_name VARCHAR(255),
    question_text TEXT,
    acad_year VARCHAR(10)
);

CREATE TABLE question_embeddings (
    question_id INT REFERENCES questions(question_id),
    embedding VECTOR(384) -- 384 for MiniLM
);
```

---

## 🚀 API Endpoints

### 1. Insert Questions (Batch)
- **Endpoint**: `POST /insert_questions`
- **Description**: Insert questions and store embeddings.
- **Request Body**:
```json
{
  "questions": [
    {
      "subject_name": "Math",
      "question_text": "What is 2 + 2?",
      "acad_year": "2024"
    }
  ]
}
```
- **Response**:
```json
{
  "message": "Questions inserted successfully"
}
```

---

### 2. Similarity Search
- **Endpoint**: `POST /find_similar`
- **Description**: Retrieve top N similar questions by cosine similarity.
- **Request Body**:
```json
{
  "question_text": "What is the formula for water?",
  "subject_name": "Science",
  "threshold": 0.2,
  "top_n": 5
}
```
- **Response**:
```json
{
  "similar_questions": [
    {
      "question_id": 1,
      "question_text": "What is H2O?",
      "similarity": 0.95
    }
  ]
}
```

---

## 🛠️ Key Components

### 1. Embedding Generation (SBERT)
**services.py**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embedding(text: str):
    return model.encode(text)
```

---

### 2. PostgreSQL Insertion (Batch)
```python
def insert_questions(questions):
    embeddings = [generate_embedding(q.question_text) for q in questions]
    # Batch insert using psycopg2
```

---

### 3. Similarity Search in PostgreSQL
```sql
SELECT q.question_id, q.question_text, 1 - (qe.embedding <=> %s::vector) AS similarity
FROM question_embeddings qe
JOIN questions q ON qe.question_id = q.question_id
WHERE q.subject_name = %s
ORDER BY similarity DESC
LIMIT %s;
```

---

## 🐳 Docker Compose (docker-compose.yml)
```yaml
version: '3.9'

services:
  fastapi:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENV=dev
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: root
      POSTGRES_DB: similarity_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    command: ["postgres", "-c", "shared_preload_libraries=vector"]

volumes:
  postgres_data:
```

---

## 📈 Performance and Scaling
- **Batch Inserts**: Reduced latency for large-scale data insertion.
- **pgvector**: Performs similarity search directly within PostgreSQL, eliminating the need for external vector search tools.
- **Lightweight Models**: Uses `paraphrase-MiniLM-L6-v2`, which is faster and smaller than full BERT models. ModernSbert is also need to be tried.

---
Feel free to modify and use.
---
