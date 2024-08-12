from typing import List, Optional

import logging_config  # Ensure this is correctly defined
import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException
from psycopg2 import extras, pool
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Get the logger
logger = logging_config.logger

embedder = SentenceTransformer('all-MiniLM-L6-V2')

CONNECTION_STRING = "dbname=testdb user=postgres password=root host=localhost port=5432"

db_pool = pool.SimpleConnectionPool(1, 30, CONNECTION_STRING)

app = FastAPI()

# Pydantic model for request
class QuestionItem(BaseModel):
    question_id: int
    question_text: str
    
class QuestionsRequest(BaseModel):
    questions: List[QuestionItem]

class SimilarityRequest(BaseModel):
    question_text: str
    threshold: Optional[float] = 0.05
    top_n: Optional[int] = 5

# Endpoint to add new questions in batch
@app.post('/insert_questions')
def insert_questions(request: QuestionsRequest):
    embeddings_list = []
    logger.info(f"Got {len(request.questions)} questions.")
    for question in request.questions:
        embedding = embedder.encode(question.question_text)
        embeddings_list.append(embedding)
        
    logger.info(f"Generated {len(embeddings_list)} embeddings successfully.")
    save_new_questions(request.questions, embeddings_list)
    return {"msg": "Questions inserted successfully"}

# Function to save new questions & its embeddings in batch size in the database
def save_new_questions(questions: List[QuestionItem], embeddings: List[np.ndarray]):
    conn = db_pool.getconn()
    try:
        curr = conn.cursor()
        # Insert questions into the question-bucket table
        embedding_values = [
            (q.question_id, q.question_text, emb.tolist())  # Convert numpy array to list for PostgreSQL
            for q, emb in zip(questions, embeddings)
        ]
        embedding_insert_query = """
            INSERT INTO question_bucket (question_id, question, embedding) VALUES %s
        """
        extras.execute_values(curr, embedding_insert_query, embedding_values, template=None)
        conn.commit()
        logger.info(f"Inserted {len(questions)} questions and {len(embeddings)} embeddings successfully.")
    except Exception as e:
        logger.error(f"Error inserting questions: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        curr.close()  # Ensure the cursor is closed properly
        db_pool.putconn(conn)

# Endpoint to find similar questions for the input
@app.post("/find_similar")
def find_similar(request: SimilarityRequest):
    if request.question_text == "":
        return {"msg": "Question text received empty"}
    
    query_embedding = embedder.encode(request.question_text).tolist()
    conn = db_pool.getconn()
    try:
        curr = conn.cursor()
        query = """
            SELECT
                question_id,
                question,
                1 - (embedding <=> %s::vector) AS similarity,
                (1 - (embedding <=> %s::vector)) * 100 AS similarity_percent
            FROM
                question_bucket
            WHERE
                1 - (embedding <=> %s::vector) >= %s
        """

        params = [query_embedding, query_embedding, query_embedding, request.threshold]
        query += " ORDER BY similarity DESC LIMIT %s"
        params.append(request.top_n)

        curr.execute(query, params)
        rows = curr.fetchall()
        similar_questions = [
            {
                "question_id": row[0],
                "question_text": row[1],
                "similarity": row[2],
                "similarity_percent": row[3]
            }
            for row in rows
        ]
        logger.info(f"Found {len(similar_questions)} similar questions.")
    except Exception as e:
        logger.error(f"Error finding similar questions: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        curr.close()  # Ensure the cursor is closed properly
        db_pool.putconn(conn)

    return {'similar_questions': similar_questions}

# Example endpoint to test the server
@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"msg": "Welcome to the question similarity API"}

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
