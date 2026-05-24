# ─────────────────────────────────────────────────────────────────────────────
# main.py — FastAPI server (the "backend")
#
# This file is the entry point for our web server. It defines the API routes
# (URLs) that the React frontend will call. Think of it as a menu of actions
# the server can perform.
#
# How a web API works:
#   Frontend (React) sends an HTTP request  →  Server (FastAPI) handles it
#   e.g. "POST /upload" with a PDF file     →  server stores it and replies
# ─────────────────────────────────────────────────────────────────────────────

# FastAPI is the web framework — it lets us define routes with simple decorators
# UploadFile = a file sent from the browser
# File       = tells FastAPI to expect a file in the request body
# HTTPException = lets us send error responses (e.g. 400 Bad Request)
from fastapi import FastAPI, UploadFile, File, HTTPException

# CORSMiddleware solves the "CORS" problem:
# By default, browsers block requests from one origin (localhost:5173) to another
# (localhost:8000) for security. We add this middleware to tell the browser it's OK.
from fastapi.middleware.cors import CORSMiddleware

# BaseModel is from Pydantic — it lets us define the shape of JSON request/response bodies.
# FastAPI uses these to automatically validate incoming data and generate API docs.
from pydantic import BaseModel

# uvicorn is the actual web server that runs our FastAPI app (like Apache/Nginx but for Python)
import uvicorn

# Import our RAG functions from rag.py — the actual AI logic lives there
from rag import ingest_pdf, query_rag, list_documents, delete_document


# ── Create the FastAPI app ────────────────────────────────────────────────────
# This is the main app object. All routes are registered on it.
# The title shows up in the auto-generated docs at http://localhost:8000/docs
app = FastAPI(title="PDF Q&A API")


# ── Set up CORS ───────────────────────────────────────────────────────────────
# This allows our React frontend (running on port 5173) to talk to this server
# (running on port 8000) without the browser blocking it.
# allow_origins=["*"] means "accept requests from any URL" — fine for development,
# but in production you'd restrict this to your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # accept requests from any origin
    allow_methods=["*"],       # allow GET, POST, DELETE, etc.
    allow_headers=["*"],       # allow any HTTP headers
)


# ── Request / Response models ─────────────────────────────────────────────────
# These classes define the shape of JSON data we send and receive.
# FastAPI uses them to:
#   1. Automatically parse and validate incoming JSON
#   2. Serialize outgoing data to JSON
#   3. Generate interactive API docs at /docs

class QueryRequest(BaseModel):
    # The user's question as a plain string
    question: str
    # Optional: if provided, only search within this specific document.
    # "str | None = None" means it can be a string OR null, defaulting to null.
    doc_id: str | None = None


class QueryResponse(BaseModel):
    # Claude's answer text
    answer: str
    # List of source chunks used to generate the answer.
    # Each dict has: doc_id, filename, chunk_text, score
    sources: list[dict]


# ── Routes (API endpoints) ────────────────────────────────────────────────────
# Each function below is a "route handler" — it runs when the frontend
# makes an HTTP request to that URL path.
# The decorator (e.g. @app.get("/")) tells FastAPI which method + path to listen on.

@app.get("/")
def root():
    """
    Health check endpoint — just confirms the server is running.
    You can visit http://localhost:8000 in your browser to see this.
    """
    return {"status": "ok", "message": "PDF Q&A API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF file from the frontend, processes it, and stores it.

    'async' means this function can handle other requests while waiting
    for the file to finish uploading (non-blocking I/O).

    UploadFile gives us metadata (filename, content type) + the file data.
    File(...) means the file is required (the ... means "required" in Pydantic).
    """
    # Validate file type — reject anything that isn't a PDF
    if not file.filename.endswith(".pdf"):
        # HTTPException sends an error response back to the frontend
        # status_code=400 means "Bad Request" (the client sent something wrong)
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read the raw bytes of the uploaded file
    # 'await' pauses here until the file is fully read, then continues
    contents = await file.read()

    # Hand the bytes off to our RAG pipeline (defined in rag.py)
    # ingest_pdf will: parse text → chunk → embed → store in ChromaDB
    # It returns a unique ID we can use to reference this document later
    doc_id = ingest_pdf(contents, file.filename)

    # Return a JSON response confirming success
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "message": "PDF indexed successfully."
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Receives a question from the frontend and returns an AI-generated answer.

    response_model=QueryResponse tells FastAPI to validate our return value
    matches the QueryResponse shape before sending it to the client.

    req is automatically parsed from the JSON body of the POST request.
    """
    # Reject empty questions — .strip() removes whitespace so "   " counts as empty
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Run the full RAG pipeline:
    # embed question → find similar chunks → build prompt → call Claude → return answer
    # Returns a tuple: (answer_string, list_of_source_dicts)
    answer, sources = query_rag(req.question, doc_id=req.doc_id)

    # Wrap in QueryResponse so FastAPI can validate and serialize it
    return QueryResponse(answer=answer, sources=sources)


@app.get("/documents")
def documents():
    """
    Returns a list of all PDFs that have been uploaded and indexed.
    The frontend uses this to populate the document sidebar.
    """
    return {"documents": list_documents()}


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):
    """
    Deletes a document and all its stored chunks from ChromaDB.

    {doc_id} in the path is a "path parameter" — FastAPI automatically extracts
    it from the URL and passes it as the doc_id argument.
    e.g. DELETE /documents/abc-123  →  doc_id = "abc-123"
    """
    delete_document(doc_id)
    return {"message": f"Document {doc_id} deleted."}


# ── Start the server ──────────────────────────────────────────────────────────
# This block only runs when you execute "python main.py" directly.
# It won't run if another file imports this module (e.g. during testing).
if __name__ == "__main__":
    uvicorn.run(
        "main:app",      # "filename:app_variable" — tells uvicorn where the app is
        host="0.0.0.0",  # listen on all network interfaces (not just localhost)
        port=8000,       # the port number — visit http://localhost:8000
        reload=True      # auto-restart when you save changes (great for development)
    )