# ─────────────────────────────────────────────────────────────────────────────
# rag.py — The RAG (Retrieval-Augmented Generation) pipeline
#
# This is the core AI file. It handles everything between "user uploads a PDF"
# and "user gets an answer".
#
# RAG in plain English:
#   Instead of asking an AI "what does this PDF say?" (it hasn't read it),
#   we:
#     1. Break the PDF into small pieces ("chunks")
#     2. Convert each chunk into a vector (a list of numbers that captures meaning)
#     3. Store those vectors in a database
#     4. When the user asks a question, convert it to a vector too
#     5. Find the chunks whose vectors are closest to the question vector
#     6. Give those chunks to Groq (Llama 3) as context and ask it to answer
#
# This way the AI is always working from actual document content, not guessing.
#
# WHY GROQ?
#   Groq is completely free (no credit card needed), extremely fast, and uses
#   Llama 3 — Meta's open-source model that rivals GPT-4 in quality.
#   Sign up at https://console.groq.com to get your free API key.
# ─────────────────────────────────────────────────────────────────────────────

import os       # for reading environment variables (like the API key)
import uuid     # for generating unique IDs for each document
import chromadb   # ChromaDB — our vector database
from groq import Groq  # Groq SDK — lets us call Llama 3 for free
from chromadb.utils import embedding_functions  # pre-built embedding model wrappers
from pypdf import PdfReader  # PDF parsing library
import io        # for converting bytes → file-like object (BytesIO)
from dotenv import load_dotenv
load_dotenv()  # automatically reads .env file and sets the variables

# ── Set up ChromaDB (vector database) ────────────────────────────────────────
# A vector database stores embeddings (vectors) and lets you search them by
# similarity. It's like a regular database but instead of "find rows where
# name='Alice'", you do "find rows whose meaning is close to this question".

# PersistentClient saves data to disk at "./chroma_db" so it survives restarts.
# If you just used chromadb.Client() it would reset every time the server restarts.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# An "embedding function" converts text → a vector (list of ~384 numbers).
# We use "all-MiniLM-L6-v2" — a small, fast model that runs locally for free.
# It downloads automatically the first time (~90MB). No API key needed.
# The numbers it produces capture semantic meaning:
#   "dog" and "puppy" will have similar vectors
#   "dog" and "skyscraper" will have very different vectors
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# A "collection" is like a table in a regular database.
# get_or_create_collection means: use it if it exists, create it if it doesn't.
# hnsw:space="cosine" sets the similarity metric to cosine similarity —
# a standard way to measure how "close" two vectors are (0 = identical, 1 = opposite)
collection = chroma_client.get_or_create_collection(
    name="pdf_chunks",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# Set up the Groq client — it automatically reads GROQ_API_KEY from the environment.
# Get your free key at https://console.groq.com (no credit card needed).
# This is the object we use to call Llama 3 later.
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# The Groq model we'll use — llama3-8b-8192 is:
#   - Free on Groq's free tier
#   - Very fast (Groq's hardware is purpose-built for inference speed)
#   - 8192 token context window (plenty for our chunks + question)
GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Configuration constants ───────────────────────────────────────────────────
# These control how we split PDFs into chunks.
# Tuning these changes the quality of retrieval:
#   Smaller chunks = more precise matches but less context per chunk
#   Larger chunks = more context but might include irrelevant text

CHUNK_SIZE    = 500   # each chunk is at most 500 characters long
CHUNK_OVERLAP = 100   # consecutive chunks share 100 characters of overlap
                      # overlap prevents answers from being split across chunk boundaries
TOP_K         = 4     # retrieve the 4 most relevant chunks per query


# ── Helper functions (private — not called from main.py) ─────────────────────
# Functions starting with _ are conventions for "internal use only"

def _extract_text(pdf_bytes: bytes) -> str:
    """
    Extracts all text from a PDF file.

    pdf_bytes: the raw binary content of the PDF file
    returns: a single string with all the text from all pages

    pypdf reads PDFs page by page. We extract each page's text and join them.
    BytesIO converts the raw bytes into a file-like object that pypdf can read
    (pypdf expects something it can .read() from, not raw bytes).
    """
    # Wrap bytes in BytesIO so pypdf can treat it like an open file
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []
    for page in reader.pages:
        # extract_text() returns the text on one page, or None if no text
        text = page.extract_text()
        if text:  # skip empty pages
            pages.append(text)

    # Join all pages with newlines into one big string
    return "\n".join(pages)


def _chunk_text(text: str) -> list[str]:
    """
    Splits a long string into smaller overlapping chunks.

    Why overlap? Imagine a key sentence falls right at the boundary of two chunks.
    Without overlap, neither chunk has the full sentence. With overlap, at least
    one chunk will contain it completely.

    Example with CHUNK_SIZE=10, CHUNK_OVERLAP=3:
      text   = "Hello World Foo Bar"
      chunk1 = "Hello Worl"   (characters 0-9)
      chunk2 = "rld Foo Ba"   (characters 7-16, starting 10-3=7)
      chunk3 = "Bar"          (characters 14-end)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        # Slice the text from start to end (or end of string if shorter)
        chunks.append(text[start:end])
        # Move start forward by (CHUNK_SIZE - CHUNK_OVERLAP)
        # This creates the overlap: next chunk starts before the current one ends
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ── Public functions (called from main.py) ────────────────────────────────────

def ingest_pdf(pdf_bytes: bytes, filename: str) -> str:
    """
    Full ingestion pipeline: PDF bytes → stored embeddings in ChromaDB.

    Steps:
      1. Generate a unique ID for this document
      2. Extract all text from the PDF
      3. Split the text into overlapping chunks
      4. Store all chunks in ChromaDB (it auto-embeds them using embedding_fn)

    Returns the doc_id so the frontend can reference this document later.
    """
    # Generate a random unique ID for this document (e.g. "a3f8c2d1-...")
    # uuid4() is random — essentially impossible to collide
    doc_id = str(uuid.uuid4())

    # Extract all text from the PDF bytes
    text = _extract_text(pdf_bytes)

    # Guard: if no text was extracted, the PDF is probably scanned/image-based
    # (those need OCR which we haven't implemented)
    if not text.strip():
        raise ValueError("Could not extract any text from this PDF. It may be scanned/image-only.")

    # Split the full text into chunks
    chunks = _chunk_text(text)

    # ChromaDB's .add() method takes parallel lists:
    #   ids       — a unique string ID for each chunk
    #   documents — the actual text of each chunk
    #   metadatas — extra info stored alongside each chunk (not embedded)

    # Build a unique ID for each chunk: "doc_id_0", "doc_id_1", etc.
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    # Metadata lets us filter later ("only search chunks from this document")
    # and display source info to the user
    metadatas = [
        {
            "doc_id": doc_id,       # which document this chunk belongs to
            "filename": filename,    # the original filename (for display)
            "chunk_index": i         # position in the document (for debugging)
        }
        for i in range(len(chunks))
    ]

    # Store everything in ChromaDB.
    # ChromaDB automatically calls embedding_fn on each chunk to create vectors.
    # After this, the chunks are searchable by semantic similarity.
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    return doc_id


def query_rag(question: str, doc_id: str | None = None) -> tuple[str, list[dict]]:
    """
    Full query pipeline: user question → AI answer + source chunks.

    Steps:
      1. (Optionally) filter to one document
      2. Embed the question and find the TOP_K most similar chunks
      3. Build a prompt that includes those chunks as context
      4. Send the prompt to Groq (Llama 3) and get an answer
      5. Return the answer + metadata about which chunks were used

    Returns a tuple: (answer_string, list_of_source_dicts)
    """
    # If a doc_id was provided, build a ChromaDB "where" filter.
    # This restricts the search to only chunks from that document.
    # Without this, we'd search across ALL uploaded documents.
    where = {"doc_id": doc_id} if doc_id else None

    # Query ChromaDB for the most similar chunks to the question.
    # ChromaDB will:
    #   1. Embed the question using embedding_fn
    #   2. Compute cosine similarity between the question vector and all stored vectors
    #   3. Return the TOP_K closest matches
    results = collection.query(
        query_texts=[question],   # list because ChromaDB supports batch queries
        n_results=TOP_K,          # return the 4 most similar chunks
        where=where,              # optional filter by doc_id
        include=["documents", "metadatas", "distances"],  # what data to return
    )

    # results["documents"] is a list of lists (one list per query — we only have one)
    # So results["documents"][0] is the list of chunk strings for our query
    docs      = results["documents"][0]   # e.g. ["chunk text 1", "chunk text 2", ...]
    metas     = results["metadatas"][0]   # e.g. [{"doc_id": ..., "filename": ...}, ...]
    distances = results["distances"][0]   # e.g. [0.12, 0.25, ...] — lower = more similar

    # If no chunks were found (empty database or no match), return early
    if not docs:
        return ("I couldn't find any relevant information in the uploaded documents.", [])

    # Build a formatted context string to inject into the prompt.
    # We label each chunk with its source so the model can cite them.
    # zip(docs, metas) pairs each chunk with its metadata
    context_blocks = "\n\n".join(
        f"[Source {i+1} — {m['filename']}]\n{doc}"
        for i, (doc, m) in enumerate(zip(docs, metas))
    )

    # Build the full prompt for Llama 3.
    # We use an f-string to inject the context and question.
    # The instructions tell the model to:
    #   - Only use the provided excerpts (not its training data)
    #   - Say so if the answer isn't in the excerpts
    #   - Cite which source it used
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided document excerpts.

DOCUMENT EXCERPTS:
{context_blocks}

QUESTION: {question}

Instructions:
- Answer using ONLY the information in the excerpts above.
- If the excerpts don't contain enough information, say so clearly.
- Cite which source(s) you used (e.g. "According to Source 1...").
- Be concise and direct.

ANSWER:"""

    # Call Groq's API — same interface as OpenAI's (chat completions format).
    # This is the standard "messages" format used by most LLM APIs:
    #   role: "system" = instructions for the model's behavior
    #   role: "user"   = the actual input from the user
    #   role: "assistant" = the model's previous replies (for multi-turn chat)
    # We only have one user message here since we stuffed everything into the prompt.
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,         # which model to use (llama3-8b-8192)
        max_tokens=1024,           # maximum length of the response
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the answer text from the response object.
    # response.choices is a list (Groq can return multiple completions).
    # [0] gets the first (and only) completion.
    # .message.content is the actual text string.
    answer = response.choices[0].message.content

    # Build the list of source objects to send back to the frontend.
    # The frontend displays these as clickable "source chips".
    sources = [
        {
            "doc_id":     m["doc_id"],
            "filename":   m["filename"],
            # Truncate long chunks for the preview (full text would be too much)
            "chunk_text": doc[:300] + ("..." if len(doc) > 300 else ""),
            # Convert cosine distance → similarity score:
            # distance=0 means identical → score=1.0
            # distance=1 means opposite  → score=0.0
            "score": round(1 - dist, 4),
        }
        for doc, m, dist in zip(docs, metas, distances)
    ]

    # Return both the answer and the sources as a tuple
    return answer, sources


def list_documents() -> list[dict]:
    """
    Returns a deduplicated list of all uploaded documents.

    ChromaDB stores one entry per chunk, so a 50-chunk PDF has 50 rows.
    We deduplicate by doc_id to return one entry per document.

    Returns: [{"doc_id": "...", "filename": "..."}, ...]
    """
    # Get all metadata from the collection (no need to return the actual text)
    results = collection.get(include=["metadatas"])

    # Use a dict keyed by doc_id to deduplicate.
    # If we see the same doc_id twice, we just overwrite with the same data.
    seen = {}
    for meta in results["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "filename": meta["filename"]}

    # Return just the values (the unique document dicts)
    return list(seen.values())


def delete_document(doc_id: str) -> None:
    """
    Deletes all chunks belonging to a document from ChromaDB.

    We can't just delete by doc_id directly — we need to find all chunk IDs
    that belong to this document, then delete those specific IDs.
    """
    # Find all chunk IDs where the metadata doc_id matches.
    # include=[] means we only need the IDs, not the text or metadata.
    results = collection.get(where={"doc_id": doc_id}, include=[])

    # Only delete if we found something (avoid errors on empty result)
    if results["ids"]:
        collection.delete(ids=results["ids"])