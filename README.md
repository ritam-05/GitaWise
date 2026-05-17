# GitaWise

GitaWise is an opinionated retrieval + assistant project built around the
Bhagavad Gita. It provides tooling to prepare the canonical dataset, create
semantic retrieval chunks, generate dense embeddings, and serve a FastAPI
backend with a Next.js frontend for interactive querying.

Key ideas:
- Convert source dataset into one retrieval chunk per verse
- Generate normalized dense embeddings for retrieval and reranking
- Store vectors in Qdrant (optional) and serve a session-aware FastAPI API
- Frontend built with Next.js for conversational UI

Repository layout (important files/folders):

- `backend/` — FastAPI backend, cache layer, query engine, vector-store helpers
- `frontend/` — Next.js app and UI components
- `datasets/` — source and processed CSV/NPY/Pickle files
- `backend/gita_vector_store/` — chunking, embedding generation, Qdrant upload
- `config.py` — central configuration and `.env` loader

Quickstart (development)
------------------------

Prerequisites:

- Python 3.10+ and pip
- Node.js 18+ and npm
- Optional: Qdrant for vector storage and any LLM API keys if you enable
	model-backed features

1) Create and activate a Python virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Optional: install frontend deps

```bash
cd frontend
npm install
cd ..
```

3) Configure environment variables

Copy a `.env` file at the repository root and set at least the following
variables if you plan to use Qdrant or external LLMs:

- `QDRANT_API_KEY`
- `QDRANT_ENDPOINT`
- `GEMINI_API_KEY`, `GROQ_API_KEY`, or `SARVAM_API_KEY` (if used)

4) Run the app (dev)

The frontend package includes a convenient `dev` script that runs Next.js
and the backend concurrently. From the `frontend/` folder run:

```bash
cd frontend
npm run dev
```

This runs the frontend and the FastAPI backend (via `uvicorn`) together.

Or run backend only:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Data pipeline (embeddings / vector store)
--------------------------------------

Prepare and upload vectors when you need to rebuild the retrieval index:

- Create semantic chunks (one chunk per verse):

```bash
python backend/gita_vector_store/create_chunks.py
```

- Generate embeddings for chunks:

```bash
python backend/gita_vector_store/generate_embeddings.py
```

- Upload generated embeddings & metadata to Qdrant:

```bash
python backend/gita_vector_store/upload_to_qdrant.py
```

Notes & developer tips
----------------------

- Configuration lives in `config.py` and is loaded from a root `.env` file.
- Embedding model and batch sizes are configurable via `config.py`.
- The backend uses a session middleware that returns an `X-Session-ID` header
	for session-aware routing.
- Scripts expect the `datasets/` folder to contain the canonical CSV files
	(see `datasets/` for example files).

Contributing
------------

Contributions are welcome — open an issue or a PR with a short description
of the change. If you add new dataset processing steps, please include
deterministic scripts under `backend/gita_vector_store/` and update `config.py`.

License
-------

This repository does not include a license file by default. Add a `LICENSE`
if you plan to make this project public.

Contact
-------

For questions or help, open an issue in this repository.
