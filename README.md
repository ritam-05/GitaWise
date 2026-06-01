# GitaWise

GitaWise is a Bhagavad Gita retrieval and assistant application. It combines a
FastAPI backend, an adaptive query engine, semantic vector retrieval, optional
session-aware caching, and a Next.js frontend for asking questions about duty,
fear, grief, action, peace, attachment, and other philosophical or emotional
themes.

The core idea is simple: prepare one searchable chunk per verse, embed those
chunks, retrieve the most relevant verses for a user's question, rerank the
matches, and generate a grounded answer with citations.

## What This Project Does

- Turns cleaned Bhagavad Gita data into retrieval-ready verse chunks.
- Generates dense sentence-transformer embeddings for each verse chunk.
- Uploads vectors and metadata to Qdrant for semantic search.
- Uses LLM-powered query analysis to identify problems, emotions, and intent.
- Routes questions through the best response path: direct guidance, emotional
  guidance, or full Gita RAG.
- Preserves conversational context through session IDs and optional cache
  storage.
- Provides a polished Next.js chat UI with citations and daily philosophy.

## Tech Stack

- Backend: Python, FastAPI, Pydantic, Uvicorn
- Retrieval: Qdrant, sentence-transformers, BGE embedding/reranker models
- LLM providers: Groq and Sarvam configuration hooks
- Cache: in-memory cache with optional Supabase persistence
- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Tooling: notebooks for data preparation and scripts for vector generation

## Quickstart

### 1. Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Configure environment variables

Create a root `.env` file. The exact values depend on which features you use,
but the backend expects these for the full retrieval and generation flow:

```env
QDRANT_API_KEY=
QDRANT_ENDPOINT=
GROQ_API_KEY=
SARVAM_API_KEY=
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_KEY=
SUPABASE_CACHE_TABLE=cache_entries
```

See `CONFIGURATION.md` for a deeper configuration walkthrough.

### 4. Run the app

From the repository root:

```bash
npm run dev
```

This delegates to the frontend dev script, which starts both:

- Next.js frontend at `http://localhost:3000`
- FastAPI backend at `http://127.0.0.1:8000`

Backend only:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend only:

```bash
cd frontend
npm run dev:frontend
```

## Data And Vector Pipeline

Run these scripts when rebuilding the retrieval index:

```bash
python backend/gita_vector_store/create_chunks.py
python backend/gita_vector_store/generate_embeddings.py
python backend/gita_vector_store/upload_to_qdrant.py
```

Pipeline stages:

1. `create_chunks.py` reads `datasets/clean_gita.csv` and writes
   `datasets/gita_chunks.csv`.
2. `generate_embeddings.py` reads `gita_chunks.csv` and writes
   `gita_embeddings.npy` plus `gita_metadata.pkl`.
3. `upload_to_qdrant.py` uploads embeddings and verse metadata into the Qdrant
   collection configured in `config.py`.

## API Overview

The backend mounts health routes and query-engine routes:

- `GET /` - health check with engine readiness.
- `GET /health` - explicit readiness endpoint.
- `GET /query-engine/today-philosophy` - deterministic daily verse summary.
- `POST /query-engine/analyze` - inspect decomposition, emotion detection, and
  retrieval query generation.
- `POST /query-engine/route` - see which route the adaptive engine chooses.
- `POST /query-engine/normalize-emotion` - map raw emotion text to canonical
  emotion labels.
- `POST /query-engine/run` - run the retrieval pipeline and return context.
- `POST /query-engine/answer` - return the final generated answer and citations.
- `DELETE /query-engine/sessions/{session_id}` - clear cached session memory.

## Project Structure

### Root

| Path | Purpose |
| --- | --- |
| `.env` | Local secrets and service credentials. This file should not be committed. |
| `.gitignore` | Ignore rules for local envs, caches, generated data, logs, and build output. |
| `README.md` | Main project guide and file map. |
| `CONFIGURATION.md` | Detailed setup notes for environment variables, model settings, and services. |
| `config.py` | Central project configuration: paths, dataset files, embedding settings, Qdrant settings, required schemas, API keys, and cache variables. |
| `requirements.txt` | Python dependency list for the backend, data pipeline, retrieval, and model tooling. |
| `package.json` | Root npm wrapper. Its `dev` script runs the frontend package dev command. |
| `package-lock.json` | Root npm lockfile for root-level JavaScript dependencies. |
| `.venv/` | Local Python virtual environment. Generated locally and ignored by git. |
| `node_modules/` | Root JavaScript dependency install folder. Generated locally and ignored by git. |
| `__pycache__/` | Python bytecode cache. Generated locally and ignored by git. |

### `backend/`

The FastAPI application and all retrieval/generation logic.

| Path | Purpose |
| --- | --- |
| `backend/main.py` | FastAPI entrypoint. Configures logging, CORS, session middleware, startup cache initialization, and route registration. |
| `backend/pyproject.toml` | Backend package/tool metadata. |
| `backend/.python-version` | Python version hint for tools such as pyenv. |
| `backend/README.md` | Backend-local README placeholder. |
| `backend/__init__.py` | Marks `backend` as an importable Python package. |
| `backend/__pycache__/` | Python runtime cache. Ignored by git. |

### `backend/routes/`

HTTP route modules exposed by FastAPI.

| Path | Purpose |
| --- | --- |
| `backend/routes/__init__.py` | Marks the routes directory as a package. |
| `backend/routes/health.py` | Implements `/` and `/health` readiness endpoints. |
| `backend/routes/query_engine.py` | Defines query-engine request/response models and endpoints for analysis, routing, answer generation, today's philosophy, and session clearing. |

### `backend/query_engine/`

The adaptive retrieval and answer-generation system.

| Path | Purpose |
| --- | --- |
| `backend/query_engine/__init__.py` | Public exports for `AdaptiveGitaEngine` and `GitaQueryEngine`. |
| `backend/query_engine/config.py` | Runtime query-engine config loaded from `.env`: model names, API keys, Qdrant settings, thresholds, reranking limits, and session settings. |
| `backend/query_engine/models.py` | Pydantic models and type labels for problems, emotions, routes, retrieved verses, engine responses, and generated answers. |
| `backend/query_engine/engine.py` | Main end-to-end retrieval engine: decomposition, emotion detection, query building, retrieval, reranking, and grounded response assembly. |
| `backend/query_engine/adaptive_engine.py` | Route-aware orchestration layer that decides whether to use direct guidance, emotion guidance, or full RAG, while handling conversation context. |
| `backend/query_engine/combined_analyzer.py` | Performs combined problem decomposition and emotion detection in a single structured analysis step. |
| `backend/query_engine/decomposer.py` | Groq JSON client and query decomposition helpers. |
| `backend/query_engine/emotion_normalizer.py` | Maps raw emotions into canonical GitaWise emotion labels. |
| `backend/query_engine/lightweight_router.py` | Classifies user queries into supported route labels before heavier retrieval work. |
| `backend/query_engine/query_builder.py` | Builds semantic retrieval query strings from detected problems and emotions. |
| `backend/query_engine/query_generator.py` | Generates expanded or optimized search queries for retrieval. |
| `backend/query_engine/retriever.py` | Retrieves candidate verses from Qdrant using embedded query vectors. |
| `backend/query_engine/embedding_model.py` | Loads and uses the embedding model for retrieval queries. |
| `backend/query_engine/reranker_model.py` | Loads the reranker model used to rescore retrieved candidates. |
| `backend/query_engine/reranker.py` | Applies global verse reranking and final context selection. |
| `backend/query_engine/generator.py` | Produces final direct or grounded responses from route decisions and retrieved context. |
| `backend/query_engine/prompts.py` | Prompt templates and system instructions for analysis, routing, and answer generation. |
| `backend/query_engine/context_builder.py` | Formats retrieved verses into context blocks for downstream generation. |
| `backend/query_engine/context_resolver.py` | Resolves follow-up references against prior conversation context. |
| `backend/query_engine/contextual_rewriter.py` | Rewrites follow-up questions into standalone queries when needed. |
| `backend/query_engine/continuation_detector.py` | Detects whether a user message continues the previous conversation. |
| `backend/query_engine/dialogue_state.py` | Represents and tracks conversational state for adaptive responses. |
| `backend/query_engine/state_transition_manager.py` | Handles transitions between dialogue states across turns. |

### `backend/cache/`

Session-aware and semantic caching utilities. These help avoid repeated work and
preserve context across a user's chat session.

| Path | Purpose |
| --- | --- |
| `backend/cache/__init__.py` | Public exports for cache manager, cache keys, session cache, conversation memory, and semantic cache. |
| `backend/cache/cache_keys.py` | Standardized key builders for cache entries and session state. |
| `backend/cache/cache_manager.py` | Multi-layer cache manager with in-memory storage and optional Supabase backing. |
| `backend/cache/semantic_cache.py` | Semantic response cache utilities for similar query reuse. |
| `backend/cache/session_cache.py` | Stores and retrieves session conversation turns and session memory. |
| `backend/cache/supabase_cache_schema.sql` | SQL schema for the optional Supabase cache table. |
| `backend/cache/__pycache__/` | Python runtime cache. Ignored by git. |

### `backend/gita_vector_store/`

Data-processing scripts for building and publishing the vector index.

| Path | Purpose |
| --- | --- |
| `backend/gita_vector_store/create_chunks.py` | Creates one structured retrieval text chunk per verse from the cleaned CSV. |
| `backend/gita_vector_store/generate_embeddings.py` | Generates normalized dense embeddings and metadata files for the verse chunks. |
| `backend/gita_vector_store/upload_to_qdrant.py` | Creates or updates the Qdrant collection and uploads vectors with payload metadata. |

### `backend/memory/`

Runtime memory directory used by local/session experiments. It is generated
locally and ignored by git.

### `datasets/`

Source, cleaned, enriched, and generated retrieval data. This directory is
ignored by git because the files can be large or regenerated.

| Path | Purpose |
| --- | --- |
| `datasets/Bhagwad_Gita.csv` | Original CSV dataset. |
| `datasets/Bhagwad_Gita.json` | JSON version of the source dataset. |
| `datasets/clean_gita.csv` | Cleaned canonical dataset consumed by chunk generation. |
| `datasets/enriched_gita.csv` | Dataset with additional topics, emotions, summaries, or enrichment fields. |
| `datasets/gita_chunks.csv` | Retrieval-ready verse chunks produced by `create_chunks.py`. |
| `datasets/gita_embeddings.npy` | NumPy array of dense verse embeddings produced by `generate_embeddings.py`. |
| `datasets/gita_metadata.pkl` | Pickled metadata records aligned with the embedding array. |

### `notebooks/`

Exploratory and preparation notebooks.

| Path | Purpose |
| --- | --- |
| `notebooks/data_cleaning.ipynb` | Notebook for cleaning the source dataset into a more reliable canonical form. |
| `notebooks/data_preparation.ipynb` | Notebook for preparing/enriching data before vector indexing. |

### `frontend/`

The Next.js application used by end users.

| Path | Purpose |
| --- | --- |
| `frontend/package.json` | Frontend scripts and dependencies. The main `dev` script runs Next.js and the FastAPI backend together with `concurrently`. |
| `frontend/package-lock.json` | Locked frontend dependency versions. |
| `frontend/next.config.ts` | Next.js configuration. |
| `frontend/next-env.d.ts` | Generated Next.js TypeScript declarations. |
| `frontend/postcss.config.mjs` | PostCSS configuration used by Tailwind. |
| `frontend/tailwind.config.ts` | Tailwind theme, content paths, and design token configuration. |
| `frontend/tsconfig.json` | TypeScript compiler configuration. |
| `frontend/tsconfig.tsbuildinfo` | TypeScript incremental build cache. Generated locally and ignored by git. |
| `frontend/.next/` | Next.js build/dev output. Generated locally and ignored by git. |
| `frontend/node_modules/` | Frontend dependency install folder. Generated locally and ignored by git. |
| `frontend/dev-*.log` | Local dev-server logs. Generated locally and ignored by git. |

### `frontend/app/`

Next.js App Router pages and global styles.

| Path | Purpose |
| --- | --- |
| `frontend/app/layout.tsx` | Root app layout, shared providers, fonts, metadata, and top-level UI shell. |
| `frontend/app/page.tsx` | Home page with the hero line art and initial chat prompt. |
| `frontend/app/chat/page.tsx` | Chat route wrapper. |
| `frontend/app/chat/chat-page-client.tsx` | Client-side chat experience: message state, request aborting, citations, and backend answer calls. |
| `frontend/app/not-found.tsx` | Custom 404 page. |
| `frontend/app/globals.css` | Global styles, Tailwind layers, theme variables, typography, and scrollbar styling. |

### `frontend/components/`

Reusable React UI pieces.

| Path | Purpose |
| --- | --- |
| `frontend/components/brand.tsx` | Brand mark/text component used in navigation or layout. |
| `frontend/components/chariot-line-art.tsx` | Decorative chariot artwork component. |
| `frontend/components/hero-line-art.tsx` | Hero illustration used on the home page. |
| `frontend/components/top-nav.tsx` | Top navigation bar. |
| `frontend/components/sidebar.tsx` | Chat page sidebar and session controls. |
| `frontend/components/chat-input.tsx` | Primary chat input used to submit questions. |
| `frontend/components/chat-message.tsx` | Renders user/assistant messages and attached citations. |
| `frontend/components/citation-card.tsx` | Displays cited chapter/verse details and verse metadata. |
| `frontend/components/floating-chatbox.tsx` | Floating chat container variant. |
| `frontend/components/floating-input.tsx` | Floating input variant for compact interactions. |
| `frontend/components/prompt-suggestions.tsx` | Suggested starting prompts for users. |
| `frontend/components/todays-philosophy-modal.tsx` | Modal for the daily philosophy endpoint. |
| `frontend/components/disclaimer-modal.tsx` | Disclaimer modal for user-facing guidance. |
| `frontend/components/theme-provider.tsx` | Theme context/provider integration. |
| `frontend/components/theme-toggle.tsx` | Light/dark theme toggle control. |
| `frontend/components/fade-in.tsx` | Small animation wrapper for entering content. |
| `frontend/components/ui/button.tsx` | Shared button primitive. |
| `frontend/components/ui/textarea.tsx` | Shared textarea primitive. |

### `frontend/lib/`

Frontend utilities and API access.

| Path | Purpose |
| --- | --- |
| `frontend/lib/api.ts` | Backend API client, session ID management, chat answer request, and session clearing helper. |
| `frontend/lib/utils.ts` | Shared frontend utility functions, including class-name merging helpers. |

### `frontend/public/`

Static assets served by Next.js.

| Path | Purpose |
| --- | --- |
| `frontend/public/transparent_gita.png` | Static image asset used by the frontend. |

## Request Flow

1. A user submits a question in the Next.js UI.
2. `frontend/lib/api.ts` creates or reuses a browser session ID.
3. The frontend posts the query to `POST /query-engine/answer`.
4. FastAPI session middleware attaches the session ID to the request.
5. `AdaptiveGitaEngine` routes the query and resolves conversation context.
6. If RAG is needed, `GitaQueryEngine` decomposes the query, detects emotions,
   builds retrieval queries, retrieves verses from Qdrant, reranks them, and
   passes selected context to the generator.
7. The backend returns an answer, route label, warnings, citations, and context.
8. The frontend renders the answer and citation cards.

## Development Notes

- Keep secrets in `.env`; never commit live API keys.
- Rebuild vector data only when the dataset or embedding model changes.
- If `query_engine.config.QueryEngineConfig` fails validation, check the root
  `.env` file for missing `GROQ_API_KEY`, `QDRANT_API_KEY`, or
  `QDRANT_ENDPOINT`.
- The frontend currently defaults to `http://127.0.0.1:8000` for backend calls.
  Set `BACKEND_URL` if you deploy the backend elsewhere.
- The backend returns an `X-Session-ID` response header, and the frontend also
  stores a browser-side session ID for continuity.

## License

This repository does not currently include a license file. Add a `LICENSE` file
before publishing or distributing the project.
