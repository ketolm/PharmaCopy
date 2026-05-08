# PharmaCopy

A Python project for crawling FDA regulatory guideline pages, indexing relevant passages into ChromaDB, and evaluating pharmaceutical marketing copy for compliance.

## How it works

The pipeline is implemented using the files under `app/src`:

- `main.py`
  - Orchestrates the full workflow.
  - Reads URLs from `data/urls.txt` and checks `data/seen_urls.txt` for already processed links.
  - Crawls any new URLs, writes passages to `data/output.jsonl`, and inserts them into ChromaDB.
  - Prompts the user to paste pharmaceutical marketing copy and evaluates it against the indexed FDA guideline passages.

- `crawler.py`
  - Loads URL lists from a text file.
  - Fetches each URL's HTML using `requests`.
  - Extracts document titles and splits HTML into discrete passages using `google_labs_html_chunker`.
  - Writes each passage as one JSON object per line into `data/output.jsonl`.
  - Records processed URLs in `data/seen_urls.txt` to avoid duplicate crawling.

- `db.py`
  - Reads passages from `data/output.jsonl`.
  - Stores unique passages in a persistent ChromaDB collection under `app/src/chroma_db`.
  - Provides retrieval helpers to query the collection for relevant passages.
  - Includes a `verify_pharma_copy` helper that returns matching guideline passages and compliance feedback.

- `generate_response.py`
  - Loads a Hugging Face instruction-following model (`Qwen/Qwen2.5-7B-Instruct`).
  - Uses retrieved passages from the database to build a grounded prompt.
  - Generates compliant, citation-aware responses or compliance evaluations.
  - Contains a detailed evaluation prompt template tailored for FDA pharmaceutical marketing reviews.

## Project files

- `pyproject.toml` — Python package metadata and dependency declarations.
- `app/src/main.py` — Entry point for the full workflow.
- `app/src/crawler.py` — URL crawler and passage extractor.
- `app/src/db.py` — ChromaDB persistence and query logic.
- `app/src/generate_response.py` — LLM prompt, response generation, and evaluation logic.
- `app/src/chroma_db/` — Persistent ChromaDB storage folder.
- `data/urls.txt` — Input list of FDA guideline URLs to crawl.
- `data/output.jsonl` — Generated passage index output.
- `data/seen_urls.txt` — Tracks URLs already crawled.

## Setup

1. Install Python 3.14 or newer.
2. Install dependencies from `pyproject.toml` using `uv`.

```powershell
python -m pip install -U pip setuptools wheel uv
uv install
```

3. The service now supports an environment-based Hugging Face cache path. Set `HF_HOME` or allow it to default to `./hf_cache`.

4. Verify `app/src/data/urls.txt` contains one FDA regulatory URL per line.

## Usage

### Option 1: Interactive CLI (local Python)

Run the complete workflow interactively on your machine:

```powershell
python app/src/main.py app/src/data/urls.txt app/src/data/output.jsonl app/src/data/seen_urls.txt pharma_copy_collection
```

The script will:

1. Check `app/src/data/urls.txt` for new URLs.
2. Crawl any new FDA guideline pages.
3. Append extracted passages to `app/src/data/output.jsonl`.
4. Insert new passages into the ChromaDB collection.
5. Prompt you to paste pharmaceutical marketing copy and evaluate it interactively.

To evaluate marketing copy, paste the text and press Enter twice.

### Option 2: FastAPI Service (Containerized)

Run the app as a long-running HTTP service for programmatic access via API.

#### Quick start with Docker Compose

```powershell
docker compose up --build
```

This will:
- Build the `pharmacopy-api` image
- Start the service on `http://localhost:8000`
- Mount local volumes for data persistence:
  - `./app/src/data` → `/app/app/src/data`
  - `./app/src/chroma_db` → `/app/app/src/chroma_db`
  - `./hf_cache` → `/app/hf_cache`

#### Or build manually with Docker

```powershell
docker build -t pharmacopy-api .
docker run --rm -p 8000:8000 pharmacopy-api
```

#### API Endpoints

- `GET /` — Service overview and endpoint list
- `GET /health` — Health check
- `POST /crawl` — Crawl new FDA guideline URLs and append passages to JSONL
- `POST /index` — Import `data/output.jsonl` into ChromaDB
- `POST /evaluate` — Evaluate pharmaceutical marketing copy
- `POST /ask` — Ask a question using the indexed FDA guidelines

View interactive API docs at `http://localhost:8000/docs`.

#### Example API Payloads

```json
POST /crawl
{
  "urls_file": "/app/app/src/data/urls.txt",
  "output_path": "/app/app/src/data/output.jsonl",
  "seen_path": "/app/app/src/data/seen_urls.txt"
}
```

```json
POST /index
{
  "output_path": "/app/app/src/data/output.jsonl",
  "collection_name": "pharma_copy_collection"
}
```

```json
POST /evaluate
{
  "marketing_copy": "Our new drug relieves pain faster than the competition.",
  "collection_name": "pharma_copy_collection"
}
```

```json
POST /ask
{
  "question": "Is it compliant to claim improved symptoms?",
  "collection_name": "pharma_copy_collection"
}
```

Or use the interactive Swagger UI at `http://localhost:8000/docs`.

## Data flow

1. `data/urls.txt` holds candidate FDA guideline URLs.
2. `crawler.py` fetches and chunks each URL into passage records.
3. `data/output.jsonl` stores each passage as a JSON line.
4. `db.py` loads the JSONL file and upserts passages into ChromaDB.
5. `generate_response.py` queries the DB and uses an LLM to evaluate or answer questions.

## Notes

- **Local mode**: The workflow is incremental. Once a URL is marked in `app/src/data/seen_urls.txt`, it will not be reprocessed.
- **Service mode**: Data is persisted in mounted volumes, so state is preserved across container restarts.
- ChromaDB avoids duplicate passage inserts by tracking passage IDs.
- The model in `generate_response.py` is large and may require GPU or sufficient CPU memory.
- If no new URLs are found, the pipeline still evaluates copy against the existing indexed database.

## Customization

### Local mode

- Edit `app/src/data/urls.txt` directly to add new FDA guideline URLs.
- Change file paths or `collection_name` by updating `main.py` command-line arguments.

### Service mode

- Edit `./app/src/data/urls.txt` on your host machine (mounted in container).
- POST to `/crawl`, `/index`, `/evaluate`, or `/ask` endpoints with custom `collection_name` values.
- Replace the model in `generate_response.py` if you want a different Hugging Face model.
- Use `verify_pharma_copy` from `db.py` directly for programmatic checks.

## Troubleshooting

- **Crawling fails**: Check network access and ensure URLs are valid.
- **ChromaDB not found**: Confirm that `./app/src/chroma_db/` exists and is writable (for service mode, check volume mounts).
- **Model loading fails**: Verify the `HF_HOME` environment variable points to a writable cache directory.
- **API not responding**: Check that port `8000` is not already in use or blocked by a firewall.
