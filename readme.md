# Chunkless RAG Chat App — Multi-Paper Edition

Chat with one or more research papers using hierarchical document
navigation (no chunking, no vector DB), a persistent paper library,
cross-paper discussion mode, and live streaming answers.

## Setup (VS Code / local machine)

1. Open this folder in VS Code.
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Usage

1. Paste your Groq API key in the sidebar.
2. Upload a PDF and confirm/edit its title (this title is the paper's
   permanent unique ID in the library — re-uploading the same title
   skips re-parsing entirely and loads the cached structure instantly).
3. Click **Process & Add to Library**.
4. Choose a mode:
   - **🔎 Single Paper** — chat with, or run the full 12-question
     report on, one paper.
   - **🔗 Cross-Paper Discussion** — select 2+ papers; every answer
     explicitly attributes each point to the paper it came from, and
     calls out agreements/differences/citations between them.
5. Ask questions in the Chat tab — answers stream in live, token by
   token, instead of waiting for the full response.
6. In Single Paper mode, click **🧠 Analyze & Generate Report** to
   stream through all 12 standard research-analysis questions, with a
   live progress bar and an auto-generated flow diagram for the
   "End-to-End Method & Data Flow" question.

## Persistent Storage

- **`memory.db`** (SQLite) — chat history, survives restarts.
- **`library.db`** + **`trees/*.json`** — every parsed paper's
  hierarchical structure, saved permanently. This is what makes
  cross-paper mode possible across sessions: once a paper is in the
  library, you never need to re-upload or re-parse it again, even
  after closing the app.

## Files

- `app.py` — Streamlit UI: library management, mode switching, chat,
  streaming, report generation.
- `rag_engine.py` — PDF parsing, hierarchical tree building, multi-query
  agentic navigation, TF-IDF/MMR backstop retrieval, cross-paper
  retrieval, streaming answer generation, flow diagrams.
- `library.py` — persistent paper storage (SQLite + JSON tree files),
  keyed by paper title so duplicates are detected and skipped.
- `memory.py` — persistent chat history (SQLite).
- `requirements.txt` — dependencies.

## Notes

- If a PDF is scanned/image-based (no real text layer), set
  `pipeline_options.do_ocr = True` in `rag_engine.py`'s
  `parse_pdf_to_markdown`.
- Groq occasionally deprecates model names. If `MODEL` in `app.py`
  throws a 404, check https://console.groq.com/docs/models and update
  the constant.
- Cross-paper mode runs retrieval once per selected paper per question,
  so cost/time scales with paper count — kept intentionally lighter
  (fewer hops/query variants) per paper than single-paper mode.