"""
Chunkless RAG engine.

- Parses a PDF into a hierarchical section tree (docling), preserving
  nesting (section > subsection > subsubsection), not flat chunks.
- An LLM agent navigates the tree via its outline, hopping across
  sections as needed (multi-hop), instead of vector similarity search.
- Multi-query retrieval rewrites the question into several phrasings so
  vocabulary mismatches between question and paper don't cause misses.
- TF-IDF + MMR is used as a relevance+diversity backstop alongside the
  agent's navigation (classical sparse vectors, not neural embeddings -
  stays vector-DB-less in spirit).
- Supports both SINGLE-PAPER retrieval and CROSS-PAPER retrieval (asking
  a question across two or more papers at once, with each point
  attributed back to the paper it came from).
- Final answers are streamed token-by-token (generator), and are
  instructed to favor bullet points / tables over long prose.
"""

import re
import difflib
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ---------- TEXT CLEANUP ----------

def clean_extracted_text(text: str) -> str:
    """
    Docling sometimes leaves raw HTML in table cells (<br>, <br/>, etc.)
    when it can't cleanly convert a table line-break to markdown. Left
    as-is, these get quoted verbatim into retrieved context and then into
    the model's answer. Convert them to real newlines/spaces instead.
    """
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)  # strip any other stray HTML tags
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# ---------- PDF PARSING ----------

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """Parses PDF preserving structure. OCR disabled since research papers
    normally have a real text layer - flip do_ocr=True if yours is scanned."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    return clean_extracted_text(markdown)


# ---------- HIERARCHICAL TREE (not flat chunks) ----------

def build_nested_section_tree(markdown_text: str) -> dict:
    lines = markdown_text.split("\n")
    root = {"title": "Document Root", "level": 0, "content": "", "children": [], "path": []}
    stack = [root]
    current_content_lines = []

    def flush_content():
        if stack[-1]["content"] == "" and current_content_lines:
            stack[-1]["content"] = "\n".join(current_content_lines).strip()
        current_content_lines.clear()

    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            flush_content()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            parent_path = stack[-1]["path"]
            node = {
                "title": title,
                "level": level,
                "content": "",
                "children": [],
                "path": parent_path + [title],
            }
            stack[-1]["children"].append(node)
            stack.append(node)
        else:
            current_content_lines.append(line)

    flush_content()
    return root


def get_outline_string(node: dict, indent: int = 0) -> str:
    lines = []
    if node["title"] != "Document Root":
        lines.append("  " * indent + f"- {node['title']}")
    for child in node["children"]:
        lines.append(get_outline_string(child, indent + 1))
    return "\n".join(l for l in lines if l)


def find_node_by_title(node: dict, title: str):
    if node["title"] == title:
        return node
    for child in node["children"]:
        found = find_node_by_title(child, title)
        if found:
            return found
    return None


def get_node_with_ancestry(node: dict) -> str:
    lineage = " > ".join(node["path"]) if node["path"] else node["title"]
    return f"[{lineage}]\n{node['content']}"


def get_all_titles(node: dict) -> list:
    titles = []
    if node["title"] != "Document Root":
        titles.append(node["title"])
    for child in node["children"]:
        titles.extend(get_all_titles(child))
    return titles


def collect_sections_with_content(node: dict) -> list:
    """Flat list of every node with real text - used only for the TF-IDF/MMR
    fallback index, never for the primary tree navigation."""
    sections = []
    if node["title"] != "Document Root" and node["content"].strip():
        sections.append(node)
    for child in node["children"]:
        sections.extend(collect_sections_with_content(child))
    return sections


def find_closest_title(candidate: str, all_titles: list, cutoff: float = 0.6):
    """Fuzzy-matches an LLM-produced heading guess against the real outline,
    so near-miss headings (case, punctuation, light rewording) still resolve
    instead of silently failing on an exact-match check."""
    if candidate in all_titles:
        return candidate
    matches = difflib.get_close_matches(candidate, all_titles, n=1, cutoff=cutoff)
    return matches[0] if matches else None


# ---------- TF-IDF + MMR HYBRID RETRIEVAL (backstop, no embeddings) ----------

def mmr_select_sections(question: str, tree: dict, top_k: int = 4, lambda_param: float = 0.7):
    """
    Maximal Marginal Relevance over section texts using TF-IDF. Picks
    sections relevant to the question AND diverse from each other, instead
    of several near-duplicate sections. Runs as a reliability backstop
    alongside LLM tree navigation.
    """
    sections = collect_sections_with_content(tree)
    if not sections:
        return []

    texts = [f"{' '.join(s['path'])} {s['content']}" for s in sections]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        doc_matrix = vectorizer.fit_transform(texts + [question])
    except ValueError:
        return []

    query_vec = doc_matrix[-1]
    doc_vecs = doc_matrix[:-1]
    relevance = cosine_similarity(doc_vecs, query_vec).flatten()

    if relevance.max() <= 0:
        return []

    selected_idx = []
    candidate_idx = list(range(len(sections)))

    while len(selected_idx) < min(top_k, len(sections)) and candidate_idx:
        if not selected_idx:
            best = max(candidate_idx, key=lambda i: relevance[i])
        else:
            selected_vecs = doc_vecs[selected_idx]
            best_score = -np.inf
            best = candidate_idx[0]
            for i in candidate_idx:
                sim_to_selected = cosine_similarity(doc_vecs[i], selected_vecs).max()
                mmr_score = lambda_param * relevance[i] - (1 - lambda_param) * sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
        selected_idx.append(best)
        candidate_idx.remove(best)

    return [sections[i] for i in selected_idx if relevance[i] > 0.05]


# ---------- MULTI-QUERY VARIANTS ----------

def generate_query_variants(question: str, client, model: str, n: int = 3) -> list:
    prompt = f"""Generate {n} different ways to ask the same question, using varied vocabulary and phrasing. This is for searching a research paper that may use different terminology than the original question.

ORIGINAL QUESTION: {question}

Respond with ONLY the {n} reworded questions, one per line, no numbering, no extra text."""

    response = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    variants = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
    return [question] + variants


GENERIC_STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "in", "on", "for", "what",
    "how", "why", "does", "do", "are", "this", "that", "these", "those",
    "explain", "paper", "part", "tell", "with", "from", "about", "step",
    "steps", "we", "you", "it", "please", "can", "could", "would",
}


def extract_candidate_terms(question: str) -> list:
    """Pulls out likely technical terms/acronyms from a question (longer
    words, mixed-case tokens like 'OPRatio', anything in quotes) so we can
    do a direct substring search for them across the paper - this catches
    specific terminology that TF-IDF's overall relevance score can bury,
    especially when the term appears in body text rather than a heading."""
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
    quoted_terms = [q for pair in quoted for q in pair if q]

    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question)
    candidate_words = [
        w for w in words
        if w.lower() not in GENERIC_STOPWORDS and (len(w) >= 5 or not w.islower())
    ]
    return list(dict.fromkeys(quoted_terms + candidate_words))  # dedupe, keep order


def keyword_exact_match_sections(question: str, tree: dict, max_matches: int = 3) -> list:
    """Direct case-insensitive substring search for distinctive terms from
    the question across every section's actual text. Cheap, and catches
    exact terminology/formula lookups (acronyms, named metrics, symbols)
    that relevance-ranked retrieval can under-rank when the term is rare
    but the section is otherwise dense with other, more common words."""
    terms = extract_candidate_terms(question)
    if not terms:
        return []

    sections = collect_sections_with_content(tree)
    scored = []
    for node in sections:
        content_lower = node["content"].lower()
        hits = sum(1 for t in terms if t.lower() in content_lower)
        if hits > 0:
            scored.append((hits, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [node for _, node in scored[:max_matches]]


def rewrite_standalone_question(question: str, chat_history: list, client, model: str) -> str:
    """
    Resolves vague follow-ups ('explain it', 'go deeper on that') into a
    self-contained question using the recent conversation, BEFORE
    retrieval runs. Retrieval only ever sees the raw question text, so a
    pronoun-only follow-up gives it nothing to search on - this fixes
    that by expanding the question first.
    """
    if not chat_history:
        return question

    # Only bother rewriting genuinely ambiguous/short follow-ups - skip
    # the extra API call for already-specific questions.
    vague_markers = ("it", "this", "that", "explain", "more", "deeper", "go deep", "elaborate")
    if len(question.split()) > 8 and not any(m in question.lower() for m in vague_markers):
        return question

    recent = chat_history[-4:]
    history_snippet = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in recent)

    prompt = f"""Conversation so far:
{history_snippet}

The user's latest message is: "{question}"

Rewrite this into a single, fully self-contained question that makes sense with no prior context (resolve any "it"/"this"/"that" references using the conversation above). If the latest message is already self-contained, just return it unchanged.

Respond with ONLY the rewritten question, nothing else."""

    response = client.chat.completions.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ---------- SINGLE-PAPER RETRIEVAL (context only, no final answer yet) ----------

def retrieve_context_multi_query(
    question: str,
    tree: dict,
    client,
    model: str,
    chat_history: list = None,
    max_hops: int = 3,
    n_variants: int = 3,
    use_mmr_backstop: bool = True,
    mmr_top_k: int = 4,
) -> dict:
    """
    Runs multi-query agentic navigation + MMR backstop, returns gathered
    context WITHOUT generating the final answer (so the caller can stream
    the answer separately). Returns: context_text, sections_visited, query_variants
    """
    chat_history = chat_history or []
    all_titles = get_all_titles(tree)
    outline = get_outline_string(tree)

    query_variants = generate_query_variants(question, client, model, n=n_variants)

    visited_titles = set()
    gathered_context = []

    history_snippet = ""
    if chat_history:
        recent = chat_history[-6:]
        history_snippet = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    for variant in query_variants:
        local_visited = []
        for _ in range(max_hops):
            nav_prompt = f"""You are navigating a research paper's table of contents to answer a question.

CONVERSATION SO FAR (for context, may be empty):
{history_snippet if history_snippet else "None"}

DOCUMENT OUTLINE:
{outline}

ALREADY VISITED (this pass): {local_visited if local_visited else "None yet"}

QUESTION: {variant}

Pick the SINGLE most relevant section heading to read next. Copy it EXACTLY as it appears in the outline above.
If none are relevant or you've gathered enough, respond with exactly: DONE

Respond with ONLY the heading text or DONE, nothing else."""

            nav_response = client.chat.completions.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": nav_prompt}],
            )
            raw_choice = nav_response.choices[0].message.content.strip()

            if raw_choice == "DONE":
                break

            matched_title = find_closest_title(raw_choice, all_titles)
            if matched_title is None:
                continue

            local_visited.append(matched_title)
            if matched_title not in visited_titles:
                visited_titles.add(matched_title)
                node = find_node_by_title(tree, matched_title)
                if node:
                    gathered_context.append(get_node_with_ancestry(node))

    if use_mmr_backstop:
        for node in mmr_select_sections(question, tree, top_k=mmr_top_k):
            if node["title"] not in visited_titles:
                visited_titles.add(node["title"])
                gathered_context.append(get_node_with_ancestry(node))

    # Exact-term backstop: catches specific technical terms/acronyms that
    # TF-IDF's overall relevance score can under-rank when they're rare
    # but sit inside an otherwise dense, common-word-heavy section.
    for node in keyword_exact_match_sections(question, tree, max_matches=3):
        if node["title"] not in visited_titles:
            visited_titles.add(node["title"])
            gathered_context.append(get_node_with_ancestry(node))

    return {
        "context_text": "\n\n".join(gathered_context),
        "sections_visited": list(visited_titles),
        "query_variants": query_variants,
    }


# ---------- CROSS-PAPER RETRIEVAL ----------

def retrieve_context_cross_paper(
    question: str,
    papers: list,
    client,
    model: str,
    chat_history: list = None,
    max_hops: int = 2,
    n_variants: int = 1,
    mmr_top_k: int = 3,
) -> dict:
    """
    papers: list of {"title": str, "tree": dict}
    Runs retrieval independently per paper (lighter settings than single-paper
    mode, since cost multiplies by paper count), tags each gathered section
    with its source paper, and returns a combined context plus a per-paper
    breakdown for the retrieval-trace UI.

    Returns: combined_context_text, context_by_paper {title: {sections_visited}}, query_variants
    """
    context_by_paper = {}
    combined_blocks = []
    all_variants = {}

    for paper in papers:
        result = retrieve_context_multi_query(
            question=question,
            tree=paper["tree"],
            client=client,
            model=model,
            chat_history=chat_history,
            max_hops=max_hops,
            n_variants=n_variants,
            use_mmr_backstop=True,
            mmr_top_k=mmr_top_k,
        )

        # Floor fallback: broad/vague comparison questions ("explain the
        # difference between these papers") often have no keyword overlap
        # with any specific section, so every retrieval pass can come back
        # empty for a given paper. Rather than silently dropping that
        # paper from the discussion, fall back to its opening section
        # (title/abstract/intro - whatever comes first in the tree) so
        # the model has at least something to compare against.
        if not result["context_text"]:
            leaf_sections = collect_sections_with_content(paper["tree"])
            if leaf_sections:
                fallback_node = leaf_sections[0]
                result["context_text"] = get_node_with_ancestry(fallback_node)
                result["sections_visited"] = [f"{fallback_node['title']} (fallback - no keyword match)"]

        context_by_paper[paper["title"]] = {"sections_visited": result["sections_visited"]}
        all_variants[paper["title"]] = result["query_variants"]

        if result["context_text"]:
            combined_blocks.append(f"===== PAPER: {paper['title']} =====\n{result['context_text']}")

    return {
        "combined_context_text": "\n\n".join(combined_blocks),
        "context_by_paper": context_by_paper,
        "query_variants_by_paper": all_variants,
    }


# ---------- STREAMED FINAL ANSWER ----------

FORMATTING_RULES = """
IMPORTANT FORMATTING RULES:
- Prefer concise bullet points over long paragraphs. Use short paragraphs only when a single connected explanation is genuinely clearer than bullets.
- When comparing multiple items (methods, papers, metrics, baselines), use a markdown table instead of prose.
- When writing mathematical equations, use Streamlit-compatible LaTeX: inline math wrapped in single dollar signs (e.g. $h$), block/display equations wrapped in double dollar signs on their own line. Do NOT use \\[ \\] or bare parentheses around math.
- Never output raw HTML tags (e.g. <br>, <div>) - use plain markdown line breaks and formatting only, even if the source text contains HTML artifacts.

ANSWER GROUNDING RULE:
- If the gathered sections directly answer the question, answer from them and treat that as the primary answer.
- If the question asks about a general concept (e.g. "what are eigenvalues", "what does covariance mean") that the gathered sections only reference but don't fully define, you may briefly explain the general concept using standard knowledge - but clearly label it, e.g. "General background (not from the paper's text):" - then tie it back to how the paper uses it if the gathered sections show that.
- Only say "not found in the paper" if the question is specifically asking whether/how the paper itself covers something and the gathered sections truly contain nothing relevant - don't say this for general conceptual questions you can reasonably explain.
"""

CROSS_PAPER_RULES = """
- This question spans MULTIPLE papers. For every point, explicitly name which paper it came from, e.g. "(Paper: <title>)".
- If the papers agree, disagree, or one paper builds on/cites ideas resembling another, call that out explicitly as a comparison point.
- Structure the answer as: a short per-paper bullet breakdown, followed by a "Comparison" section (table if comparing methods/metrics/results), followed by a one-line synthesis.
"""


def stream_final_answer(
    question: str,
    context_text: str,
    client,
    model: str,
    chat_history: list = None,
    cross_paper: bool = False,
):
    """
    Generator yielding text chunks as the final answer streams in.
    Use with st.write_stream() in Streamlit for live token-by-token display.
    """
    chat_history = chat_history or []
    history_snippet = ""
    if chat_history:
        recent = chat_history[-6:]
        history_snippet = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    rules = FORMATTING_RULES + (CROSS_PAPER_RULES if cross_paper else "")

    prompt = f"""You are answering a question about {"multiple research papers" if cross_paper else "a research paper"} using ONLY the sections gathered below.

CONVERSATION SO FAR (for context, may be empty):
{history_snippet if history_snippet else "None"}

GATHERED SECTIONS:
{context_text if context_text else "No relevant sections found."}

CURRENT QUESTION: {question}

Answer clearly and directly. If it's a follow-up question, use the conversation context.
If the answer truly isn't in the gathered sections, say so.
{rules}"""

    stream = client.chat.completions.create(
        model=model,
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------- FLOW DIAGRAM GENERATION ----------

def generate_flow_diagram(context_text: str, client, model: str) -> str:
    """
    Asks the LLM to describe the method's data flow as Graphviz DOT source.
    Returned string is passed straight to st.graphviz_chart() - Streamlit
    renders DOT client-side, no extra system install needed.
    """
    if not context_text.strip():
        return ""

    prompt = f"""Based on the text below, describe the end-to-end data/method flow as a Graphviz DOT diagram (a directed graph).

TEXT:
{context_text[:6000]}

Rules:
- Output ONLY valid Graphviz DOT code, starting with "digraph G {{" and ending with "}}".
- No markdown code fences, no explanation, no extra text.
- Use short, readable node labels (a few words each).
- Use -> for directed edges showing the order data/processing flows.
- Keep it to at most 12 nodes - summarize into the major stages only."""

    response = client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    dot_code = response.choices[0].message.content.strip()
    dot_code = re.sub(r"^```[a-zA-Z]*\n?", "", dot_code)
    dot_code = re.sub(r"```$", "", dot_code).strip()
    return dot_code


# ---------- FULL REPORT (12 STANDARD QUESTIONS) ----------

REPORT_QUESTIONS = [
    ("Problem & Motivation",
     "What problem is being solved, why is it important, and what is the research question/hypothesis?"),
    ("Prior Work & Gap",
     "What has already been done, what are its limitations, and what gap does this paper address?"),
    ("Contribution",
     "What is the paper's main new idea/contribution, and how is it different from previous work?"),
    ("Dataset",
     "What data is used, where does it come from, what are the inputs/outputs, and how is it prepared?"),
    ("End-to-End Method & Data Flow",
     "How does data move from raw input to final output? Describe every major step and explain the purpose of each step."),
    ("Mathematics & Technical Foundation",
     "What equations, algorithms, objectives, assumptions, and technical mechanisms make the method work?"),
    ("Experimental Setup",
     "How was the method trained/tested/evaluated, and what metrics and configurations were used?"),
    ("Baselines & Comparison",
     "What methods are used as baselines, why are they appropriate, and is the comparison fair?"),
    ("Results & Interpretation",
     "What do the main tables, figures, and results actually show, and do they support the research question?"),
    ("Ablation / Robustness / Generalization",
     "Which components actually matter, how robust is the method, and does it generalize?"),
    ("Limitations & Critical Evaluation",
     "What assumptions, weaknesses, biases, missing experiments, or failure conditions should we be aware of?"),
    ("Final Research Assessment",
     "What is genuinely valuable/new about this work, how convincing is it, what are its main weaknesses, and what should researchers do next?"),
]

DIAGRAM_SECTION_TITLE = "End-to-End Method & Data Flow"


def compile_report_markdown(document_name: str, results: list) -> str:
    """Assembles all 12 answered sections into one downloadable markdown report."""
    lines = [f"# Research Paper Analysis Report", f"**Document:** {document_name}", ""]
    for r in results:
        lines.append(f"## {r['index']}. {r['title']}")
        lines.append("")
        lines.append(f"*Question: {r['question']}*")
        lines.append("")
        lines.append(r["answer"])
        lines.append("")
        if r.get("diagram"):
            lines.append("```dot")
            lines.append(r["diagram"])
            lines.append("```")
            lines.append("")
        if r.get("sections_visited"):
            lines.append(f"<sub>Sections referenced: {', '.join(r['sections_visited'])}</sub>")
        lines.append("\n---\n")
    return "\n".join(lines)