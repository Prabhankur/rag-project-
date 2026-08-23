"""
Chunkless RAG Chat App - multi-paper edition
Run with: streamlit run app.py

- Upload papers into a persistent library (parsed once, cached forever -
  re-uploading the same title skips re-parsing).
- Two modes:
    Single Paper        - chat/analyze one paper at a time.
    Cross-Paper          - chat across 2+ papers; answers cite which
                            paper each point comes from.
- Answers stream token-by-token live (no waiting for the full response).
- Full 12-question report (single-paper mode) streams per-question too.
"""

import streamlit as st
import tempfile
import os
import uuid
from groq import Groq

import memory
import library
from rag_engine import (
    parse_pdf_to_markdown,
    build_nested_section_tree,
    retrieve_context_multi_query,
    retrieve_context_cross_paper,
    stream_final_answer,
    generate_flow_diagram,
    compile_report_markdown,
    rewrite_standalone_question,
    REPORT_QUESTIONS,
    DIAGRAM_SECTION_TITLE,
)

st.set_page_config(page_title="Chunkless RAG Chat", layout="wide")
memory.init_db()
library.init_library_db()

MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models if this changes

# ---------- SESSION STATE ----------
defaults = {
    "session_id": None,
    "client": None,
    "messages": [],
    "mode": "single",
    "active_papers": [],  # list of {"id","title","tree"}
    "report_results": [],
    "report_markdown": None,
    "message_diagrams": {},  # {message_index: dot_string} - persists across reruns
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

analyze_clicked = False

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("⚙️ Setup")

    api_key = st.text_input("Groq API Key", type="password")
    if api_key:
        st.session_state.client = Groq(api_key=api_key.strip())

    st.divider()
    st.subheader("📄 Add Paper to Library")

    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    default_title = os.path.splitext(uploaded_file.name)[0] if uploaded_file else ""
    paper_title = st.text_input(
        "Paper title (unique ID - edit to match the paper's real title)",
        value=default_title,
    )

    if uploaded_file and st.button("Process & Add to Library", use_container_width=True):
        if not st.session_state.client:
            st.error("Enter your Groq API key first.")
        elif not paper_title.strip():
            st.error("Give the paper a title first.")
        else:
            cached = library.get_paper_by_title(paper_title)
            if cached:
                st.info(f"'{paper_title}' is already in the library ✅ — using cached structure, no re-parsing needed.")
            else:
                with st.spinner("Parsing document structure (this can take a minute)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    markdown = parse_pdf_to_markdown(tmp_path)
                    tree = build_nested_section_tree(markdown)
                    library.save_paper(paper_title, uploaded_file.name, tree)
                st.success(f"Processed and saved '{paper_title}' to the library ✅")
            st.rerun()

    st.divider()
    st.subheader("📚 Library")

    papers = library.list_papers()
    if not papers:
        st.caption("No papers yet — upload one above.")
    else:
        titles = [p["title"] for p in papers]

        mode_label = st.radio("Mode", ["🔎 Single Paper", "🔗 Cross-Paper Discussion"])
        st.session_state.mode = "single" if "Single" in mode_label else "cross"

        if st.session_state.mode == "single":
            chosen = st.selectbox("Select a paper", titles)
            selected = [p for p in papers if p["title"] == chosen]
        else:
            chosen = st.multiselect("Select 2+ papers to discuss together", titles)
            selected = [p for p in papers if p["title"] in chosen]
            if 0 < len(selected) < 2:
                st.warning("Pick at least 2 papers for cross-paper mode.")

        # Rebuild active_papers (load tree fresh each time selection changes)
        new_active = [{"id": p["id"], "title": p["title"], "tree": library.load_tree(p["id"])} for p in selected]
        active_titles = [p["title"] for p in st.session_state.active_papers]
        if [p["title"] for p in new_active] != active_titles:
            st.session_state.active_papers = new_active
            st.session_state.messages = []
            st.session_state.report_results = []
            st.session_state.report_markdown = None
            if new_active:
                st.session_state.session_id = str(uuid.uuid4())
                memory.create_session(
                    st.session_state.session_id,
                    title=" + ".join(p["title"] for p in new_active),
                    document_name=" + ".join(p["title"] for p in new_active),
                )

        with st.expander("🗑️ Manage library"):
            for p in papers:
                col1, col2 = st.columns([4, 1])
                col1.write(p["title"])
                if col2.button("🗑️", key=f"del_paper_{p['id']}"):
                    library.delete_paper(p["id"])
                    st.rerun()

        if st.session_state.mode == "single" and len(st.session_state.active_papers) == 1:
            st.divider()
            st.subheader("🧠 Full Analysis")
            st.caption("Streams answers to 12 standard research-analysis questions.")
            analyze_clicked = st.button("🧠 Analyze & Generate Report", use_container_width=True)

    st.divider()
    st.subheader("🕑 Past Chat Sessions")
    for s in memory.list_sessions():
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(s["title"], key=f"load_{s['id']}", use_container_width=True):
                st.session_state.session_id = s["id"]
                st.session_state.messages = memory.load_history(s["id"])
                st.info("Reselect the same paper(s) above to continue asking questions - chat text is restored, papers must be reselected from the library.")
        with col2:
            if st.button("🗑️", key=f"del_sess_{s['id']}"):
                memory.delete_session(s["id"])
                st.rerun()


# ---------- MAIN AREA ----------
st.title("📚 Chunkless RAG — Chat with your Papers")

if not st.session_state.active_papers:
    st.info("Add a paper to the library, then select it (Single Paper) or pick 2+ (Cross-Paper Discussion) in the sidebar.")
else:
    is_cross = st.session_state.mode == "cross" and len(st.session_state.active_papers) >= 2
    paper_names = ", ".join(p["title"] for p in st.session_state.active_papers)
    st.caption(f"**Mode:** {'🔗 Cross-Paper Discussion' if is_cross else '🔎 Single Paper'} — {paper_names}")

    # ---------- FULL REPORT (streams live, single-paper only) ----------
    if analyze_clicked and not is_cross:
        tree = st.session_state.active_papers[0]["tree"]
        doc_name = st.session_state.active_papers[0]["title"]
        results = []
        progress_bar = st.progress(0, text="Starting analysis...")

        for i, (title, question) in enumerate(REPORT_QUESTIONS, start=1):
            progress_bar.progress((i - 1) / len(REPORT_QUESTIONS), text=f"({i}/{len(REPORT_QUESTIONS)}) {title}")
            st.markdown(f"### {i}. {title}")
            st.caption(question)

            with st.spinner("Retrieving relevant sections..."):
                ctx = retrieve_context_multi_query(
                    question=question,
                    tree=tree,
                    client=st.session_state.client,
                    model=MODEL,
                    max_hops=3,
                    n_variants=2,
                )

            answer = st.write_stream(
                stream_final_answer(question, ctx["context_text"], st.session_state.client, MODEL)
            )

            diagram = ""
            if title == DIAGRAM_SECTION_TITLE and ctx["context_text"]:
                with st.spinner("Building flow diagram..."):
                    diagram = generate_flow_diagram(ctx["context_text"], st.session_state.client, MODEL)
                if diagram:
                    st.markdown("**Flow diagram:**")
                    try:
                        st.graphviz_chart(diagram)
                    except Exception:
                        st.code(diagram, language="dot")

            with st.expander("🔍 Retrieval trace"):
                st.write("**Query variants tried:**")
                for q in ctx["query_variants"]:
                    st.write(f"- {q}")
                st.write("**Sections visited:**")
                for s in ctx["sections_visited"]:
                    st.write(f"- {s}")

            results.append({
                "index": i, "title": title, "question": question,
                "answer": answer, "sections_visited": ctx["sections_visited"],
                "diagram": diagram,
            })
            st.divider()

        progress_bar.progress(1.0, text="Done ✅")
        st.session_state.report_results = results
        st.session_state.report_markdown = compile_report_markdown(doc_name, results)
        st.success("Analysis complete ✅ — see the download button below or in the Full Report tab.")
        st.download_button(
            "⬇️ Download Report (Markdown)",
            data=st.session_state.report_markdown,
            file_name=f"{doc_name}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    tab_chat, tab_report = st.tabs(["💬 Chat", "📊 Full Report"])

    # ---------- CHAT TAB ----------
    with tab_chat:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if i in st.session_state.message_diagrams:
                    st.markdown("**Flow diagram:**")
                    try:
                        st.graphviz_chart(st.session_state.message_diagrams[i])
                    except Exception:
                        st.code(st.session_state.message_diagrams[i], language="dot")

        raw_question = st.chat_input("Ask something...")

        if raw_question:
            # Resolve vague follow-ups ("explain it", "go deeper on that")
            # into a self-contained question before retrieval even runs -
            # otherwise retrieval has nothing to search on for these.
            question = rewrite_standalone_question(
                raw_question, st.session_state.messages, st.session_state.client, MODEL
            )

            st.session_state.messages.append({"role": "user", "content": raw_question})
            memory.save_message(st.session_state.session_id, "user", raw_question)
            with st.chat_message("user"):
                st.markdown(raw_question)

            assistant_msg_idx = len(st.session_state.messages)  # index this answer will get

            with st.chat_message("assistant"):
                if is_cross:
                    with st.spinner("Searching across papers..."):
                        cross = retrieve_context_cross_paper(
                            question=question,
                            papers=st.session_state.active_papers,
                            client=st.session_state.client,
                            model=MODEL,
                            chat_history=st.session_state.messages[:-1],
                        )
                    answer = st.write_stream(
                        stream_final_answer(
                            question, cross["combined_context_text"],
                            st.session_state.client, MODEL,
                            chat_history=st.session_state.messages[:-1],
                            cross_paper=True,
                        )
                    )
                    with st.expander("🔍 Retrieval trace (per paper)"):
                        if question != raw_question:
                            st.caption(f"Interpreted as: \"{question}\"")
                        for title, info in cross["context_by_paper"].items():
                            st.write(f"**{title}**")
                            for s in info["sections_visited"]:
                                st.write(f"  - {s}")
                    context_for_diagram = cross["combined_context_text"]
                else:
                    tree = st.session_state.active_papers[0]["tree"]
                    with st.spinner("Navigating document..."):
                        ctx = retrieve_context_multi_query(
                            question=question,
                            tree=tree,
                            client=st.session_state.client,
                            model=MODEL,
                            chat_history=st.session_state.messages[:-1],
                        )
                    answer = st.write_stream(
                        stream_final_answer(
                            question, ctx["context_text"],
                            st.session_state.client, MODEL,
                            chat_history=st.session_state.messages[:-1],
                        )
                    )
                    with st.expander("🔍 Retrieval trace"):
                        if question != raw_question:
                            st.caption(f"Interpreted as: \"{question}\"")
                        st.write("**Query variants tried:**")
                        for q in ctx["query_variants"]:
                            st.write(f"- {q}")
                        st.write("**Sections visited:**")
                        for s in ctx["sections_visited"]:
                            st.write(f"- {s}")
                    context_for_diagram = ctx["context_text"]

                if context_for_diagram and st.button("🗺️ Visualize as flow diagram", key=f"diag_{assistant_msg_idx}"):
                    with st.spinner("Building diagram..."):
                        dot = generate_flow_diagram(context_for_diagram, st.session_state.client, MODEL)
                    if dot:
                        st.session_state.message_diagrams[assistant_msg_idx] = dot
                        st.rerun()

            st.session_state.messages.append({"role": "assistant", "content": answer})
            memory.save_message(st.session_state.session_id, "assistant", answer)

    # ---------- REPORT TAB ----------
    with tab_report:
        if is_cross:
            st.info("Full report generation is available in Single Paper mode.")
        elif not st.session_state.report_results:
            st.info("Click '🧠 Analyze & Generate Report' in the sidebar to generate the full 12-question analysis.")
        else:
            st.download_button(
                "⬇️ Download Report (Markdown)",
                data=st.session_state.report_markdown,
                file_name="report.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_report_tab",
            )
            st.divider()
            for r in st.session_state.report_results:
                st.markdown(f"### {r['index']}. {r['title']}")
                st.caption(r["question"])
                st.markdown(r["answer"])
                if r.get("diagram"):
                    st.markdown("**Flow diagram:**")
                    try:
                        st.graphviz_chart(r["diagram"])
                    except Exception:
                        st.code(r["diagram"], language="dot")
                with st.expander("🔍 Retrieval trace"):
                    st.write("**Sections visited:**")
                    for s in r["sections_visited"]:
                        st.write(f"- {s}")
                st.divider()