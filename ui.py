import streamlit as st

from main import (
    build_knowledge_base,
    create_vector_store,
    build_rag_graph
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "rag_graph" not in st.session_state:
    st.session_state["rag_graph"] = None


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📚 Research Paper RAG")

    st.caption(
        "Upload research papers and ask questions "
        "using an intelligent RAG workflow."
    )

    st.divider()

    # --------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------

    st.subheader("📄 Upload Papers")

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # --------------------------------------------------
    # CHUNKING STRATEGY
    # --------------------------------------------------

    st.subheader("🧩 Chunking")

    chunking_strategy = st.radio(
        "Select strategy",
        ["simple", "semantic"],
        help=(
            "Simple uses fixed-size chunks. "
            "Semantic creates chunks based on meaning."
        )
    )

    st.divider()

    # --------------------------------------------------
    # PROCESS PAPERS
    # --------------------------------------------------

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} PDF(s) selected"
        )

        if st.button(
            "🚀 Process Papers",
            use_container_width=True
        ):

            # ------------------------------------------
            # STEP 1 — LOAD + CHUNK
            # ------------------------------------------

            with st.status(
                "Processing research papers...",
                expanded=True
            ) as status:

                st.write("📄 Loading PDF documents...")

                chunks = build_knowledge_base(
                    uploaded_files,
                    chunking_strategy
                )

                st.write(
                    f"🧩 Created {len(chunks)} chunks"
                )

                # --------------------------------------
                # STEP 2 — VECTOR STORE
                # --------------------------------------

                st.write(
                    "🔢 Creating embeddings and FAISS index..."
                )

                vector_store = create_vector_store(
                    chunks
                )

                # --------------------------------------
                # STEP 3 — LANGGRAPH
                # --------------------------------------

                st.write(
                    "🧠 Building LangGraph workflow..."
                )

                rag_graph = build_rag_graph(
                    vector_store
                )

                # --------------------------------------
                # SAVE EVERYTHING
                # --------------------------------------

                st.session_state["vector_store"] = (
                    vector_store
                )

                st.session_state["rag_graph"] = (
                    rag_graph
                )

                # New papers = fresh conversation
                st.session_state["message_history"] = []

                status.update(
                    label="Papers processed successfully!",
                    state="complete",
                    expanded=False
                )


    # --------------------------------------------------
    # PROJECT STATUS
    # --------------------------------------------------

    st.divider()

    st.subheader("⚙️ System Status")

    if st.session_state["vector_store"] is not None:

        st.success("Knowledge Base: Ready")
        st.success("RAG Workflow: Ready")

    else:

        st.info("Upload and process papers to begin.")


    # --------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------

    if st.session_state["message_history"]:

        st.divider()

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state["message_history"] = []

            st.rerun()


# ==================================================
# MAIN PAGE
# ==================================================

st.title("Research Paper Assistant")

st.markdown(
    """
Ask questions about your uploaded research papers.

The system uses **FAISS retrieval, CrossEncoder reranking,
query decomposition, and LangGraph-based routing** to
generate grounded answers with page-level citations.
"""
)


# ==================================================
# EMPTY STATE
# ==================================================

if st.session_state["vector_store"] is None:

    st.divider()

    st.info(
        "📄 Upload one or more research papers from the "
        "sidebar and click **Process Papers** to start."
    )

    st.stop()


# ==================================================
# CHAT HISTORY
# ==================================================

for message in st.session_state["message_history"]:

    if message["role"] == "user":

        with st.chat_message("user"):

            st.write(
                message["content"]
            )

    else:

        with st.chat_message("assistant"):

            st.write(
                message["content"]
            )


# ==================================================
# CHAT INPUT
# ==================================================

query = st.chat_input(
    "Ask something about your research papers..."
)


if query:

    # --------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": query
        }
    )


    # --------------------------------------------------
    # SHOW USER MESSAGE
    # --------------------------------------------------

    with st.chat_message("user"):

        st.write(query)


    # --------------------------------------------------
    # RUN LANGGRAPH
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your query..."
        ):

            result = st.session_state[
                "rag_graph"
            ].invoke(
                {
                    "query": query,
                    "query_type": "",
                    "is_general": False,
                    "documents": [],
                    "message_history": (
                        st.session_state[
                            "message_history"
                        ]
                    ),
                    "answer": ""
                }
            )


        # ==================================================
        # WORKFLOW INDICATOR
        # ==================================================

        if result["is_general"]:

            st.caption(
                "💬 General conversation"
            )

        elif result["query_type"] == "COMPARISON":

            st.caption(
                "⚖️ Comparison workflow"
            )

        else:

            st.caption(
                "🔎 Standard retrieval workflow"
            )


        # ==================================================
        # ANSWER
        # ==================================================

        answer = result["answer"]

        st.write(answer)


        # --------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # --------------------------------------------------

        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # ==================================================
        # SOURCES
        # ==================================================

        documents = result["documents"]

        if (
            not result["is_general"]
            and documents
        ):

            with st.expander(
                f"📚 Sources ({len(documents)})"
            ):

                for i, document in enumerate(
                    documents,
                    1
                ):

                    source = document.metadata.get(
                        "source_file",
                        "Unknown"
                    )

                    page = document.metadata.get(
                        "page",
                        0
                    ) + 1

                    st.markdown(
                        f"**Source {i}**  \n"
                        f"📄 `{source}`  \n"
                        f"📖 Page **{page}**"
                    )

                    # Actual retrieved chunk
                    with st.expander(
                        "View retrieved content"
                    ):

                        st.write(
                            document.page_content
                        )

                    if i < len(documents):

                        st.divider()