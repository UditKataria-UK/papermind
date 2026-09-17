# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import os
import re
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from sentence_transformers import CrossEncoder

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# LANGGRAPH STATE
# --------------------------------------------------

class RAGState(TypedDict):
    query: str
    query_type: str
    is_general: bool
    documents: list
    message_history: list
    answer: str


# ==================================================
# PDF LOADING
# ==================================================

def load_pdf(uploaded_file):

    # Streamlit UploadedFile ko temporary PDF me save kar rahe hain
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:

        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        # Har page ke metadata me original PDF ka naam store kar rahe hain
        for document in documents:
            document.metadata["source_file"] = uploaded_file.name

        return documents

    finally:

        # Temporary PDF delete kar denge
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==================================================
# SIMPLE CHUNKING
# ==================================================

def simple_chunking(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


# ==================================================
# SEMANTIC CHUNKING
# ==================================================

def semantic_chunking(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    return splitter.split_documents(documents)


# ==================================================
# BUILD KNOWLEDGE BASE
# ==================================================

def build_knowledge_base(uploaded_files, chunking_strategy):

    all_documents = []

    # Saare uploaded PDFs load karenge
    for uploaded_file in uploaded_files:

        documents = load_pdf(uploaded_file)

        all_documents.extend(documents)

    # User ke selected strategy ke according chunking
    if chunking_strategy == "simple":

        chunks = simple_chunking(all_documents)

    else:

        chunks = semantic_chunking(all_documents)

    return chunks


# ==================================================
# CREATE VECTOR STORE
# ==================================================

def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# ==================================================
# BASIC RETRIEVAL
# ==================================================

def retrieve_documents(vector_store, query, k=5):

    documents = vector_store.similarity_search(
        query,
        k=k
    )

    return documents


# ==================================================
# GROUP DOCUMENTS BY PAPER
# ==================================================

def group_documents_by_paper(documents):

    grouped = {}

    for document in documents:

        source = document.metadata.get(
            "source_file",
            "Unknown Paper"
        )

        if source not in grouped:
            grouped[source] = []

        grouped[source].append(document)

    return grouped


# ==================================================
# RERANKING
# ==================================================

def rerank_documents(query, documents, top_k=3):

    if not documents:
        return []

    # CrossEncoder relevance score calculate karega
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    # Document + score ko saath rakh rahe hain
    scored_documents = list(
        zip(documents, scores)
    )

    # Highest score first
    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        document
        for document, score in scored_documents[:top_k]
    ]


# ==================================================
# NORMAL SEARCH
# ==================================================

def search_documents(vector_store, query):

    # Pehle FAISS se candidate documents
    documents = retrieve_documents(
        vector_store,
        query,
        k=5
    )

    # Phir CrossEncoder se better ranking
    reranked_documents = rerank_documents(
        query,
        documents,
        top_k=3
    )

    return reranked_documents


# ==================================================
# QUERY DECOMPOSITION
# ==================================================

def decompose_query(query):

    prompt = f"""
You are a query decomposition system for a research paper RAG system.

Break the user's research question into 2 to 4 focused
retrieval queries.

Rules:
- Each query should target a different aspect of the question.
- Keep queries short and useful for document retrieval.
- Do not answer the question.
- Return ONLY the retrieval queries, one per line.
- Do not use numbering.

User query:
{query}
"""

    response = llm.invoke(prompt)

    raw_queries = response.content.strip().splitlines()

    queries = []

    for line in raw_queries:

        line = line.strip()

        # Numbering remove kar rahe hain
        line = re.sub(
            r"^\d+[\.\)\-]\s*",
            "",
            line
        )

        if line:
            queries.append(line)

    # Maximum 4 queries
    queries = queries[:4]

    # Agar LLM useful query na de to original query use karo
    if not queries:
        queries = [query]

    return queries


# ==================================================
# BALANCED SEARCH
# ==================================================

def balanced_search_documents(vector_store, query):

    # Complex question ko multiple retrieval queries me todte hain
    sub_queries = decompose_query(query)

    all_documents = []

    for sub_query in sub_queries:

        documents = retrieve_documents(
            vector_store,
            sub_query,
            k=10
        )

        all_documents.extend(documents)

    # Duplicate chunks remove karna
    unique_documents = {}

    for document in all_documents:

        source = document.metadata.get(
            "source_file",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            0
        )

        content = document.page_content

        key = (
            source,
            page,
            content
        )

        unique_documents[key] = document

    documents = list(unique_documents.values())

    # Paper-wise grouping
    grouped_documents = group_documents_by_paper(
        documents
    )

    final_documents = []

    # Har paper se relevant documents retrieve karne ki
    # koshish kar rahe hain taaki comparison biased na ho
    for paper, paper_documents in grouped_documents.items():

        reranked = rerank_documents(
            query,
            paper_documents,
            top_k=2
        )

        final_documents.extend(reranked)

    return final_documents


# ==================================================
# QUERY CLASSIFICATION
# ==================================================

def classify_query(query):

    prompt = f"""
You are a query classifier for a Research Paper RAG system.

Classify the user's query into exactly ONE category:

NORMAL
- Question about one research paper
- Explanation of concepts
- Methodology
- Dataset
- Results
- Architecture
- Experiments

COMPARISON
- Comparing two or more research papers
- Asking differences between papers
- Asking similarities between papers
- Comparing methodologies, results, datasets, models, etc.

User query:
{query}

Return ONLY:
NORMAL
or
COMPARISON
"""

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    if result == "COMPARISON":
        return "COMPARISON"

    return "NORMAL"


# ==================================================
# GENERAL QUERY DETECTION
# ==================================================

def is_general_query(query):
    """
    Research RAG me jaane se pehle LLM decide karega ki
    user normal/general conversation kar raha hai ya
    research paper ke baare me question pooch raha hai.

    Isme hardcoded words nahi hain.
    """

    prompt = f"""
You are an intent classifier for a Research Paper RAG system.

Classify the user's message into exactly ONE category.

GENERAL:
- Greetings
- Casual conversation
- Small talk
- Thanks
- Asking how are you
- Asking for general help
- Casual messages
- Casual messages containing typos
- Messages that do NOT require information from the uploaded papers

RESEARCH:
- Questions about uploaded research papers
- Questions about paper content
- Methodology
- Dataset
- Results
- Experiments
- Architecture
- Models
- Findings
- Comparing papers
- Any question that requires information from the uploaded documents

Examples:

"hello" -> GENERAL
"hey bro" -> GENERAL
"hllo" -> GENERAL
"thanks a lot" -> GENERAL
"how are you?" -> GENERAL

"what is the methodology used in the paper?" -> RESEARCH
"what dataset was used?" -> RESEARCH
"compare these two papers" -> RESEARCH
"explain the architecture proposed in the paper" -> RESEARCH

User message:
{query}

Return ONLY one word:

GENERAL

or

RESEARCH
"""

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    # Sirf GENERAL ko True maanenge.
    # Baaki kisi unexpected response ko research route me bhejenge.
    if result == "GENERAL":
        return True

    return False


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(
    query,
    documents,
    message_history,
    query_type
):

    history_text = ""

    if message_history:

        history_text = "\n".join(
            [
                f'{message["role"]}: {message["content"]}'
                for message in message_history
            ]
        )

    grouped_documents = group_documents_by_paper(
        documents
    )

    context_parts = []

    for paper, paper_documents in grouped_documents.items():

        paper_context = f"\n===== {paper} =====\n"

        for document in paper_documents:

            page = document.metadata.get(
                "page",
                0
            ) + 1

            paper_context += (
                f"\n[Page {page}]\n"
                f"{document.page_content}\n"
            )

        context_parts.append(paper_context)

    context = "\n".join(context_parts)

    if query_type == "COMPARISON":

        instructions = """
You are answering a comparison question about research papers.

Compare the papers using ONLY the provided context.

Clearly separate the papers and compare their:
- methodology
- datasets
- models
- experiments
- results
- other relevant aspects

Use a table when useful.

Every factual claim taken from the papers must include
a citation in this exact format:

[Paper Name | Page X]

Do not use outside knowledge.
"""

    else:

        instructions = """
You are answering a question about research papers.

Answer using ONLY the provided paper context.

Every factual claim taken from the papers must include
a citation in this exact format:

[Paper Name | Page X]

Do not use outside knowledge.

If the answer cannot be found in the provided context,
say:

"The provided research papers do not contain enough information
to answer this question."
"""

    prompt = f"""
{instructions}

Previous conversation:
{history_text}

Research paper context:
{context}

User question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content


# ==================================================
# LANGGRAPH NODES
# ==================================================

def check_query_node(state):
    """
    Graph ka first node.

    Pehle decide karte hain ki query general hai
    ya research related.

    Isse unnecessary FAISS retrieval avoid hota hai.
    """

    is_general = is_general_query(
        state["query"]
    )

    return {
        "is_general": is_general
    }


def general_node(state):
    """
    General conversation ke liye RAG retrieval skip karte hain.

    Isliye FAISS / reranker / research classification
    run nahi hoti.
    """

    response = llm.invoke(
        state["query"]
    )

    return {
        "answer": response.content,
        "documents": []
    }


def classify_node(state):
    """
    Research query ko NORMAL ya COMPARISON me classify karta hai.
    """

    query_type = classify_query(
        state["query"]
    )

    return {
        "query_type": query_type
    }


def search_node(state, vector_store):
    """
    Normal research query ke liye
    FAISS + CrossEncoder reranking.
    """

    documents = search_documents(
        vector_store,
        state["query"]
    )

    return {
        "documents": documents
    }


def comparison_node(state, vector_store):
    """
    Comparison query ke liye:

    Query Decomposition
          ↓
    Multiple Retrievals
          ↓
    Deduplication
          ↓
    Paper-wise grouping
          ↓
    Reranking
    """

    documents = balanced_search_documents(
        vector_store,
        state["query"]
    )

    return {
        "documents": documents
    }


def generate_node(state):
    """
    Retrieved documents ke basis par final answer generate karta hai.
    """

    answer = generate_answer(
        query=state["query"],
        documents=state["documents"],
        message_history=state["message_history"],
        query_type=state["query_type"]
    )

    return {
        "answer": answer
    }


# ==================================================
# LANGGRAPH ROUTING
# ==================================================

def route_after_check(state):
    """
    General query hai to direct general node.
    Otherwise research workflow.
    """

    if state["is_general"]:
        return "general"

    return "research"


def route_query(state):
    """
    Research query ko NORMAL ya COMPARISON workflow me bhejta hai.
    """

    if state["query_type"] == "COMPARISON":
        return "comparison"

    return "normal"


# ==================================================
# BUILD RAG GRAPH
# ==================================================

def build_rag_graph(vector_store):

    graph = StateGraph(RAGState)

    # --------------------------------------------------
    # NODES
    # --------------------------------------------------

    graph.add_node(
        "check_query",
        check_query_node
    )

    graph.add_node(
        "general",
        general_node
    )

    graph.add_node(
        "classify",
        classify_node
    )

    # vector_store graph state me store nahi kar rahe.
    # Lambda ke through nodes ko vector_store de rahe hain.
    graph.add_node(
        "search",
        lambda state: search_node(
            state,
            vector_store
        )
    )

    graph.add_node(
        "comparison",
        lambda state: comparison_node(
            state,
            vector_store
        )
    )

    graph.add_node(
        "generate",
        generate_node
    )

    # --------------------------------------------------
    # START → QUERY CHECK
    # --------------------------------------------------

    graph.add_edge(
        START,
        "check_query"
    )

    # --------------------------------------------------
    # GENERAL vs RESEARCH
    # --------------------------------------------------

    graph.add_conditional_edges(
        "check_query",
        route_after_check,
        {
            "general": "general",
            "research": "classify"
        }
    )

    # General answer ke baad graph finish
    graph.add_edge(
        "general",
        END
    )

    # --------------------------------------------------
    # NORMAL vs COMPARISON
    # --------------------------------------------------

    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "normal": "search",
            "comparison": "comparison"
        }
    )

    # --------------------------------------------------
    # RETRIEVAL → GENERATION
    # --------------------------------------------------

    graph.add_edge(
        "search",
        "generate"
    )

    graph.add_edge(
        "comparison",
        "generate"
    )

    # --------------------------------------------------
    # GENERATION → END
    # --------------------------------------------------

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()