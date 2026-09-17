# 📚 PaperMind — Research Paper Assistant

PaperMind is an intelligent Research Paper Assistant that allows users to upload one or more research papers and ask questions about their content.

It uses a RAG (Retrieval-Augmented Generation) pipeline with **FAISS retrieval, query decomposition, CrossEncoder reranking, and LangGraph-based workflow routing** to generate grounded answers with page-level citations.

---

## 🚀 Features

- 📄 Upload one or multiple research papers
- ✂️ Choose between **Simple Chunking** and **Semantic Chunking**
- 🔎 Semantic search using **FAISS**
- 🧠 **Query Decomposition** for complex research questions
- 🎯 **CrossEncoder Reranking** for improved document relevance
- 🔀 **LangGraph-based workflow routing**
- 💬 Normal research question answering
- 📊 Multi-paper comparison
- 🤖 LLM-powered answer generation using Groq
- 📑 Page-level citations
- 💬 Conversational chat interface
- 🖥️ Streamlit-based web application

---

## 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │   Upload PDFs    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │   PDF Loading    │
                    │   PyPDFLoader    │
                    └────────┬─────────┘
                             ↓
                  ┌───────────────────────┐
                  │     Chunking          │
                  │ Simple / Semantic     │
                  └───────────┬───────────┘
                              ↓
                  ┌───────────────────────┐
                  │ HuggingFace Embeddings│
                  └───────────┬───────────┘
                              ↓
                       ┌────────────┐
                       │   FAISS    │
                       │ Vector DB  │
                       └─────┬──────┘
                             │
                       User Query
                             ↓
                  ┌───────────────────────┐
                  │   Query Classification │
                  └───────────┬───────────┘
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
                 GENERAL             RESEARCH
                    │                   ↓
                    ↓             ┌───────────┐
                 LLM Answer       │ Classify  │
                                  │ Query     │
                                  └─────┬─────┘
                                        ↓
                              ┌─────────┴─────────┐
                              │                   │
                           NORMAL            COMPARISON
                              │                   │
                              ↓                   ↓
                         Retrieval          Paper-wise
                              │              Retrieval
                              └─────────┬─────────┘
                                        ↓
                              Query Decomposition
                                        ↓
                                   FAISS Search
                                        ↓
                              CrossEncoder Reranking
                                        ↓
                                  Context Selection
                                        ↓
                                  Groq LLM
                                        ↓
                           Grounded Answer + Citations
```

---

## 🧠 How It Works

### 1. PDF Processing

Uploaded research papers are loaded using `PyPDFLoader`.

Each document is also tagged with its original PDF filename so that the generated answer can reference the correct paper.

### 2. Chunking

Paper content can be processed using two strategies:

**Simple Chunking**

Uses `RecursiveCharacterTextSplitter` with overlapping chunks.

**Semantic Chunking**

Uses `SemanticChunker` with HuggingFace embeddings to create chunks based on semantic similarity.

### 3. Embeddings

Paper chunks are converted into vector representations using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through HuggingFace embeddings.

### 4. Vector Search

The generated embeddings are stored in **FAISS**.

For a user query, FAISS retrieves the most relevant chunks from the uploaded papers.

### 5. Query Decomposition

Complex research questions are decomposed into smaller focused retrieval queries.

For example:

```text
Original Query:
"Compare the datasets, architectures and results of these papers."

        ↓

Sub-queries:
1. What datasets are used?
2. What architectures are proposed?
3. What results are reported?
```

This helps retrieve relevant information for different parts of a complex question.

### 6. CrossEncoder Reranking

Initial FAISS retrieval provides candidate documents.

These candidates are then reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The highest-scoring chunks are used as the final context.

### 7. LangGraph Workflow

LangGraph controls the application workflow.

```text
START
  ↓
Check Query
  ↓
 ┌───────────────┐
 │               │
GENERAL        RESEARCH
 │               │
 ↓               ↓
LLM           Classify
Answer           ↓
          ┌──────┴──────┐
          ↓             ↓
       NORMAL       COMPARISON
          ↓             ↓
       Search       Compare Papers
          └──────┬──────┘
                 ↓
              Generate
                 ↓
                END
```

This allows the system to route different types of user queries through different workflows.

### 8. Grounded Answer Generation

The final response is generated using the retrieved paper context.

The model is instructed to answer using the available paper context and provide citations in the following format:

```text
[Paper Name | Page X]
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web interface |
| LangChain | RAG pipeline components |
| LangGraph | Workflow orchestration |
| FAISS | Vector similarity search |
| HuggingFace Embeddings | Text embeddings |
| SemanticChunker | Semantic chunking |
| CrossEncoder | Document reranking |
| PyPDFLoader | PDF processing |
| Groq | LLM inference |
| PyTorch | Deep learning backend |

---

## 📂 Project Structure

```text
papermind/
│
├── main.py                 # RAG pipeline and LangGraph workflow
├── ui.py                   # Streamlit application
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
└── .python-version
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/UditKataria-UK/papermind.git
cd papermind
```

### 2. Create a virtual environment

Using `uv`:

```bash
uv sync
```

Or using Python:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

You can use `.env.example` as a template.

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run ui.py
```

The application will open in your browser.

---

## 💡 Example Queries

After uploading research papers, you can ask questions such as:

### Single Paper

```text
What is the main objective of this paper?
```

```text
What methodology does the paper use?
```

```text
What dataset was used in the experiments?
```

```text
What are the main findings?
```

### Complex Research Query

```text
Explain the methodology, dataset and experimental results of this paper.
```

### Multiple Papers

Upload multiple research papers and ask:

```text
Compare the methodologies used in these papers.
```

```text
Compare the datasets and experimental results of these papers.
```

---

## 🎯 Why PaperMind?

Research papers contain large amounts of technical information distributed across multiple sections.

PaperMind combines retrieval, reranking and workflow-based reasoning to make it easier to:

- Understand research papers
- Find specific information
- Compare multiple papers
- Retrieve relevant sections
- Get answers grounded in the uploaded documents

---

## 🔬 Key RAG Components

PaperMind demonstrates several practical RAG concepts:

```text
Document Loading
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Search
      ↓
Query Decomposition
      ↓
Reranking
      ↓
Context Selection
      ↓
LLM Generation
      ↓
Citation
```

---

## 🌐 Live Demo

🚀 **PaperMind:**  
[Add your Streamlit Cloud URL here]

---

## 📌 Future Improvements

Potential future improvements include:

- RAG evaluation and benchmarking
- More advanced citation verification
- Persistent conversation memory
- Additional research-specific workflows
- Deployment and performance optimizations

---

## 👨‍💻 Author

**Udit Kataria**

B.Tech — Artificial Intelligence & Machine Learning

GitHub: [UditKataria-UK](https://github.com/UditKataria-UK)

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.