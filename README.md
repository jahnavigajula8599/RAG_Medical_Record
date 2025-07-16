# RAG_Medical_Record
From Unstructured EHR’s to Trustworthy Clinical Answers
Here’s a `README.md` for your GitHub repository containing the RAG demo using **Ollama + DeepSeek 8B + OpenSearch** for clinical document Q\&A.

---

## 🧠 RAG (Retrieval-Augmented Generation) for Healthcare

This project demonstrates a **local RAG pipeline** to answer clinical questions from medical documents using:

* **Ollama** with `deepseek-r1:8b`
* **OpenSearch** for vector search
* **Sentence Transformers** for embedding
* **PyMuPDF + OCR (Tesseract)** for parsing PDF content

---

### 📁 Repository Structure

| File                                                      | Description                                                                                                                                     |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `pdf_to_text_local.py`                                    | Converts a local PDF into structured text with `[PAGE X START] ... [PAGE X END]` delimiters. Handles both extractable text and image-based OCR. |
| `Ollama_deepseek_8b_rag.py`                               | Full RAG pipeline: indexes document into OpenSearch, retrieves top-k relevant pages, and queries Ollama with context.                           |
| `RAG (Retrieval Augmented Generation) in Healthcare.pptm` | PowerPoint presentation explaining the RAG architecture, use case in healthcare, and demo screenshots.                                          |

---

### ⚙️ Requirements

```bash
pip install sentence-transformers opensearch-py pytesseract pymupdf Pillow
```

* **Ollama** must be running locally with the model `deepseek-r1:8b` pulled:

  ```bash
  ollama pull deepseek-r1:8b
  ollama serve
  ```

* **OpenSearch** must be running locally (default `localhost:9200`).

* For OCR:

  * Ensure [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed and in your system path.

---

### 🏁 Quick Start

1. **Convert PDF to Text**

```bash
python pdf_to_text_local.py
```

* Input: PDF file (configured inside the script)
* Output: `.txt` file with page tags (e.g., `Hyponatremia Synthetic Pos Complex.txt`)

2. **Run the RAG Demo**

```bash
python Ollama_deepseek_8b_rag.py
```

This:

* Indexes the `.txt` file into OpenSearch
* Retrieves top-1 relevant page for each question
* Sends the context + question to Ollama for final answer

---

### ❓Example Questions Used

* “Did patient have a rash?”
* “What is the reason patient admitted?”

Each result includes:

* Matched page
* Retrieved context
* Final LLM answer

---

### 🧩 Architecture Diagram (see slides)

* **PDF Parser →** Converts PDF into clean, paginated text
* **Embedder →** Encodes each page using `intfloat/e5-base-v2`
* **OpenSearch →** Stores and retrieves top-k pages using cosine similarity
* **Ollama →** Generates final response using retrieved context

---

### 🏥 Use Case: Clinical Chart QA

This RAG system helps answer clinical questions from scanned or structured patient records, enabling:

* Auditing support
* Coding assistance
* Medical review automation

---

### 📌 Notes

* Supports page-level provenance (i.e., you know which page answer came from)
* Adaptable to other embedding models or LLMs
* Currently uses **E5-base-v2**, but can be replaced with domain-specific models


