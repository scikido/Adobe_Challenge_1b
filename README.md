# PDF Knowledge Extraction & Summarization System

## Overview

This project is designed to process a collection of PDF documents, extract their structural outlines, and identify the most relevant sections based on a user-defined persona and text similarity query. It then generates concise, persona- and context-specific summaries of these sections using an LLM (TinyLlama via Ollama). The system is modular, leveraging NLP, ML, and PDF parsing techniques to automate knowledge extraction for use cases like research, content discovery, or information retrieval.


### Working Diagram:
<img width="685" height="429" alt="Screenshot 2025-07-28 at 9 40 20 PM" src="https://github.com/user-attachments/assets/a3692238-f524-4d10-9a52-494c5774439f" />

---

## Approach

The system follows a multi-stage pipeline to transform raw PDFs into actionable, context-aware summaries:

### 1. PDF Parsing & Outline Extraction
- **PDF Loading:** All PDF files placed in `app/input/` are processed using `pdfplumber` to extract text and layout information from each page.
- **Header/Footer Removal:** The system detects and removes repeated headers and footers by analyzing text position, font size, and frequency across pages, ensuring only core content is retained.
- **Line Grouping & Annotation:** Characters are grouped into lines based on their vertical position. Each line is annotated with metadata such as font size, color, and position.
- **Heading Detection:** Headings are detected using a combination of font size outliers, line spacing, and character count heuristics. Both font-based and spacing-based methods are used to robustly identify section titles and subheadings.
- **Outline Construction:** Detected headings and their associated content are structured into a hierarchical outline (JSON), capturing the document's logical structure. These outlines are saved in `app/output/` for downstream processing.

### 2. NLP Preprocessing & Relevance Scoring
- **Text Preprocessing:** All section titles and content are preprocessed using NLTK: tokenization, stopword removal, and lemmatization. This standardizes the text and improves downstream matching.
- **Context Query Construction:** The persona, query text, and document title are combined to form a context query, representing the user's information need.
- **TF-IDF Vectorization:** Both the context query and all section texts are vectorized using TF-IDF, capturing the importance of terms relative to the corpus.
- **Cosine Similarity:** The system computes cosine similarity between the context query and each section, quantifying their semantic relevance.
- **Heuristic Boosts:** Sections with keywords like 'introduction', 'overview', or 'objective' receive a relevance boost, as these are often high-level summaries.
- **Fallback Matching:** If no section meets the minimum relevance threshold, a fallback keyword-based matching ensures at least some results are returned.

### 3. Section Selection & Summarization
- **Top Section Selection:** The most relevant sections (by score) are selected for further analysis. The number of sections is configurable.
- **Subsection Extraction:** For each top section, the system can further extract and analyze subsections or sub-content if present.
- **LLM Summarization:** Each selected section (or its sub-content) is summarized using a local LLM (TinyLlama via Ollama). The prompt is tailored to the persona and query context, ensuring the summary is context-aware and actionable.

### 4. Aggregation & Output Generation
- **Result Aggregation:** The system aggregates metadata, selected sections, and their summaries into a single output JSON.
- **Output Structure:** The output includes:
  - Metadata (input documents, persona, query text, timestamp)
  - Extracted sections (document, section title, importance rank, page number)
  - Subsection analysis (summaries for each top section)
- **Extensibility:** The modular design allows for easy adaptation to new personas, query types, or document types by adjusting the input JSON and/or the NLP pipeline.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/scikido/Adobe_Challenge_1b.git
   cd Adobe_Challenge_1b
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run Ollama with TinyLlama**
   - [Install Ollama](https://ollama.com/download)
   - Pull the TinyLlama model:
     ```bash
     ollama pull tinyllama
     ```
   - Start the Ollama server (if not already running):
     ```bash
     ollama serve
     ```

---

## Usage

### 1. Prepare Input

- Place your PDF files in the `app/input/` directory.
- Create an input JSON (e.g., `challenge1b_input.json`) specifying:
  - The persona (role)
  - The query text (what you want to find similar content for)
  - The list of document filenames

**Example:**
```json
{
  "persona": { "role": "Travel Planner" },
  "query_text": { "text": "Find information about travel destinations and cultural experiences" },
  "documents": [
    { "filename": "South of France - Cities.pdf", "title": "South of France - Cities" },
    ...
  ]
}
```

### 2. Run the System

```bash
python integrated_system.py
```

- This will:
  - Process all PDFs in `app/input/` and generate outlines in `app/output/`.
  - Read your input JSON, extract and summarize relevant sections, and write the results to `challenge1b_output.json`.

### 3. Review Output

- The output JSON (e.g., `challenge1b_output.json`) contains:
  - Metadata (input docs, persona, query text, timestamp)
  - Extracted sections (with document, section title, importance rank, page number)
  - Subsection analysis (summaries for each top section)

**Example Output:**
```json
{
  "metadata": { ... },
  "extracted_sections": [
    {
      "document": "South of France - Cities.pdf",
      "section_title": "Travel Tips",
      "importance_rank": 1,
      "page_number": 1
    },
    ...
  ],
  "subsection_analysis": [
    {
      "document": "South of France - Cities.pdf",
      "refined_text": "Montpellier is a city located in southeastern France...",
      "page_number": 1
    },
    ...
  ]
}
```

---

## FastAPI Interface

The system also provides a REST API interface for programmatic access and integration with other services.

### Starting the API Server

**Linux/Mac:**
```bash
cd app
chmod +x start_api.sh
./start_api.sh
```

**Windows:**
```cmd
cd app
start_api.bat
```

**Manual start:**
```bash
cd app
pip install -r requirements_api.txt
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. **GET /** - API Information
Returns information about available endpoints and API version.

#### 2. **GET /health** - Health Check
Returns the health status of the API server.

#### 3. **POST /upload-pdfs** - Upload and Process PDFs
Upload PDF files and start processing them.

**Parameters:**
- `files`: List of PDF files to upload
- `persona`: Role of the person (e.g., "Travel Planner")
- `query_text`: Text to find similar content for

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "uploaded",
  "message": "Successfully uploaded X PDF files. Processing started."
}
```

#### 4. **POST /process-documents** - Process Existing Documents
Process documents that are already in the input directory.

**Request Body:**
```json
{
  "persona": {"role": "Travel Planner"},
  "query_text": {"text": "Find information about travel destinations and cultural experiences"},
  "documents": [
    {"filename": "South of France - Cities.pdf", "title": "South of France - Cities"}
  ]
}
```

#### 5. **GET /job-status/{job_id}** - Check Job Status
Monitor the progress of a processing job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": "Processing completed successfully",
  "result": { ... },
  "error": null
}
```

#### 6. **GET /list-jobs** - List All Jobs
Get a list of all processing jobs and their statuses.

### Testing the API

Use the provided test script to verify the API functionality:

```bash
cd app
python test_api.py
```

### API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI) or `http://localhost:8000/redoc` for ReDoc documentation.

---

## Notes

- The system requires a running Ollama server with the TinyLlama model for summarization.
- All intermediate outlines are stored as JSON in `app/output/`.
- The code is modular and can be extended for other personas, query types, or document types.
- The FastAPI interface provides asynchronous processing with background tasks for better performance.
- The system now focuses on finding similar text content rather than specific job scenarios, making it more flexible for general information retrieval tasks.

--- 
