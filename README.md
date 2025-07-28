# PDF Knowledge Extraction & Summarization System

## Overview

This project is designed to process a collection of PDF documents, extract their structural outlines, and identify the most relevant sections based on a user-defined persona and job-to-be-done. It then generates concise, persona- and task-specific summaries of these sections using an LLM (TinyLlama via Ollama). The system is modular, leveraging NLP, ML, and PDF parsing techniques to automate knowledge extraction for use cases like travel planning, research, or content curation.


### Working Diagram:
<img width="685" height="429" alt="Screenshot 2025-07-28 at 9 40 20 PM" src="https://github.com/user-attachments/assets/a3692238-f524-4d10-9a52-494c5774439f" />

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
- **Context Query Construction:** The persona, job description, and document title are combined to form a context query, representing the user's information need.
- **TF-IDF Vectorization:** Both the context query and all section texts are vectorized using TF-IDF, capturing the importance of terms relative to the corpus.
- **Cosine Similarity:** The system computes cosine similarity between the context query and each section, quantifying their semantic relevance.
- **Heuristic Boosts:** Sections with keywords like 'introduction', 'overview', or 'objective' receive a relevance boost, as these are often high-level summaries.
- **Fallback Matching:** If no section meets the minimum relevance threshold, a fallback keyword-based matching ensures at least some results are returned.

### 3. Section Selection & Summarization
- **Top Section Selection:** The most relevant sections (by score) are selected for further analysis. The number of sections is configurable.
- **Subsection Extraction:** For each top section, the system can further extract and analyze subsections or sub-content if present.
- **LLM Summarization:** Each selected section (or its sub-content) is summarized using a local LLM (TinyLlama via Ollama). The prompt is tailored to the persona and job, ensuring the summary is context-aware and actionable.

### 4. Aggregation & Output Generation
- **Result Aggregation:** The system aggregates metadata, selected sections, and their summaries into a single output JSON.
- **Output Structure:** The output includes:
  - Metadata (input documents, persona, job, timestamp)
  - Extracted sections (document, section title, importance rank, page number)
  - Subsection analysis (summaries for each top section)
- **Extensibility:** The modular design allows for easy adaptation to new personas, jobs, or document types by adjusting the input JSON and/or the NLP pipeline.

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
  - The job to be done (task)
  - The list of document filenames

**Example:**
```json
{
  "persona": { "role": "Travel Planner" },
  "job_to_be_done": { "task": "Plan a trip of 4 days for a group of 10 college friends." },
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
  - Metadata (input docs, persona, job, timestamp)
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

## Notes

- The system requires a running Ollama server with the TinyLlama model for summarization.
- All intermediate outlines are stored as JSON in `app/output/`.
- The code is modular and can be extended for other personas, jobs, or document types.

--- 
