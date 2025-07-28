# PDF Knowledge Extraction System - Simple Docker Commands

## Prerequisites

- Docker installed and running
- Internet connection for initial build only
- PDF files in `app/input/` directory
- `challenge1b_input.json` configured

## Build Command (run once, requires internet)

```bash
docker build -t pdf-system .
```

## Run Command (works offline after build)

```bash
docker run --name pdf-processor pdf-system
```

## Extract Results

### Copy main results
```bash
docker cp pdf-processor:/app/challenge1b_output.json ./results.json
```

### Copy PDF outlines
```bash
docker cp pdf-processor:/app/app/output ./pdf_outlines
```

## Cleanup

### Stop and remove container
```bash
docker stop pdf-processor && docker rm pdf-processor
```

### Remove image (optional)
```bash
docker rmi pdf-system
```

## What Happens

### Build
- ✓ Installs Python dependencies from `requirements.txt`
- ✓ Installs Ollama with TinyLlama model (for offline operation)
- ✓ Downloads NLTK data
- ✓ Sets up complete offline environment

### Run
- ✓ Starts Ollama service
- ✓ Processes all PDFs in `app/input/`
- ✓ Generates `challenge1b_output.json` (main results)
- ✓ Creates individual outlines in `app/output/`
- ✓ Works completely offline

### Results
- ✓ `challenge1b_output.json` - Main aggregated results
- ✓ `app/output/*.json` - Individual PDF outlines
- ✓ Persona-specific summaries based on `challenge1b_input.json` 