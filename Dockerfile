# PDF Knowledge Extraction & Summarization System - Offline Docker Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (following README requirements)
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (as per README installation steps)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install Python dependencies (step 2 from README)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the complete application code
COPY . .

# Create necessary directories as per README structure
RUN mkdir -p app/input app/output

# Download NLTK data (required for NLP preprocessing)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Pre-download TinyLlama model for offline operation (step 3 from README)
RUN ollama serve & \
    sleep 15 && \
    ollama pull tinyllama && \
    pkill -f ollama

# Expose Ollama port
EXPOSE 11434

# Create simple startup script
RUN echo '#!/bin/bash\n\
echo "Starting PDF Knowledge Extraction System..."\n\
ollama serve &\n\
sleep 10\n\
echo "Processing PDFs with integrated_system.py..."\n\
python integrated_system.py\n\
echo "Results generated:"\n\
echo "- Main output: challenge1b_output.json"\n\
echo "- PDF outlines: app/output/"\n\
echo "Processing complete. Container ready for result extraction."\n\
tail -f /dev/null' > /app/run.sh && chmod +x /app/run.sh

# Default command runs the main system
CMD ["/app/run.sh"] 