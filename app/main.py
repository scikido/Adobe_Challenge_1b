from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os
import shutil
import tempfile
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append('..')

# Import the processing functions
from process_pdfs import process_pdf_files
from integrated_system import aggregate_relevant_sections

app = FastAPI(
    title="PDF Knowledge Extraction API",
    description="Simple API to process PDFs and find relevant sections",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Input model
class DocumentInfo(BaseModel):
    filename: str
    title: str

class ProcessingRequest(BaseModel):
    persona: dict
    query_text: dict
    documents: List[DocumentInfo]

@app.post("/process-pdfs")
async def process_pdfs_only(files: List[UploadFile] = File(...)):
    """
    Process PDFs only - extracts outlines and saves to output directory.
    This endpoint only runs process_pdfs.py functionality.
    
    Input: PDF files uploaded via multipart form
    Output: Confirmation that PDF processing is complete
    """
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / "input"
            temp_output = Path(temp_dir) / "output"
            temp_input.mkdir()
            temp_output.mkdir()
            
            # Save uploaded PDFs to temp input directory
            uploaded_files = []
            for file in files:
                if not file.filename.lower().endswith('.pdf'):
                    raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
                
                file_path = temp_input / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                uploaded_files.append(file.filename)
            
            # Process PDFs using process_pdfs.py with custom directories
            process_pdf_files(input_dir=str(temp_input), output_dir=str(temp_output))
            
            # Copy results to app/output directory
            app_output = Path("../app/output")
            app_output.mkdir(exist_ok=True)
            
            for json_file in temp_output.glob("*.json"):
                shutil.copy2(json_file, app_output / json_file.name)
        
        return {
            "message": "PDF processing completed successfully",
            "status": "success",
            "files_processed": uploaded_files,
            "output_directory": "../app/output"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_pdfs(
    files: List[UploadFile] = File(...),
    persona: str = "HR professional",
    query_text: str = "Find relevant information",
    max_sections: int = 5,
    max_subsections: int = 5
):
    """
    Complete workflow - process PDFs and return relevant sections based on query text.
    This endpoint runs the complete integrated_system.py workflow.
    
    Input: PDF files + query parameters
    Output: JSON with extracted sections and summaries
    """
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / "input"
            temp_output = Path(temp_dir) / "output"
            temp_input.mkdir()
            temp_output.mkdir()
            
            # Save uploaded PDFs to temp input directory
            uploaded_files = []
            for file in files:
                if not file.filename.lower().endswith('.pdf'):
                    raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
                
                file_path = temp_input / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                uploaded_files.append(file.filename)
            
            # Process PDFs using process_pdfs.py with custom directories
            process_pdf_files(input_dir=str(temp_input), output_dir=str(temp_output))
            
            # Get relevant sections using integrated_system.py
            result = aggregate_relevant_sections(
                str(temp_output), 
                query_text, 
                persona, 
                uploaded_files,
                max_sections=max_sections, 
                max_subsections=max_subsections
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "PDF Knowledge Extraction API", 
        "endpoints": {
            "process_pdfs": "/process-pdfs - Upload PDFs and process them only",
            "process": "/process - Upload PDFs and run complete workflow with query processing"
        },
        "usage": {
            "process_pdfs": "POST with PDF files via multipart form",
            "process": "POST with PDF files + query parameters via multipart form"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 