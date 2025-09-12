"""File handling utilities."""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Try to import PDF processing library
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def validate_directory(dir_path: str) -> Path:
    """
    Validate that a directory exists and is accessible.
    
    Args:
        dir_path: Path to the directory to validate
        
    Returns:
        Path object for the validated directory
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If the path is not a directory
        PermissionError: If the directory is not accessible
    """
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {dir_path}")
    
    if not os.access(dir_path, os.R_OK):
        raise PermissionError(f"Directory is not readable: {dir_path}")
    
    return dir_path


def read_text_files(dir_path: str, exts: Tuple[str, ...] = (".txt", ".md", ".pdf")) -> List[str]:
    """
    Read all text files from a directory and its subdirectories.
    Supports text files (.txt, .md) and PDF files (.pdf).
    
    Args:
        dir_path: Path to the directory containing text files
        exts: Tuple of file extensions to include (case-insensitive)
        
    Returns:
        List of file contents as strings
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If no readable files are found
        ImportError: If PDF processing is requested but PyPDF2 is not available
    """
    dir_path = validate_directory(dir_path)
    
    # Check if PDF processing is needed and available
    if ".pdf" in exts and not PDF_AVAILABLE:
        raise ImportError(
            "PDF processing requested but PyPDF2 is not available. "
            "Install it with: pip install PyPDF2"
        )
    
    out: List[str] = []
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and any(file_path.name.lower().endswith(ext) for ext in exts):
            try:
                if file_path.suffix.lower() == ".pdf":
                    # Handle PDF files
                    content = extract_pdf_text(str(file_path))
                    if content:  # Only add non-empty files
                        out.append(content)
                else:
                    # Handle text files
                    with open(file_path, "r", encoding="utf-8") as fh:
                        content = fh.read().strip()
                        if content:  # Only add non-empty files
                            out.append(content)
            except (UnicodeDecodeError, PermissionError, OSError, ValueError, ImportError) as e:
                print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
                continue
    
    if not out:
        raise ValueError(f"No readable text files found in {dir_path}")
    
    return out


def extract_pdf_text(file_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
        
    Raises:
        ImportError: If PyPDF2 is not available
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a valid PDF or is empty
    """
    if not PDF_AVAILABLE:
        raise ImportError(
            "PyPDF2 is required for PDF processing. "
            "Install it with: pip install PyPDF2"
        )
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")
    
    try:
        with open(path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            if len(pdf_reader.pages) == 0:
                raise ValueError(f"PDF file is empty: {file_path}")
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_content.append(page_text.strip())
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1} in {file_path}: {e}", file=sys.stderr)
                    continue
            
            if not text_content:
                raise ValueError(f"No text content found in PDF: {file_path}")
            
            return "\n\n".join(text_content)
            
    except Exception as e:
        if isinstance(e, (ImportError, FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Error reading PDF file {file_path}: {e}")


def get_file_info(file_path: str) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    stat = path.stat()
    
    return {
        "name": path.name,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "extension": path.suffix,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }
