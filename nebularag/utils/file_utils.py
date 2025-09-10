"""File handling utilities."""

import os
import sys
from pathlib import Path
from typing import List, Tuple


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


def read_text_files(dir_path: str, exts: Tuple[str, ...] = (".txt", ".md")) -> List[str]:
    """
    Read all text files from a directory and its subdirectories.
    
    Args:
        dir_path: Path to the directory containing text files
        exts: Tuple of file extensions to include (case-insensitive)
        
    Returns:
        List of file contents as strings
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If no readable files are found
    """
    dir_path = validate_directory(dir_path)
    
    out: List[str] = []
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and any(file_path.name.lower().endswith(ext) for ext in exts):
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                    if content:  # Only add non-empty files
                        out.append(content)
            except (UnicodeDecodeError, PermissionError, OSError) as e:
                print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
                continue
    
    if not out:
        raise ValueError(f"No readable text files found in {dir_path}")
    
    return out


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
