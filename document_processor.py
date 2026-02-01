"""
Document processor module for extracting text from various file formats.
Supports PDF, DOCX, and Markdown files with metadata extraction.
"""

import os
from typing import Dict, List, Tuple
from datetime import datetime
import PyPDF2
import pdfplumber
from docx import Document


class DocumentProcessor:
    """Handles extraction of text from various document formats."""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.md', '.txt'}
    
    def __init__(self):
        pass
    
    @staticmethod
    def clean_metadata_for_chromadb(metadata: Dict) -> Dict:
        """
        Clean metadata to only include ChromaDB-compatible types.
        ChromaDB only accepts: str, int, float, bool, or None.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Cleaned metadata dictionary with only simple types
        """
        cleaned = {}
        
        for key, value in metadata.items():
            # Skip complex types (lists, dicts)
            if isinstance(value, (str, int, float, bool, type(None))):
                cleaned[key] = value
            elif isinstance(value, list):
                # For lists, we can store the count or skip
                if key == 'page_texts':
                    # Skip page_texts as it's too complex
                    continue
                elif key == 'sections':
                    # Skip sections as it's too complex
                    continue
                else:
                    # For other lists, just skip or convert to string if needed
                    continue
            elif isinstance(value, dict):
                # Skip nested dictionaries
                continue
        
        return cleaned
    
    def process_file(self, file_path: str) -> Dict[str, any]:
        """
        Process a file and extract text with metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing extracted text, metadata, and file info
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {self.SUPPORTED_FORMATS}")
        
        # Extract text based on file type
        if file_ext == '.pdf':
            text, metadata = self._extract_pdf(file_path)
        elif file_ext == '.docx':
            text, metadata = self._extract_docx(file_path)
        elif file_ext in ['.md', '.txt']:
            text, metadata = self._extract_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Add common metadata
        metadata.update({
            'source': os.path.basename(file_path),
            'file_path': file_path,
            'file_type': file_ext[1:],  # Remove the dot
            'processed_at': datetime.now().isoformat()
        })
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    def _extract_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from PDF using pdfplumber with PyPDF2 fallback."""
        text_parts = []
        metadata = {'pages': 0, 'page_texts': []}
        
        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                metadata['pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    metadata['page_texts'].append({
                        'page_number': page_num,
                        'text': page_text
                    })
        except Exception as e:
            # Fallback to PyPDF2
            print(f"pdfplumber failed, using PyPDF2 fallback: {e}")
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata['pages'] = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages, start=1):
                        page_text = page.extract_text() or ""
                        text_parts.append(page_text)
                        metadata['page_texts'].append({
                            'page_number': page_num,
                            'text': page_text
                        })
            except Exception as e2:
                raise Exception(f"Failed to extract PDF with both methods: {e2}")
        
        full_text = "\n\n".join(text_parts)
        return full_text, metadata
    
    def _extract_docx(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        
        paragraphs = []
        metadata = {
            'paragraphs': 0,
            'sections': []
        }
        
        current_section = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
                current_section.append(text)
                
                # Detect potential section headers (simple heuristic)
                if para.style.name.startswith('Heading'):
                    if current_section:
                        metadata['sections'].append({
                            'heading': current_section[0] if current_section else '',
                            'content': '\n'.join(current_section)
                        })
                        current_section = []
        
        # Add last section
        if current_section:
            metadata['sections'].append({
                'heading': '',
                'content': '\n'.join(current_section)
            })
        
        metadata['paragraphs'] = len(paragraphs)
        full_text = "\n\n".join(paragraphs)
        
        return full_text, metadata
    
    def _extract_text(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from plain text or markdown files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        metadata = {
            'lines': len(text.split('\n')),
            'characters': len(text)
        }
        
        return text, metadata
    
    def process_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, any]:
        """
        Process an uploaded file from memory.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        # Create temporary file
        import tempfile
        
        file_ext = os.path.splitext(filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            result = self.process_file(tmp_path)
            result['metadata']['original_filename'] = filename
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
