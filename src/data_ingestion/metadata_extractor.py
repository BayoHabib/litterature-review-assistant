# src/data_ingestion/metadata_extractor.py

import PyPDF2
from typing import Dict, Any, Optional, BinaryIO
import re
from datetime import datetime

class MetadataExtractor:
    def __init__(self, pdf_file: BinaryIO):
        self.pdf_file = pdf_file

    def extract_metadata(self) -> Dict[str, Any]:
        reader = PyPDF2.PdfReader(self.pdf_file)
        metadata = reader.metadata
        
        # Extract text from the first page for additional processing
        first_page_text = reader.pages[0].extract_text()

        extracted_metadata = {
            "title": self._extract_title(metadata, first_page_text),
            "authors": self._extract_authors(metadata, first_page_text),
            "publication_date": self._extract_publication_date(metadata, first_page_text),
            "abstract": self._extract_abstract(reader),
            "keywords": self._extract_keywords(first_page_text),
            "doi": self._extract_doi(first_page_text),
        }

        return extracted_metadata


    def _extract_title(self, metadata: Dict[str, Any], first_page_text: str) -> str:
        if metadata.get('/Title'):
            return metadata['/Title']
        # Fallback: Try to extract title from the first page text
        # This is a simple heuristic and might need refinement
        lines = first_page_text.split('\n')
        return lines[0] if lines else "Unknown Title"

    def _extract_authors(self, metadata: Dict[str, Any], first_page_text: str) -> list:
        if metadata.get('/Author'):
            return [author.strip() for author in metadata['/Author'].split(',')]
        # Fallback: Try to extract authors from the first page text
        # This is a simple heuristic and might need refinement
        author_line = re.search(r'(?<=\n).*?(?=\n)', first_page_text)
        if author_line:
            return [author.strip() for author in author_line.group().split(',')]
        return ["Unknown Author"]

    def _extract_publication_date(self, metadata: Dict[str, Any], first_page_text: str) -> Optional[str]:
        if metadata.get('/CreationDate'):
            # Parse the date string from metadata
            date_str = metadata['/CreationDate'][2:10]  # Format: 'D:YYYYMMDD'
            return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        # Fallback: Try to find a date in the first page text
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', first_page_text)
        return date_match.group() if date_match else None

    def _extract_abstract(self, reader: PyPDF2.PdfReader) -> Optional[str]:
        # Simple heuristic: Look for "Abstract" and extract the following paragraph
        for page in reader.pages[:3]:  # Check first 3 pages
            text = page.extract_text()
            abstract_match = re.search(r'(?i)abstract\s*(.*?)\n\n', text, re.DOTALL)
            if abstract_match:
                return abstract_match.group(1).strip()
        return None

    def _extract_keywords(self, first_page_text: str) -> list:
        # Look for keywords section
        keyword_match = re.search(r'(?i)keywords?:?\s*(.*?)(?:\n\n|\Z)', first_page_text, re.DOTALL)
        if keyword_match:
            keywords = keyword_match.group(1)
            return [kw.strip() for kw in keywords.split(',')]
        return []

    def _extract_doi(self, first_page_text: str) -> Optional[str]:
        doi_match = re.search(r'\b(10\.\d{4,}(?:\.\d+)*\/\S+)\b', first_page_text)
        return doi_match.group() if doi_match else None

