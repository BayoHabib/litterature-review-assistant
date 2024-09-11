import PyPDF2
from PyPDF2.errors import PdfReadError
from typing import BinaryIO, Dict, Any

class PDFProcessor:
    def __init__(self, pdf_file: BinaryIO):
        self.pdf_file = pdf_file
        self.reader = PyPDF2.PdfReader(self.pdf_file)

    def extract_text(self) -> str:
        text = ""
        for page in self.reader.pages:
            text += page.extract_text() + "\n"
        return text

    def get_metadata(self) -> Dict[str, Any]:
        metadata = self.reader.metadata
        return {
            "title": metadata.get('/Title', 'Unknown'),
            "author": metadata.get('/Author', 'Unknown'),
            "subject": metadata.get('/Subject', 'Unknown'),
            "creator": metadata.get('/Creator', 'Unknown'),
            "producer": metadata.get('/Producer', 'Unknown'),
        }


def process_pdf(pdf_path: str) -> tuple[str, Dict[str, Any]]:
    try:
        with open(pdf_path, 'rb') as file:
            processor = PDFProcessor(file)
            text = processor.extract_text()
            metadata = processor.get_metadata()
        return text, metadata
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    except PdfReadError:
        raise ValueError(f"The file {pdf_path} is not a valid PDF.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing {pdf_path}: {str(e)}")