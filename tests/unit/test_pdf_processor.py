import unittest
from unittest.mock import mock_open, patch
from io import BytesIO
from src.data_ingestion.pdf_processor import PDFProcessor, process_pdf
from PyPDF2.errors import PdfReadError  # Add this import


class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        # Create a mock PDF content
        self.mock_pdf_content = b'%PDF-1.3 (mock PDF content)'

    @patch('PyPDF2.PdfReader')
    def test_extract_text(self, mock_pdf_reader):
        # Mock the PdfReader to return a predictable result
        mock_pdf_reader.return_value.pages = [
            type('MockPage', (), {'extract_text': lambda: 'Page 1 content'}),
            type('MockPage', (), {'extract_text': lambda: 'Page 2 content'})
        ]

        with BytesIO(self.mock_pdf_content) as pdf_file:
            processor = PDFProcessor(pdf_file)
            text = processor.extract_text()

        self.assertEqual(text, 'Page 1 content\nPage 2 content\n')

    @patch('PyPDF2.PdfReader')
    def test_get_metadata(self, mock_pdf_reader):
        # Mock the PdfReader to return predictable metadata
        mock_pdf_reader.return_value.metadata = {
            '/Title': 'Test PDF',
            '/Author': 'Test Author'
        }

        with BytesIO(self.mock_pdf_content) as pdf_file:
            processor = PDFProcessor(pdf_file)
            metadata = processor.get_metadata()

        self.assertEqual(metadata['title'], 'Test PDF')
        self.assertEqual(metadata['author'], 'Test Author')

    @patch('builtins.open', new_callable=mock_open, read_data=b'%PDF-1.3 (mock PDF content)')
    @patch('PyPDF2.PdfReader')
    def test_process_pdf(self, mock_pdf_reader, mock_file):
        # Mock the PdfReader to return predictable results
        mock_pdf_reader.return_value.pages = [
            type('MockPage', (), {'extract_text': lambda: 'Test content'})
        ]
        mock_pdf_reader.return_value.metadata = {'/Title': 'Test PDF'}

        text, metadata = process_pdf('fake_path.pdf')

        self.assertEqual(text, 'Test content\n')
        self.assertEqual(metadata['title'], 'Test PDF')

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            process_pdf('non_existent.pdf')

    @patch('builtins.open', new_callable=mock_open, read_data=b'Not a PDF')
    @patch('PyPDF2.PdfReader')
    def test_invalid_pdf(self, mock_pdf_reader, mock_file):
        mock_pdf_reader.side_effect = PdfReadError("Invalid PDF")
        
        with self.assertRaises(ValueError) as context:
            process_pdf('invalid.pdf')
        
        self.assertIn("The file invalid.pdf is not a valid PDF.", str(context.exception))


if __name__ == '__main__':
    unittest.main()