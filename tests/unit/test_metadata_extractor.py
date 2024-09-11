# tests/unit/test_metadata_extractor.py

import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from src.data_ingestion.metadata_extractor import MetadataExtractor

class TestMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.mock_pdf_file = BytesIO(b"Mock PDF content")
        self.extractor = MetadataExtractor(self.mock_pdf_file)

    @patch('PyPDF2.PdfReader')
    def test_extract_title(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.metadata = {'/Title': 'Sample Title'}
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = ""
        mock_pdf_reader.return_value = mock_reader

        metadata = self.extractor.extract_metadata()
        self.assertEqual(metadata['title'], 'Sample Title')

    @patch('PyPDF2.PdfReader')
    def test_extract_authors(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.metadata = {'/Author': 'John Doe, Jane Smith'}
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = ""
        mock_pdf_reader.return_value = mock_reader

        metadata = self.extractor.extract_metadata()
        self.assertEqual(metadata['authors'], ['John Doe', 'Jane Smith'])

    @patch('PyPDF2.PdfReader')
    def test_extract_publication_date(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.metadata = {'/CreationDate': 'D:20210101'}
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = ""
        mock_pdf_reader.return_value = mock_reader

        metadata = self.extractor.extract_metadata()
        self.assertEqual(metadata['publication_date'], '2021-01-01')

    @patch('PyPDF2.PdfReader')
    def test_extract_abstract(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.metadata = {}
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = "Abstract\nThis is a sample abstract.\n\nIntroduction"
        mock_pdf_reader.return_value = mock_reader

        metadata = self.extractor.extract_metadata()
        self.assertEqual(metadata['abstract'], 'This is a sample abstract.')

    @patch('PyPDF2.PdfReader')
    def test_extract_keywords(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.metadata = {}
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = "Keywords: AI, Machine Learning, Data Science"
        mock_pdf_reader.return_value = mock_reader

        metadata = self.extractor.extract_metadata()
        self.assertEqual(metadata['keywords'], ['AI', 'Machine Learning', 'Data Science'])

    @patch('PyPDF2.PdfReader')
    def test_extract_doi(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.metadata = {}
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = "DOI: 10.1234/abcd.5678"
        mock_pdf_reader.return_value = mock_reader

        metadata = self.extractor.extract_metadata()
        self.assertEqual(metadata['doi'], '10.1234/abcd.5678')

if __name__ == '__main__':
    unittest.main()