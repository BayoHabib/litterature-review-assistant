# tests/unit/test_citation_manager.py

import unittest
from src.output_processing.citation_manager import format_citation, generate_bibliography

class TestCitationManager(unittest.TestCase):

    def test_format_citation(self):
        citation_data = {
            "authors": "Smith, J. and Doe, A.",
            "year": "2020",
            "title": "Machine Learning Applications",
            "journal": "Journal of AI",
            "volume": "5",
            "issue": "2",
            "pages": "123-145"
        }
        expected_output = "Smith, J. and Doe, A. (2020). Machine Learning Applications. Journal of AI, 5(2), 123-145."
        self.assertEqual(format_citation(citation_data), expected_output)

    def test_format_citation_missing_data(self):
        citation_data = {
            "authors": "Smith, J.",
            "year": "2020",
            "title": "AI Basics"
        }
        expected_output = "Smith, J. (2020). AI Basics."
        self.assertEqual(format_citation(citation_data), expected_output)

    def test_generate_bibliography(self):
        citations = [
            {
                "authors": "Smith, J.",
                "year": "2020",
                "title": "AI Basics",
                "journal": "Tech Review"
            },
            {
                "authors": "Doe, A.",
                "year": "2021",
                "title": "Machine Learning Advances",
                "journal": "AI Journal"
            }
        ]
        expected_output = "1. Smith, J. (2020). AI Basics. Tech Review.\n2. Doe, A. (2021). Machine Learning Advances. AI Journal."
        self.assertEqual(generate_bibliography(citations), expected_output)

    def test_generate_bibliography_empty_list(self):
        self.assertEqual(generate_bibliography([]), "")

if __name__ == '__main__':
    unittest.main()