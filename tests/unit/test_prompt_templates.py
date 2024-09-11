# tests/unit/test_prompt_templates.py

import unittest
from src.prompt_engineering.prompt_templates import PromptTemplates

class TestPromptTemplates(unittest.TestCase):
    def setUp(self):
        self.paper_info = {
            "title": "Test Paper",
            "authors": "John Doe, Jane Smith",
            "abstract": "This is a test abstract for prompt engineering."
        }

    def test_summarize_paper_prompt(self):
        prompt = PromptTemplates.format_prompt('SUMMARIZE_PAPER', **self.paper_info)
        self.assertIn(self.paper_info['title'], prompt)
        self.assertIn(self.paper_info['authors'], prompt)
        self.assertIn(self.paper_info['abstract'], prompt)
        self.assertIn("Summarize the following academic paper", prompt)

    def test_extract_key_points_prompt(self):
        prompt = PromptTemplates.format_prompt('EXTRACT_KEY_POINTS', **self.paper_info)
        self.assertIn(self.paper_info['title'], prompt)
        self.assertIn(self.paper_info['abstract'], prompt)
        self.assertIn("Extract the 3-5 most important key points", prompt)

    def test_compare_papers_prompt(self):
        compare_info = {
            "title1": "Paper 1",
            "authors1": "Author 1",
            "abstract1": "Abstract 1",
            "title2": "Paper 2",
            "authors2": "Author 2",
            "abstract2": "Abstract 2"
        }
        prompt = PromptTemplates.format_prompt('COMPARE_PAPERS', **compare_info)
        for key, value in compare_info.items():
            self.assertIn(value, prompt)
        self.assertIn("Compare and contrast the following two academic papers", prompt)

    def test_generate_research_questions_prompt(self):
        prompt = PromptTemplates.format_prompt('GENERATE_RESEARCH_QUESTIONS', **self.paper_info)
        self.assertIn(self.paper_info['title'], prompt)
        self.assertIn(self.paper_info['abstract'], prompt)
        self.assertIn("generate 3 potential research questions", prompt)

    def test_evaluate_methodology_prompt(self):
        prompt = PromptTemplates.format_prompt('EVALUATE_METHODOLOGY', **self.paper_info)
        self.assertIn(self.paper_info['title'], prompt)
        self.assertIn(self.paper_info['abstract'], prompt)
        self.assertIn("Evaluate the research methodology", prompt)

    def test_literature_review_outline_prompt(self):
        topic = "Artificial Intelligence in Healthcare"
        prompt = PromptTemplates.format_prompt('LITERATURE_REVIEW_OUTLINE', topic=topic)
        self.assertIn(topic, prompt)
        self.assertIn("Create an outline for a literature review", prompt)

    def test_invalid_template_name(self):
        with self.assertRaises(ValueError):
            PromptTemplates.format_prompt('NONEXISTENT_TEMPLATE', **self.paper_info)

if __name__ == '__main__':
    unittest.main()