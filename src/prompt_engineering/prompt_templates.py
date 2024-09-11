# src/prompt_engineering/prompt_templates.py

from string import Template

class PromptTemplates:
    SUMMARIZE_PAPER = Template("""
Summarize the following academic paper in a concise manner. Focus on the main research question, methodology, key findings, and conclusions. The summary should be about 3-4 sentences long.

Title: $title
Authors: $authors
Abstract: $abstract

Summary:
""")

    EXTRACT_KEY_POINTS = Template("""
Extract the 3-5 most important key points from the following academic paper abstract. Each key point should be a single sentence.

Title: $title
Abstract: $abstract

Key Points:
1.
""")

    COMPARE_PAPERS = Template("""
Compare and contrast the following two academic papers. Focus on their research questions, methodologies, findings, and conclusions. Highlight any significant agreements or disagreements between the papers.

Paper 1:
Title: $title1
Authors: $authors1
Abstract: $abstract1

Paper 2:
Title: $title2
Authors: $authors2
Abstract: $abstract2

Comparison:
""")

    GENERATE_RESEARCH_QUESTIONS = Template("""
Based on the following abstract, generate 3 potential research questions for future studies that build upon or address gaps in this research.

Title: $title
Abstract: $abstract

Potential Research Questions:
1.
""")

    EVALUATE_METHODOLOGY = Template("""
Evaluate the research methodology described in the following abstract. Consider the appropriateness of the method for the research question, potential limitations, and suggestions for improvement.

Title: $title
Abstract: $abstract

Methodology Evaluation:
""")

    LITERATURE_REVIEW_OUTLINE = Template("""
Create an outline for a literature review on the topic of "$topic". The outline should include main sections and subsections, covering key themes, methodologies, findings, and gaps in the current research.

Literature Review Outline:
""")

    @classmethod
    def format_prompt(cls, template_name: str, **kwargs) -> str:
        """
        Format a prompt template with the given parameters.
        
        :param template_name: Name of the template to use (e.g., 'SUMMARIZE_PAPER')
        :param kwargs: Keyword arguments to fill in the template
        :return: Formatted prompt string
        """
        template = getattr(cls, template_name, None)
        if template is None:
            raise ValueError(f"Template '{template_name}' not found")
        return template.safe_substitute(**kwargs)

# Example usage
if __name__ == "__main__":
    paper_info = {
        "title": "Advanced Machine Learning Techniques",
        "authors": "John Doe, Jane Smith",
        "abstract": "This paper explores cutting-edge machine learning algorithms..."
    }
    
    summary_prompt = PromptTemplates.format_prompt('SUMMARIZE_PAPER', **paper_info)
    print(summary_prompt)

    key_points_prompt = PromptTemplates.format_prompt('EXTRACT_KEY_POINTS', **paper_info)
    print(key_points_prompt)