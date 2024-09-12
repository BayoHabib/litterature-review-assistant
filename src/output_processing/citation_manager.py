# src/output_processing/citation_manager.py

from typing import Dict, List

def format_citation(citation_data: Dict[str, str]) -> str:
    """
    Format a citation based on the provided data.
    """
    # Handle authors
    authors = citation_data.get('authors', 'Anonymous')
    if authors == 'Anonymous':
        authors = 'Anonymous.'
    
    # Handle year
    year = citation_data.get('year', 'n.d.')
    
    # Handle title
    title = citation_data.get('title', '')
    
    # Start building the citation
    citation = f"{authors} ({year}). {title}."
    
    # Handle journal information
    if 'journal' in citation_data:
        journal = citation_data['journal']
        volume = citation_data.get('volume', '')
        issue = citation_data.get('issue', '')
        pages = citation_data.get('pages', '')
        
        citation += f" {journal}"
        if volume:
            citation += f", {volume}"
            if issue:
                citation += f"({issue})"
        if pages:
            citation += f", {pages}"
        citation += "."
    
    # Handle book information
    elif 'publisher' in citation_data:
        location = citation_data.get('location', '')
        publisher = citation_data['publisher']
        if location:
            citation += f" {location}: {publisher}."
        else:
            citation += f" {publisher}."
    
    return citation.strip()

def generate_bibliography(citations: List[Dict[str, str]]) -> str:
    """
    Generate a numbered bibliography from a list of citations.
    """
    if not citations:
        return ""
    
    # Format each citation and add numbering, maintaining original order
    formatted_citations = [f"{i+1}. {format_citation(citation)}" for i, citation in enumerate(citations)]
    
    # Join all formatted citations with newlines
    return "\n".join(formatted_citations)