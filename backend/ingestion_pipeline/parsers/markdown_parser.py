"""
MVP1 parser - extraction texte + tableaux via pymupdf4llm, en un seul passage.
Approche standard et entierement automatisee : pas de traitement particulier
pour les tableaux complexes (rotation, cellules grisees, etc.), pymupdf4llm
fait de son mieux et produit des tableaux Markdown.
"""
import pymupdf
import pymupdf4llm

from .base import Parser


class MarkdownParser(Parser):
    def extract(self, pdf_path: str, start_page: int | None = None, end_page: int | None = None) -> list[dict]:
        total_pages = pymupdf.open(pdf_path).page_count

        first = start_page if start_page is not None else 1
        last = end_page if end_page is not None else total_pages
        pages_0_indexed = list(range(first - 1, last))

        page_chunks = pymupdf4llm.to_markdown(pdf_path, pages=pages_0_indexed, page_chunks=True)

        return [
            {
                "page": chunk["metadata"]["page_number"],
                "markdown": chunk["text"],
            }
            for chunk in page_chunks
            if chunk["text"].strip()
        ]
        
        
