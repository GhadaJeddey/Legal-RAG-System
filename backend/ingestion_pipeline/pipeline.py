"""
Ingestion pipeline entrypoint.
Input  : data/raw/<nom_du_pdf>.pdf
Output : data/output/markdown_pages.json (une entree par page)
         data/output/document.md (concatenation, pour relecture rapide)

Usage:
    python pipeline.py <chemin_vers_pdf>
    python pipeline.py <chemin_vers_pdf> --start-page 10 --end-page 50
"""
import json
import sys
import argparse
from pathlib import Path

from parsers.markdown_parser import MarkdownParser

# Active parser for this pipeline run. Swap/select via config as new parsing strategies are added (
PARSER = MarkdownParser()


def main():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("pdf_path")
    parser.add_argument("--start-page", type=int, default=None,
                         help="Page de depart (1-indexee). Defaut: debut du document.")
    parser.add_argument("--end-page", type=int, default=None,
                         help="Page de fin (incluse). Defaut: fin du document.")
    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        print(f"Fichier introuvable: {args.pdf_path}")
        sys.exit(1)

    if args.start_page is not None and args.start_page < 1:
        print("--start-page doit etre >= 1")
        sys.exit(1)
    if (args.start_page is not None and args.end_page is not None
            and args.start_page > args.end_page):
        print("--start-page ne peut pas etre superieur a --end-page")
        sys.exit(1)

    pages = PARSER.extract(args.pdf_path, start_page=args.start_page, end_page=args.end_page)
    print(f"{len(pages)} pages extraites.")

    if not pages:
        print("Aucune page extraite.")
        sys.exit(0)

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "markdown_pages.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Sauvegarde: {json_path}")

    md_path = output_dir / "document.md"
    with open(md_path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"\n\n<!-- page {p['page']} -->\n\n")
            f.write(p["markdown"])
    print(f"Sauvegarde: {md_path}")

if __name__ == "__main__":
    main()
