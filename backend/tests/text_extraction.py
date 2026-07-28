import pdfplumber 
import json
import re 

def extract_with_pdfplumber(pdf_path, start, end):
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start, min(end, len(pdf.pages))):
            page = pdf.pages[i]
            text = page.extract_text() or ""

            # Table extraction (default settings — tune later if needed)
            tables = page.extract_tables()

            # Also try explicit strategies to compare
            tables_lines = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                }
            )

            results.append({
                "page": i + 1,
                "text": text,
                "tables_default": tables,
                "tables_lines_strategy": tables_lines,
                "num_tables_default": len(tables),
            })
            
    return results


if __name__ == "__main__":
    pdf_path = "PCG--1er-janvier-2025.pdf"
    start_page = 79
    end_page = 85
    extracted_data = extract_with_pdfplumber(pdf_path, start_page, end_page)
    
    # Save the extracted data to a JSON file

    with open("outputs\extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)