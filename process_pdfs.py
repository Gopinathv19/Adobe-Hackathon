from pathlib import Path
import json
from extractor import main
from inference import start, title
import shutil
import os

def process_pdfs():
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    intermediate_files = []
    for pdf_file in input_dir.glob("*.pdf"):
        # Prepare per-PDF filenames
        title_csv = f"{pdf_file.stem}_title.csv"
        output_csv = f"{pdf_file.stem}_output.csv"
        inference_json = f"{pdf_file.stem}_inference_output.json"
        intermediate_files.extend([title_csv, output_csv, inference_json])

        # --- Extract features for this PDF only ---
        from extractor import extract_features_from_pdf
        import pandas as pd
        # Title (page 0)
        title_data = extract_features_from_pdf(str(pdf_file), start_page=0, end_page=1)
        if title_data:
            pd.DataFrame(title_data).to_csv(title_csv, index=False)
        # Outline (pages 1+)
        total_pages = 1
        try:
            import fitz
            total_pages = fitz.open(str(pdf_file)).page_count
        except Exception:
            pass
        if total_pages > 1:
            outline_data = extract_features_from_pdf(str(pdf_file), start_page=1, end_page=total_pages)
            if outline_data:
                pd.DataFrame(outline_data).to_csv(output_csv, index=False)
        else:
            outline_data = []
            pd.DataFrame([]).to_csv(output_csv, index=False)

        # --- Inference for this PDF only ---
        from inference import start, title
        # Title extraction
        title0 = title(title_csv)
        # Outline inference
        start(output_csv)
        # Rename inference_output.json to per-PDF name
        if os.path.exists("inference_output.json"):
            shutil.move("inference_output.json", inference_json)
        # Read the outline from per-PDF inference output
        with open(inference_json, "r", encoding="utf-8") as f:
            outline = json.load(f)

        # Ensure each outline entry has the required keys
        filtered_outline = []
        for entry in outline:
            if all(k in entry for k in ("level", "text", "page")):
                filtered_outline.append({
                    "level": str(entry["level"]),
                    "text": str(entry["text"]),
                    "page": int(entry["page"])
                })

        filtered_outline = [
            {
                "level": o["level"],
                "text": o["text"],
                "page": o["page"]
            }
            for o in filtered_outline
        ]

        # Extract title
        if title0 and isinstance(title0, list):
            if isinstance(title0[0], dict) and 'text' in title0[0]:
                doc_title = title0[0]['text']
            else:
                doc_title = title0[0] if len(title0) == 1 else ' '.join(title0)
        else:
            doc_title = ""

        output = {
            "title": doc_title,
            "outline": filtered_outline
        }

        # Save output as <pdfname>.json
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # --- Cleanup all intermediate files ---
    for file in intermediate_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception:
            pass

        # Generate schema for this PDF (as in main.py)
        outline_schema = {
            "type": "object",
            "properties": {
                "level": { "type": "string" },
                "text": { "type": "string" },
                "page": { "type": "integer" }
            },
            "required": ["level", "text", "page"]
        }
        repeated_outline_schema = [outline_schema for _ in filtered_outline]
        schema = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "properties": {
                "title": { "type": "string" },
                "outline": {
                    "type": "array",
                    "items": repeated_outline_schema
                }
            },
            "required": ["title", "outline"]
        }
        schema_file = f"schema\\{pdf_file.stem}.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_pdfs()
