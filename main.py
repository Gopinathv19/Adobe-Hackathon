from extractor import main
from inference import start, title
import json

main()
title0 = title("title.csv")
start("output.csv")

# Read the outline from inference_output.json
with open("inference_output.json", "r", encoding="utf-8") as f:
    outline = json.load(f)

# Ensure each outline entry has the required keys
filtered_outline = []
for entry in outline:
    # Only keep the required keys and ensure correct types
    if all(k in entry for k in ("level", "text", "page")):
        filtered_outline.append({
            "level": str(entry["level"]),
            "text": str(entry["text"]),
            "page": int(entry["page"])
        })

# Remove any extra keys from each outline entry to strictly match the schema
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

with open(r"output\output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Generate schema_output.json that repeats the schema for each chunk in outline (for educational purposes)
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
with open(r"schema\schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, ensure_ascii=False, indent=2)