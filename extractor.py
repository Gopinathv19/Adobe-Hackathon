import numpy as np
import fitz
import time
from pathlib import Path
import pandas as pd

def merge_spans(spans, page_width, page_height, toc_entries):
    """Merge spans into complete text lines, handling column layouts, and exclude TOC entries using get_toc()."""
    if not spans:
        return []

    # Group spans by y-bands dynamically based on font size
    y_bands = {}
    for span in spans:
        bbox = span["bbox"]
        font_size = int(span["size"])
        norm_y0 = bbox[1] / page_height
        y_threshold = 1.2 * (font_size / page_height)
        y_key = round(norm_y0 / y_threshold) * y_threshold
        if y_key not in y_bands:
            y_bands[y_key] = []
        y_bands[y_key].append(span)

    merged_lines = []
    in_toc_section = False  # Flag to track TOC context
    toc_titles = {entry[1] for entry in toc_entries}  # Set of TOC titles for quick lookup

    for y_key in sorted(y_bands.keys()):
        band_spans = y_bands[y_key]
        band_spans.sort(key=lambda s: s["bbox"][0])

        current_line = []
        prev_x0 = None
        col_break_threshold = 0.2 * page_width
        for span in band_spans:
            bbox = span["bbox"]
            norm_x0 = bbox[0] / page_width
            text = span["text"].strip()
            font_size = int(span["size"])
            font = span["font"]
            is_bold = 1 if any(indicator in font.lower() for indicator in ["bold", "black", "heavy", "demi", "extra", "thick", "strong", "ultra"]) or span["flags"] & 2 else 0
            page_num = span["bbox"][0] // page_width  # Approximate page association

            # Check for TOC heading to set the flag
            if text.lower() == "table of contents" and not in_toc_section:
                in_toc_section = True
                current_line.append({"text": text, "bbox": bbox, "font_size": font_size, "is_bold": is_bold})
                continue

            # Skip TOC entries if in TOC section and match get_toc() titles
            if in_toc_section:
                # Check if this line matches a TOC entry title
                line_text = " ".join(s["text"] for s in current_line + [{"text": text}]).strip()
                # End TOC section if a significant y-gap or new major heading
                if len(current_line) > 0 and abs(norm_y0 - (current_line[-1]["bbox"][1] / page_height)) > 0.05:
                    in_toc_section = False

            if prev_x0 is None:
                prev_x0 = norm_x0
                current_line.append({"text": text, "bbox": bbox, "font_size": font_size, "is_bold": is_bold})
            else:
                if norm_x0 - prev_x0 > col_break_threshold / page_width:
                    if current_line:
                        line_text = " ".join(s["text"] for s in current_line).strip()
                        #print(f"Extracted Line: '{line_text}' (y_key: {y_key}, page_x0: {prev_x0})")
                        merged_lines.append(current_line)
                    current_line = [{"text": text, "bbox": bbox, "font_size": font_size, "is_bold": is_bold}]
                    prev_x0 = norm_x0
                else:
                    current_line.append({"text": text, "bbox": bbox, "font_size": font_size, "is_bold": is_bold})

        if current_line:
            line_text = " ".join(s["text"] for s in current_line).strip()
            #print(f"Extracted Line: '{line_text}' (y_key: {y_key}, page_x0: {prev_x0})")
            merged_lines.append(current_line)

    result = []
    for line in merged_lines:
        line_text = " ".join(s["text"] for s in line).strip()
        line_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]
        max_font_size = 0
        is_bold_line = 0
        for span in line:
            bbox = span["bbox"]
            line_bbox[0] = min(line_bbox[0], bbox[0])
            line_bbox[1] = min(line_bbox[1], bbox[1])
            line_bbox[2] = max(line_bbox[2], bbox[2])
            line_bbox[3] = max(line_bbox[3], bbox[3])
            max_font_size = max(max_font_size, span["font_size"])
            is_bold_line = max(is_bold_line, span["is_bold"])
        result.append({
            "text": line_text,
            "font_size": max_font_size,
            "is_bold": is_bold_line,
            "bbox": line_bbox
        })
    return result

def extract_features_from_pdf(pdf_path, start_page=0, end_page=None):
    """Extract all text and features from a PDF file for a specific page range, excluding footers."""
    start_time = time.time()
    doc = fitz.open(pdf_path)
    data = []
    font_sizes = []
    text_blocks = []

    # Get TOC from the PDF
    toc_entries = doc.get_toc()

    BOTTOM_EXCLUSION_THRESHOLD = 0.9
    end_page = end_page if end_page is not None else len(doc)

    for page_num in range(start_page, end_page):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]

        all_spans = []
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    font_size = int(span["size"])
                    font = span["font"]
                    bbox = span["bbox"]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]

                    if width < 2 or height < 1:
                        continue

                    norm_y0 = bbox[1] / page_height
                    if norm_y0 > BOTTOM_EXCLUSION_THRESHOLD:
                        continue

                    all_spans.append({"text": text, "bbox": bbox, "size": font_size, "font": font, "flags": span["flags"]})

        merged_lines = merge_spans(all_spans, page_width, page_height, toc_entries)

        page_blocks = []
        for line in merged_lines:
            text = line["text"]
            font_size = line["font_size"]
            is_bold = line["is_bold"]
            bbox = line["bbox"]
            norm_y0 = bbox[1] / page_height
            norm_x0 = (max(bbox[0] - 0.05 * page_width, 0)) / page_width
            centeredness = round(norm_x0, 2)
            text_length = len(text)

            block_data = {
                "text": text,
                "font_size": font_size,
                "is_bold": is_bold,
                "text_length": text_length,
                "centeredness": centeredness,
                "norm_y0": norm_y0,
                "y1": bbox[3],
                "page_no": page_num  # Renamed from 'page' to 'page_no' for clarity
            }
            page_blocks.append(block_data)
            font_sizes.append(font_size)

        page_blocks.sort(key=lambda x: x["norm_y0"])
        for i, block in enumerate(page_blocks):
            spacing_before = 0
            spacing_after = 0
            if i > 0:
                spacing_before = (block["norm_y0"] * page_height - page_blocks[i - 1]["y1"]) / page_height
                spacing_before = round(max(spacing_before, 0), 2)
            if i < len(page_blocks) - 1:
                spacing_after = (page_blocks[i + 1]["norm_y0"] * page_height - block["y1"]) / page_height
                spacing_after = round(max(spacing_after, 0), 2)
            block["spacing_before"] = spacing_before
            block["spacing_after"] = spacing_after

        text_blocks.extend(page_blocks)

    font_sizes = np.array(font_sizes)
    if len(font_sizes) > 0:
        min_size = np.min(font_sizes)
        max_size = np.max(font_sizes)
        all_scores = []
        for block in text_blocks:
            base_score = (block["font_size"] - min_size) / (max_size - min_size) if max_size > min_size else 0.5
            modifiers = 0
            if block["is_bold"] == 1:
                modifiers += 0.2
            if block["centeredness"] > 0.2:
                modifiers -= 0.1
            if block["spacing_before"] > 0.1:
                modifiers += 0.15
            if block["spacing_after"] > 0.1:
                modifiers += 0.15
            if block["text_length"] >= 4 and block["text_length"] <= 70:
                modifiers += 0.1
            combined_score = min(base_score + modifiers, 1.0)
            all_scores.append(combined_score)
        
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            if max_score > min_score:
                ranks = [(score - min_score) / (max_score - min_score) for score in all_scores]
            else:
                ranks = [1.0] * len(all_scores)
            ranks = np.round(ranks, 2)
        else:
            ranks = [0.0] * len(text_blocks)
        
        for block, rank in zip(text_blocks, ranks):
            block["font_size_rank"] = rank
        
        #print(f"Font sizes in {pdf_path} (pages {start_page} to {end_page-1}): {np.unique(font_sizes)}")
        #print(f"Min score: {min(all_scores):.2f}, Max score: {max(all_scores):.2f}")
        #print(f"Sample ranks: {[r for r in ranks[:5]]}...")
    else:
        for block in text_blocks:
            block["font_size_rank"] = 0.0
        #print(f"No font sizes extracted from {pdf_path} (pages {start_page} to {end_page-1})")

    for block in text_blocks:
        is_valid_length = 1 if 4 <= block["text_length"] <= 70 else 0
        data.append({
            "text": block["text"],
            "is_bold": block["is_bold"],
            "font_size": block["font_size"],
            "font_size_rank": block["font_size_rank"],
            "centeredness": block["centeredness"],
            "spacing_before": block["spacing_before"],
            "spacing_after": block["spacing_after"],
            "is_valid_length": is_valid_length,
            "page_no": block["page_no"]  # Ensure page_no is included as the last column
        })

    doc.close()
    #print(f"Processed {pdf_path} (pages {start_page} to {end_page-1}) in {time.time() - start_time:.2f} seconds")
    return data

def create_dataset(pdf_dir, title_csv="title.csv", output_csv="output.csv"):
    """Create two datasets: one for page 0 (title.csv) and one for pages 1+ (output.csv)."""
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.is_dir():
        #print(f"Error: '{pdf_dir}' is not a directory. Please provide a directory path.")
        return
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    #print(f"Found {len(pdf_files)} PDF files in '{pdf_dir}': {[f.name for f in pdf_files]}")

    # Data for title.csv (page 0 only)
    title_data = []
    # Data for output.csv (pages 1+)
    output_data = []

    for pdf_file in pdf_files:
        # Extract features from page 0 for title.csv
        title_data.extend(extract_features_from_pdf(str(pdf_file), start_page=0, end_page=1))

        # Extract features from pages 1+ for output.csv
        total_pages = fitz.open(str(pdf_file)).page_count
        if total_pages > 1:  # Only process if there are pages beyond 0
            output_data.extend(extract_features_from_pdf(str(pdf_file), start_page=1, end_page=total_pages))

    # Save to title.csv
    if title_data:
        df_title = pd.DataFrame(title_data)
        df_title.to_csv(title_csv, index=False)
        #print(f"Title data saved to {title_csv} with {len(df_title)} entries")
    
    # Save to output.csv
    if output_data:
        df_output = pd.DataFrame(output_data)
        df_output.to_csv(output_csv, index=False)
        #print(f"Output data saved to {output_csv} with {len(df_output)} entries")
    

def main():
    pdf_dir = "input/"  # Replace with the actual directory containing your PDFs
    create_dataset(pdf_dir)