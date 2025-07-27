import pdfplumber
import pandas as pd
from collections import defaultdict, Counter
import re
import json
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get the contents of the page
def normalized_to_rgb(normalized_tuple):
    """
    Converts a normalized RGB tuple (0.0–1.0) to a standard 0–255 RGB tuple.

    Example:
        Input: (0.0588, 0.278, 0.38)
        Output: (15, 71, 97)
    """
    if not isinstance(normalized_tuple, (tuple, list)) or len(normalized_tuple) != 3:
        raise ValueError("Input must be a tuple or list of three floats (R, G, B)")

    return tuple(min(255, max(0, int(round(c * 255)))) for c in normalized_tuple)

def get_page_content(file_path):
    with pdfplumber.open(file_path) as pdf:
        page_content = []

        for page_num, page in enumerate(pdf.pages):
            line_groups = defaultdict(list)

            # Group characters by rounded 'top'
            
            for char in page.chars:
                rounded_top = round(char['top'], 1)
                line_groups[rounded_top].append(char)

            lines_on_page = []
            
            # Sort top values (ascending = top of page to bottom)
            for top in sorted(line_groups.keys()):
                chars = sorted(line_groups[top], key=lambda c: c['x0'])  # left to right
                line_text = ''.join(c['text'] for c in chars).strip()

                # Get the most common font size from the characters in the line
                sizes = [round(c['size'], 2) for c in chars]
                most_common_size = Counter(sizes).most_common(1)[0][0] if sizes else None

                # Get the most common colour of the charachters
                stroking_color = [c['stroking_color'] for c in chars]
                most_common_stroking_color = Counter(stroking_color).most_common(1)[0][0] if sizes else None
                if not isinstance(most_common_stroking_color, (tuple, list)) or len(most_common_stroking_color) != 3:
                    most_common_stroking_color = (0, 0, 0)
                else:
                    most_common_stroking_color = normalized_to_rgb(most_common_stroking_color)

                # leftmost position of the line
                leftmost_x0 = min(c['x0'] for c in chars) if chars else None
                
                

                if line_text:
                    lines_on_page.append({
                        'text': line_text,
                        'font_size': most_common_size,
                        'top': top,
                        'stroking_color':most_common_stroking_color,
                        'x0': leftmost_x0,  
                    })

            page_content.append(lines_on_page)
    return page_content


# Detect Headers and Footers
def detect_repeated_header_dicts(page_content, num_lines=7, min_fraction=0.8, top_tolerance=0.5):
    """
    Detects repeated headers in the first `num_lines` lines of each page.
    Returns a list of unique line dictionaries to remove, matching both text and position (top).
    """
    if len(page_content) < 2:
        return []
    
    # Map (text, rounded_top) to a representative dict
    key_to_dict = {}
    header_keys = []

    for page in page_content:
        lines = page[:num_lines]
        for line in lines:
            rounded_top = round(line['top'] / top_tolerance) * top_tolerance
            key = (line['text'], rounded_top)
            if key not in key_to_dict:
                key_to_dict[key] = line  # Store the first occurrence as representative
            header_keys.append(key)

    # Count how often each (text, top) pair appears
    key_counts = Counter(header_keys)
    num_pages = len(page_content)
    # A line is a header if it appears on at least `min_fraction` of pages
    header_candidates = {key for key, count in key_counts.items() if count >= min_fraction * num_pages}

    # Collect one representative dict for each header candidate
    header_dicts = [key_to_dict[key] for key in header_candidates]
    return header_dicts


# for d in header_dicts:
#     print(d)



def detect_footer_by_position_and_font(page_content, min_fraction=0.8, top_tolerance=2.0, font_tolerance=0.2):
    """
    Detects repeated footers by clustering lines with similar 'top' and 'font_size' across pages.
    Returns a list of (top, font_size) pairs that are likely footers.
    """
    footer_candidates = defaultdict(list)
    num_pages = len(page_content)
    
    for page in page_content:
        if not page:
            continue
        # Consider last 2 lines as possible footers
        for line in page[-2:]:
            rounded_top = round(line['top'] / top_tolerance) * top_tolerance
            rounded_font = round(line['font_size'] / font_tolerance) * font_tolerance
            key = (rounded_top, rounded_font)
            footer_candidates[key].append(line)
    
    # Find (top, font_size) pairs that appear on enough pages
    footer_keys = [key for key, lines in footer_candidates.items() if len(lines) >= min_fraction * num_pages]
    
    # Collect all line dicts matching those keys
    footers = []
    for page in page_content:
        for line in page[-2:]:
            rounded_top = round(line['top'] / top_tolerance) * top_tolerance
            rounded_font = round(line['font_size'] / font_tolerance) * font_tolerance
            if (rounded_top, rounded_font) in footer_keys:
                footers.append(line)
    return footers


# for f in footer_dicts:
#     print(f)

## Remove detected Headers and Footer
def remove_headers_and_footers(page_content, header_dicts, footer_dicts, fields=('text', 'font_size')):
    """
    Removes lines from page_content that match any header/footer dict on the specified fields.
    """
    def dict_key(d):
        return tuple(d.get(f) for f in fields)

    header_keys = {dict_key(d) for d in header_dicts}
    footer_keys = {dict_key(d) for d in footer_dicts}

    cleaned_page_content = []
    for page in page_content:
        cleaned_page = [
            line for line in page
            if dict_key(line) not in header_keys and dict_key(line) not in footer_keys
        ]
        cleaned_page_content.append(cleaned_page)
    return cleaned_page_content



## Add line_spacing_below variable
def annotate_line_spacing(page_content):
    """
    Adds a 'line_spacing_below' key to each line dict, representing the vertical space to the next line.
    The last line on each page gets None for this key.
    """
    for page in page_content:
        # Sort lines by 'top' (ascending: top of page to bottom)
        sorted_lines = sorted(page, key=lambda l: l['top'])
        for i, line in enumerate(sorted_lines):
            if i < len(sorted_lines) - 1:
                spacing = round(sorted_lines[i+1]['top'] - line['top'], 0)
            else:
                spacing = -1
            line['line_spacing_below'] = spacing
    return page_content


## Filter the headings based only on font size
def is_valid_heading(text):
    text = text.strip()
    # Check if it contains at least one alphanumeric or heading-like character
    if not re.search(r'[A-Za-z0-9:\-\(\)]', text):
        return False
    
    # Exclude links (URLs, web addresses, etc.)
    link_patterns = [
        r'https?://',           # http:// or https://
        r'www\.',              # www.
        r'\.com',              # .com
        r'\.org',              # .org
        r'\.net',              # .net
        r'\.edu',              # .edu
        r'\.gov',              # .gov
        r'\[.*\]\(.*\)',       # Markdown links [text](url)
        r'<.*>',               # HTML-like tags
        r'ftp://',             # ftp://
        r'file://',            # file://
    ]
    
    for pattern in link_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True

def uncommon_font_size_texts_per_page(page_content):
    """
    For each page, returns a list of line dicts whose font size is NOT the dominant one,
    and whose text passes heading validity checks.
    Adds 'page_index' to each line for reference.
    
    Special logic for first page:
        - If dominant font size is also the largest, return those lines.
        - Otherwise, return lines with the largest size.
    Returns:
        List of lists: Each sublist contains the line dicts for one page.
    """
    result = []

    for page_idx, page in enumerate(page_content):
        font_sizes = [line['font_size'] for line in page]
        if not font_sizes or len(set(font_sizes)) == 1:
            result.append([])
            continue

        font_size_counts = Counter(font_sizes)
        most_common_size, count = font_size_counts.most_common(1)[0]
        
        max_count = max(font_size_counts.values())
        tied_sizes = [size for size, count in font_size_counts.items() if count == max_count]
        dominant_size = max(tied_sizes)
        largest_size = max(font_sizes)

        if page_idx == 0:
            # First page logic
            if dominant_size == largest_size:
                lines_to_add = [
                    line | {"page_index": page_idx}
                    for line in page
                    if line['font_size'] == dominant_size and is_valid_heading(line['text'])
                ]
            else:
                lines_to_add = [
                    line | {"page_index": page_idx}
                    for line in page
                    if line['font_size'] == largest_size or line['font_size'] != dominant_size and is_valid_heading(line['text'] )
                ]
        else:
            # Other pages logic
            lines_to_add = [
                line | {"page_index": page_idx}
                for line in page
                if line['font_size'] != dominant_size
                and line['font_size'] > most_common_size
                and is_valid_heading(line['text'])
            ]

        result.append(lines_to_add)

    return result



## Detect Heading based on spacing

def detect_headings_by_char_count_and_spacing(page_content, char_count_threshold=50):
    """
    Detect headings based on:
    1. Character count less than threshold
    2. Line doesn't end with full stop or comma
    3. Line spacing below is greater than most common spacing on that page
    4. Line alignment matches majority of headings (using x0 position)
    
    Args:
        page_content: List of pages, each containing list of line dicts
        char_count_threshold: Maximum character count for a line to be considered heading
    
    Returns:
        List of lists: Each sublist contains heading line dicts for one page
    """
    result = []
    
    for page_idx, page in enumerate(page_content):
        if not page:
            result.append([])
            continue
            
        # Get valid lines (exclude lines with spacing -1)
        valid_lines = [line for line in page if line.get('line_spacing_below', -1) != -1]
        
        if not valid_lines:
            result.append([])
            continue
        
        # Calculate most common spacing for this page
        spacings = [line['line_spacing_below'] for line in valid_lines]
        spacing_counts = Counter(spacings)
        most_common_spacing = spacing_counts.most_common(1)[0][0] if spacing_counts else 0
        
        # print(f"Page {page_idx + 1}: Most common spacing = {most_common_spacing}")
        
        # First pass: collect potential headings to determine alignment pattern
        potential_headings = []
        for line in valid_lines:
            text = line['text'].strip()
            char_count = len(text)
            spacing_below = line.get('line_spacing_below', -1)
            
            # Check conditions for heading detection (without alignment check)
            is_short_line = char_count < char_count_threshold
            doesnt_end_with_punctuation = not (text.endswith('.') or text.endswith(',') or text.endswith(':'))
            has_large_spacing = spacing_below > most_common_spacing
            has_meaningful_content = is_valid_heading(text)
            
            if (is_short_line and 
                doesnt_end_with_punctuation and 
                has_large_spacing and 
                has_meaningful_content):
                potential_headings.append(line)
        
        # Determine alignment pattern from potential headings
        if potential_headings:
            # Get x0 positions (left alignment) of potential headings
            x0_positions = [line.get('x0', 0) for line in potential_headings]
            x0_counts = Counter(x0_positions)
            most_common_x0 = x0_counts.most_common(1)[0][0] if x0_counts else 0
            
            # Allow some tolerance for alignment (within 5 points)
            alignment_tolerance = 5.0
            
            # print(f"  Most common heading alignment (x0): {most_common_x0}")
            # print(f"  Alignment tolerance: ±{alignment_tolerance}")
        else:
            most_common_x0 = 0
            alignment_tolerance = 5.0
        
        headings_on_page = []
        
        for line in valid_lines:
            text = line['text'].strip()
            char_count = len(text)
            spacing_below = line.get('line_spacing_below', -1)
            x0_position = line.get('x0', 0)
            
            # Check conditions for heading detection
            is_short_line = char_count < char_count_threshold
            doesnt_end_with_punctuation = not (text.endswith('.') or text.endswith(','))
            has_large_spacing = spacing_below > most_common_spacing
            has_meaningful_content = is_valid_heading(text)
            
            # Check alignment consistency
            has_consistent_alignment = abs(x0_position - most_common_x0) <= alignment_tolerance
            
            if (is_short_line and 
                doesnt_end_with_punctuation and 
                has_large_spacing and 
                has_meaningful_content and
                has_consistent_alignment):
                
                heading_info = line.copy()
                heading_info.update({
                    'page_index': page_idx,
                #     'char_count': char_count,
                #     'spacing_below': spacing_below,
                #     'most_common_spacing': most_common_spacing,
                #     'x0_position': x0_position,
                #     'most_common_x0': most_common_x0,
                #     'heading_score': spacing_below / most_common_spacing if most_common_spacing > 0 else 1.0
                })
                headings_on_page.append(heading_info)
                
                # print(f"  Heading detected: '{text}' (chars: {char_count}, spacing: {spacing_below}, x0: {x0_position})")
        
        result.append(headings_on_page)
    
    return result


## Merge headings detected from font and spacing
def line_key(line):
    """Create a hashable key for a line dict based on its main identifying fields."""
    return (
        line.get('text', '').strip(),
        round(float(line.get('font_size', 0)), 2),  # Round to 2 decimal places
        round(float(line.get('top', 0)), 1),        # Round to 1 decimal place
        tuple(line.get('stroking_color', ())),
        round(float(line.get('x0', 0)), 1),         # Round to 1 decimal place
        int(line.get('page_index', -1))
    )

def merge_line_pages_no_duplicates(l1, l2):
    """
    Merge two lists of lists (per page), preserving the nested structure,
    and ensuring each line dict appears only once per page (by main fields).
    """
    max_len = max(len(l1), len(l2))
    merged = [[] for _ in range(max_len)]

    for i in range(max_len):
        page1 = l1[i] if i < len(l1) else []
        page2 = l2[i] if i < len(l2) else []

        seen = set()
        unique_lines = []
        
        # Process all lines from both pages
        all_lines = page1 + page2
        
        for line in all_lines:
            key = line_key(line)
            if key not in seen:
                seen.add(key)
                unique_lines.append(line)
        
        merged[i] = unique_lines

    return merged


## Vote if title or heading based only on font
def add_voting_and_deduplicate(uncommon_lines_per_page):
    # 1. Collect all unique font sizes
    all_font_sizes = set()
    for page in uncommon_lines_per_page:
        for line in page:
            all_font_sizes.add(line['font_size'])
    # 2. Sort font sizes descending
    sorted_sizes = sorted(all_font_sizes, reverse=True)
    print(sorted_sizes)
    # 3. Map font sizes to voting labels
    voting_labels = ['title', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']
    size_to_label = {size: voting_labels[i] for i, size in enumerate(sorted_sizes) if i < len(voting_labels)}

    # 4. For each page, deduplicate by (font_size, text) and add voting
    result = []
    for page in uncommon_lines_per_page:
        seen = set()
        deduped_page = []
        for line in page:
            key = (line['font_size'], line['text'])
            if key not in seen:
                seen.add(key)
                label = size_to_label.get(line['font_size'], None)
                line = line.copy()  # Avoid mutating original
                if label:
                    line['voting'] = {label: 1}
                else:
                    line['voting'] = {}
                deduped_page.append(line)
        result.append(deduped_page)

    return result


# To see the result:
# for page in heading:
#     for line in page:
#         print(line)


def add_sub_content_to_headings(headings, page_content):
    # Flatten headings into a list with (page_index, top, heading_dict)
    flat_headings = []
    for page in headings:
        for h in page:
            flat_headings.append((h['page_index'], h['top'], h))
    # Sort by page_index, then top
    flat_headings.sort()
    
    # For each heading, find the next heading (could be on same or later page)
    for idx, (page_idx, top, h) in enumerate(flat_headings):
        # Find next heading
        if idx + 1 < len(flat_headings):
            next_page_idx, next_top, _ = flat_headings[idx + 1]
        else:
            next_page_idx, next_top = float('inf'), float('inf')
        
        sub_lines = []
        cur_page = page_idx
        cur_top = top
        while True:
            lines = page_content[cur_page]
            # Get lines after cur_top (if first page), or all lines (if next page)
            if cur_page == page_idx:
                candidate_lines = [line for line in lines if line['top'] > cur_top]
            else:
                candidate_lines = lines
            # If this is the page of the next heading, only take lines before next_top
            if cur_page == next_page_idx:
                candidate_lines = [line for line in candidate_lines if line['top'] < next_top]
            # Add their text
            sub_lines.extend(line['text'] for line in candidate_lines)
            # Stop if we've reached the page of the next heading
            if cur_page == next_page_idx or cur_page >= len(page_content) - 1:
                break
            cur_page += 1
            cur_top = -float('inf')  # On next page, take all lines from top
        h['sub_content'] = "\n".join(sub_lines).strip()
    return headings


# Convert to JSON outline
def headings_to_json_outline(headings):
    """
    Convert a list of lists of heading dicts to the required JSON outline format.
    - The title is the first line that has 'title' in its voting, combined with the next if it's also a title.
    - The outline contains only lines with voting keys that are not 'title'.
    """
    # Find the first line with 'title' in its voting for the title
    title = ""
    title_lines_to_skip = set()  # Track which title lines to skip in outline
    
    for page in headings:
        for i, h in enumerate(page):
            level = next(iter(h['voting'].keys()))
            if level.lower() == "title":
                title = h['text']
                title_lines_to_skip.add((h['page_index'], h['text']))  # Mark first title to skip
                
                # Check if the next heading is also a title
                if i + 1 < len(page):
                    next_h = page[i + 1]
                    next_level = next(iter(next_h['voting'].keys()))
                    if next_level.lower() == "title":
                        title += "  " + next_h['text']
                        title_lines_to_skip.add((next_h['page_index'], next_h['text']))  # Mark second title to skip
                title += "  "
                break
        if title:
            break

    outline = []
    for page in headings:
        for h in page:
            # Skip if this line was used in the title
            if (h['page_index'], h['text']) in title_lines_to_skip:
                continue
            outline.append({
                "level": next(iter(h['voting'].keys())),
                "text": h['text'] + " ",
                "page": h['page_index'],
                "top": h['top'],
                "sub_content": h['sub_content']
            })

    return {
        "title": title,
        "outline": outline
    }




def process_pdf_files():
    """Process all PDF files in the input directory."""
    input_dir = Path("./app/input")
    output_dir = Path("./app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    total_start_time = time.time()
    
    for pdf_file in pdf_files:
        file_start_time = time.time()
        try:
            logger.info(f"Processing {pdf_file.name}")
            
            # Extract page content
            page_content = get_page_content(str(pdf_file))
            
            # Detect and remove headers and footers
            header_dicts = detect_repeated_header_dicts(page_content)
            footer_dicts = detect_footer_by_position_and_font(page_content)
            page_content = remove_headers_and_footers(page_content, header_dicts, footer_dicts)
            
            # Annotate line spacing
            page_content = annotate_line_spacing(page_content)
            
            # Detect headings by font size
            uncommon_lines_per_page = uncommon_font_size_texts_per_page(page_content)
            
            # Detect headings by spacing
            spacing_headings = detect_headings_by_char_count_and_spacing(page_content, char_count_threshold=50)
            
            # Merge and deduplicate headings
            final_headings = merge_line_pages_no_duplicates(uncommon_lines_per_page, spacing_headings)
            heading = add_voting_and_deduplicate(final_headings)
            headings_with_subcontent = add_sub_content_to_headings(heading, page_content)
            # Convert to JSON outline
            result = headings_to_json_outline(headings_with_subcontent)
            
            # Save output
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            file_processing_time = time.time() - file_start_time
            logger.info(f"Successfully processed {pdf_file.name} in {file_processing_time:.2f} seconds")
            
        except Exception as e:
            file_processing_time = time.time() - file_start_time
            logger.error(f"Failed to process {pdf_file.name} after {file_processing_time:.2f} seconds: {str(e)}")
            # Create error output
            error_output = {
                "title": "Processing Error",
                "outline": []
            }
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_output, f, indent=2)
    
    total_processing_time = time.time() - total_start_time
    logger.info(f"Completed processing {len(pdf_files)} PDF files in {total_processing_time:.2f} seconds")
    logger.info(f"Average time per PDF: {total_processing_time/len(pdf_files):.2f} seconds")



if __name__ == "__main__":
    process_pdf_files()