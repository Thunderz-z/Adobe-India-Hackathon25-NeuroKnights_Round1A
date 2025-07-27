import fitz
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from scipy import stats
from langdetect import detect, DetectorFactory

# Initialize language detector with fixed seed for consistency
DetectorFactory.seed = 0

# Multilingual patterns and keywords
MULTI_NUMBERED = {
    'en': [r'^\d+(\.\d+)*[\.\-]?\s*', r'^Chapter\s+\d+', r'^Section\s+\d+'],
    'es': [r'^Cap[íi]tulo\s+\d+', r'^Secci[óo]n\s+\d+'],
    'fr': [r'^Chapitre\s+\d+', r'^Section\s+\d+'],
    'de': [r'^Kapitel\s+\d+', r'^Abschnitt\s+\d+'],
    'it': [r'^Capitolo\s+\d+', r'^Sezione\s+\d+'],
    'pt': [r'^Cap[ií]tulo\s+\d+', r'^Se[çc][ãa]o\s+\d+'],
    'ru': [r'^Глава\s+\d+', r'^Раздел\s+\d+'],
    'ar': [r'^[\u0621-\u064A]+\s+\d+', r'^الفصل\s+\d+'],
    'zh': [r'^第[一二三四五六七八九十\d]+章', r'^[一二三四五六七八九十\d]+\.'],
    'ja': [r'^第[一二三四五六七八九十\d]+章', r'^[一二三四五六七八九十\d]+\.'],
    'ko': [r'^[가-힣]+\s+\d+장', r'^제\d+장'],
    'hi': [r'^[\u0900-\u097F]+\s+\d+', r'^अध्याय\s+\d+'],
}

MULTI_KEYWORDS = {
    'en': ['introduction', 'summary', 'table of contents', 'references', 'acknowledgements', 
           'abstract', 'conclusion', 'overview', 'background', 'methodology', 'results', 
           'discussion', 'revision history', 'appendix', 'bibliography', 'contents', 'preface'],
    'es': ['introducción', 'resumen', 'índice', 'referencias', 'agradecimientos', 
           'resumen ejecutivo', 'conclusión', 'metodología', 'resultados'],
    'fr': ['introduction', 'résumé', 'sommaire', 'références', 'remerciements', 
           'conclusion', 'méthodologie', 'résultats', 'discussion'],
    'de': ['einführung', 'zusammenfassung', 'inhalt', 'literaturverzeichnis', 'dank', 
           'fazit', 'methodik', 'ergebnisse', 'diskussion'],
    'it': ['introduzione', 'sommario', 'indice', 'riferimenti', 'ringraziamenti', 
           'conclusione', 'metodologia', 'risultati', 'discussione'],
    'pt': ['introdução', 'resumo', 'índice', 'referências', 'agradecimentos', 
           'conclusão', 'metodologia', 'resultados', 'discussão'],
    'ru': ['введение', 'резюме', 'содержание', 'литература', 'благодарности', 
           'заключение', 'методология', 'результаты'],
    'ar': ['مقدمة', 'ملخص', 'الفهرس', 'المراجع', 'شكر', 'خلاصة', 'النتائج', 'المناقشة'],
    'zh': ['目录', '摘要', '结论', '参考文献', '致谢', '引言', '概述', '背景', '方法', '结果'],
    'ja': ['目次', '概要', '結論', '参考文献', '謝辞', 'はじめに', '緒言', '背景', '方法', '結果'],
    'ko': ['목차', '요약', '결론', '참고문헌', '감사', '서론', '배경', '방법', '결과'],
    'hi': ['परिचय', 'सारांश', 'अनुक्रमणिका', 'निष्कर्ष', 'संदर्भ', 'पृष्ठभूमि', 'विधि'],
}

# Function to clean text for heading detection
def clean_text(t):
    return t.strip().replace('\n', ' ').replace('\r', '')

# Utility to parse numbering prefix like "2.1.3" and return level depth (e.g., 3)
def numbering_prefix_level(text):
    m = re.match(r'^(\d+(\.\d+)*)(\s+|\.|-)', text.strip())
    if m:
        return m.group(1).count('.') + 1  # Count dots + 1 = depth
    return None

def extract_text_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue  # skip images/graphics
            for line in block["lines"]:
                for span in line["spans"]:
                    text = clean_text(span["text"])
                    if not text:
                        continue
                    spans.append({
                        "text": text,
                        "font_size": span["size"],
                        "bold": bool(span.get("flags", 0) & 2),
                        "italic": bool(span.get("flags", 0) & 1),
                        "page": page_num,
                        "bbox": span["bbox"],
                        "y_pos": span["bbox"][1],
                        "x_pos": span["bbox"][0],
                        "origin": (page_num, block, line),
                    })
    doc.close()
    return spans

def analyze_font_hierarchy_statistical(spans):
    """
    Use statistical analysis instead of K-means for more robust font size classification
    """
    font_sizes = [span["font_size"] for span in spans]
    font_counter = Counter(font_sizes)
    
    # Get font sizes sorted by frequency (most common first)
    common_sizes = font_counter.most_common()
    
    if len(common_sizes) <= 1:
        return {common_sizes[0][0]: 1} if common_sizes else {}
    
    # Statistical approach: use percentiles and outlier detection
    font_array = np.array(font_sizes)
    
    # Calculate percentiles
    percentiles = np.percentile(font_array, [25, 50, 75, 90, 95])
    q25, median, q75, p90, p95 = percentiles
    
    # Identify potential heading sizes using statistical thresholds
    heading_thresholds = []
    
    # Very large fonts (top 5%) - likely main headings
    if p95 > median + 2:
        heading_thresholds.append(("H1", p95, float('inf')))
    
    # Large fonts (75th-95th percentile) - likely H1/H2
    if p90 > median + 1:
        heading_thresholds.append(("H2", p90, p95 if p95 > median + 2 else float('inf')))
    
    # Medium-large fonts (50th-75th percentile) - likely H2/H3
    if q75 > median:
        heading_thresholds.append(("H3", q75, p90))
    
    # Above median but not too large - likely H3/H4
    if median < q75:
        heading_thresholds.append(("H4", median + 0.5, q75))
    
    # Create font size to level mapping
    font_level_map = {}
    
    # First, assign body text level (most common size gets level 5)
    most_common_size = common_sizes[0][0]
    default_level = 5
    
    for size in font_counter.keys():
        font_level_map[size] = default_level
    
    # Then assign heading levels based on thresholds
    level_counter = 1
    for label, min_size, max_size in sorted(heading_thresholds, key=lambda x: -x[1]):
        for size in font_counter.keys():
            if min_size <= size < max_size:
                font_level_map[size] = level_counter
        level_counter += 1
    
    return font_level_map

def detect_script_type(text):
    """Detect script type for multilingual bonus scoring"""
    if not text:
        return "unknown"
    
    first_char = ord(text[0])
    
    # CJK (Chinese, Japanese, Korean)
    if 0x4E00 <= first_char <= 0x9FFF:  # CJK Unified Ideographs
        return "cjk"
    elif 0x3040 <= first_char <= 0x309F:  # Hiragana
        return "hiragana"
    elif 0x30A0 <= first_char <= 0x30FF:  # Katakana
        return "katakana"
    elif 0xAC00 <= first_char <= 0xD7AF:  # Hangul
        return "hangul"
    
    # Arabic
    elif 0x0600 <= first_char <= 0x06FF:
        return "arabic"
    
    # Devanagari (Hindi)
    elif 0x0900 <= first_char <= 0x097F:
        return "devanagari"
    
    # Cyrillic (Russian, etc.)
    elif 0x0400 <= first_char <= 0x04FF:
        return "cyrillic"
    
    # Latin (most European languages)
    elif 0x0020 <= first_char <= 0x024F:
        return "latin"
    
    return "unknown"

def calculate_heading_probability(span, font_level_map, all_spans):
    """
    Calculate probability that a span is a heading using multiple features including multilingual support
    """
    text = span["text"].strip()
    score = 0
    
    # Detect language
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    
    span["lang"] = lang
    
    # Font size score (40% of total score)
    font_level = font_level_map.get(span["font_size"], 5)
    font_score = max(0, 40 - (font_level - 1) * 8)  # H1=40, H2=32, H3=24
    score += font_score
    
    # Bold text bonus (15% of total score)
    if span["bold"]:
        score += 15
    
    # Multilingual numbering pattern bonus (20% of total score)
    numbered_patterns = MULTI_NUMBERED.get(lang, []) + MULTI_NUMBERED.get('en', [])
    for pattern in numbered_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            score += 20
            break
    
    # Fallback English patterns if no language-specific match
    if score == font_score + (15 if span["bold"] else 0):
        fallback_patterns = [
            r'^(\d+\.?\s+)',                     
            r'^(\d+\.\d+\.?\s+)',              
            r'^(\d+\.\d+\.\d+\.?\s+)',         
            r'^([A-Z]\.?\s+)',                 
            r'^([IVX]+\.?\s+)',                
        ]
        for pattern in fallback_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                score += 20
                break
    
    # Multilingual heading keywords (10% of total score)
    text_lower = text.lower()
    heading_keywords = MULTI_KEYWORDS.get(lang, []) + MULTI_KEYWORDS.get('en', [])
    for keyword in heading_keywords:
        if keyword in text_lower:
            score += 10
            break
    
    # Length penalty/bonus (10% of total score)
    if 3 <= len(text) <= 100:
        score += 10
    elif len(text) > 150:
        score -= 15
    
    # All caps bonus for reasonable length (5% of total score)
    if 5 <= len(text) <= 50 and text.isupper() and any(c.isalpha() for c in text):
        score += 5
    
    # Script-based multilingual bonus for detection (10% of total score)
    script_type = detect_script_type(text)
    if script_type in ["cjk", "hiragana", "katakana", "hangul", "arabic", "devanagari", "cyrillic"]:
        score += 10
    
    return score

def is_heading_candidate(span):
    # Enhanced filtering with multilingual support
    text = span["text"].strip()
    
    if len(text) < 2 or len(text) > 200:
        return False
    
    # Skip pure numbers, pure punctuation, or very common non-heading patterns
    if re.fullmatch(r'[\d\s\.\-_]+', text):
        return False
    
    if re.fullmatch(r'[^\w\s]+', text):  # Only punctuation
        return False
    
    # More lenient for non-Latin scripts
    script_type = detect_script_type(text)
    if script_type not in ["latin"]:
        return True  # Be more inclusive for non-Latin scripts
    
    # Skip if it's mostly lowercase with no special patterns (for Latin scripts only)
    if text.islower() and not any(pattern in text.lower() for pattern in ['introduction', 'abstract', 'summary', 'conclusion']):
        return False
    
    return True

def assign_heading_level_advanced(spans, font_level_map):
    """
    Advanced heading level assignment using probability scoring with multilingual support
    """
    headings = []
    
    # Calculate heading probabilities
    span_scores = []
    for span in spans:
        if not is_heading_candidate(span):
            continue
        
        score = calculate_heading_probability(span, font_level_map, spans)
        span_scores.append((span, score))
    
    # Sort by score (descending) and apply threshold
    span_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Dynamic threshold based on score distribution
    scores = [s[1] for s in span_scores]
    if scores:
        # Use 70th percentile as threshold, but minimum of 30
        threshold = max(30, np.percentile(scores, 70))
    else:
        threshold = 30
    
    for span, score in span_scores:
        if score < threshold:
            continue
        
        text = span["text"]
        
        # Determine level from numbering prefix first
        prefix_level = numbering_prefix_level(text)
        if prefix_level is not None:
            level = min(prefix_level, 4)  # Cap at H4
        else:
            # Use font-based level detection but adjust based on score
            font_level = font_level_map.get(span["font_size"], 5)
            if score >= 60:
                level = min(font_level, 1)  # High score = likely H1
            elif score >= 50:
                level = min(font_level, 2)  # Medium-high score = likely H1-H2
            elif score >= 40:
                level = min(font_level, 3)  # Medium score = likely H1-H3
            else:
                level = min(font_level, 4)  # Lower score = H1-H4
        
        headings.append({
            "text": text,
            "level": level,
            "page": span["page"],
            "score": score,
            "y_pos": span["y_pos"],
            "lang": span.get("lang", "unknown")
        })
    
    return headings

def extract_title(headings):
    # Enhanced title extraction with multilingual support
    if not headings:
        return "Untitled"
    
    # Look for title-like patterns first (multilingual)
    title_patterns = [
        r'.*overview.*foundation.*level.*extension.*',
        r'.*foundation.*level.*extension.*',
        r'.*syllabus.*',
        r'.*curriculum.*',
        r'.*guide.*',
        r'.*概要.*',  # Japanese overview
        r'.*指南.*',  # Chinese guide
        r'.*교과.*',  # Korean curriculum
    ]
    
    for heading in headings:
        text_lower = heading["text"].lower()
        for pattern in title_patterns:
            if re.search(pattern, text_lower):
                return heading["text"]
    
    # Find the first H1 heading
    h1_headings = [h for h in headings if h["level"] == 1]
    if h1_headings:
        # Sort by page and position
        h1_headings.sort(key=lambda x: (x["page"], x.get("y_pos", 0)))
        return h1_headings[0]["text"]
    
    # Fallback to first heading
    return headings[0]["text"] if headings else "Untitled"

def build_outline(headings, title):
    # Enhanced outline building with duplicate removal and multilingual support
    outline = []
    seen_texts = set()
    
    # Sort headings by page, then by y-position (top to bottom)
    sorted_headings = sorted(headings, key=lambda x: (x["page"], x.get("y_pos", 0)))
    
    for h in sorted_headings:
        text = h["text"].strip()
        lang = h.get("lang", "unknown")
        
        # Skip title and duplicates
        if text.lower() == title.lower() and h["level"] == 1:
            continue
        
        # Create a key for duplicate detection (case insensitive, page-aware)
        dup_key = (text.lower(), h["page"])
        if dup_key in seen_texts:
            continue
        seen_texts.add(dup_key)
        
        outline.append({
            "level": f"H{h['level']}",
            "text": text,
            "page": h["page"],
            "language": lang
        })
    
    return outline

def process_pdf(pdf_path):
    spans = extract_text_spans(pdf_path)
    if not spans:
        return {"title": "Untitled", "outline": []}
    
    font_level_map = analyze_font_hierarchy_statistical(spans)
    headings = assign_heading_level_advanced(spans, font_level_map)
    
    title = extract_title(headings)
    outline = build_outline(headings, title)
    
    return {
        "title": title,
        "outline": outline
    }

def process_pdfs(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            result = process_pdf(pdf_file)
            output_path = output_dir / f"{pdf_file.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(result, f_out, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python process_pdfs.py <input_folder> <output_folder>")
        exit(1)
    process_pdfs(sys.argv[1], sys.argv[2])
