import fitz
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from langdetect import detect, DetectorFactory


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize


# Ensure NLTK data is ready

nltk.data.find('tokenizers/punkt_tab')
nltk.data.find('tokenizers/punkt')



DetectorFactory.seed = 0


def extract_sections(pdf_path):
    """
    Generalized section extraction for any document type
    """
    doc = fitz.open(pdf_path)
    section_list = []
    current_section = None
    
    for page_number, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0: continue
            
            # Get font information for this block
            font_sizes = []
            is_bold = False
            line_text = ""
            
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span['text'].strip()
                    if text:
                        line_text += text + " "
                        font_sizes.append(span['size'])
                        if span.get('flags', 0) & 2:  # Bold flag
                            is_bold = True
            
            line_text = line_text.strip()
            if not line_text or len(line_text) < 3: 
                continue


            # GENERALIZED heading detection (works for any domain)
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10
            
            is_heading = (
                len(line_text) < 150 and (
                    # Numbered sections (universal pattern)
                    bool(re.match(r'^(\d+(\.\d+)*[\.\-]?\s+)', line_text)) or
                    bool(re.match(r'^([A-Z][\.\)]?\s+)', line_text)) or  # A. B. etc.
                    bool(re.match(r'^([IVX]+[\.\)]?\s+)', line_text)) or  # Roman numerals
                    
                    # Font-based detection (universal)
                    (is_bold and len(line_text) < 100) or
                    (avg_font_size > 12 and len(line_text) < 80) or
                    
                    # Case-based patterns (universal)
                    (line_text.isupper() and 5 < len(line_text) < 60) or
                    (line_text.istitle() and len(line_text) < 80 and ' ' in line_text) or
                    
                    # Common academic/business patterns (broad categories)
                    bool(re.match(r'^(abstract|introduction|summary|conclusion|references|bibliography|appendix|methodology|results|discussion|background|literature|analysis|evaluation|assessment|findings|recommendations|executive summary|table of contents|acknowledgements|preface)', line_text.lower())) or
                    bool(re.match(r'^(chapter|section|part|volume|book|unit)\s+\d+', line_text.lower()))
                )
            )
            
            if is_heading:
                if current_section and len(current_section["content"].strip()) > 20:
                    section_list.append(current_section)
                current_section = {
                    "section_title": line_text,
                    "page": page_number,
                    "content": ""
                }
            elif current_section:
                current_section["content"] += " " + line_text
    
    # Including the last section
    if current_section and len(current_section["content"].strip()) > 20:
        section_list.append(current_section)
    
    doc.close()
    return section_list


def get_miniLM():
    return SentenceTransformer("/models/all-MiniLM-L6-v2")


def build_section_embeddings(sections, model):
    sentences = [((sec["section_title"] or "") + " " + sec["content"][:800]).strip() for sec in sections]
    embeddings = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def adaptive_threshold(scores, min_results=8, percentile=85):
    """
    Ensures we always have substantial output
    """
    if len(scores) < min_results:
        return 0.0  # Return everything if too few results based on the input pdf
    
    threshold = max(0.05, min(0.25, scores[min_results-1]))
    return threshold


def salient_sections(sections, embeddings, query, topn=15):  
    if not embeddings.any() or len(sections) == 0:
        return []
    
    model = get_miniLM()
    query_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, embeddings)[0]
    ranked = sorted(zip(range(len(sections)), sims), key=lambda x: -x[1])
    
    # Extract scores for threshold calculation
    scores = [score for idx, score in ranked]
    threshold = adaptive_threshold(scores, min_results=8)  
    
    result_idxs = []
    for idx, score in ranked[:topn]:
        if score >= threshold or len(result_idxs) < 5:
            result_idxs.append((idx, score))
    
    return [(sections[idx], score) for idx, score in result_idxs]


def extract_subsections(section, persona_job, model, topn=5, char_limit=350):  
    """
    sub-section extraction
    """
    try:
        sentences = sent_tokenize(section["content"])
    except:
        content = section["content"]
        sentences = re.split(r'[.!?]+\s+', content)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 15]  # Lowered from 20
    if not sentences: 
        return []
    
    # Sentence and section optimization
    if len(sentences) > 40:  
        sentences = sentences[:40]
    
    try:
        sent_embs = model.encode(sentences, convert_to_numpy=True)
        query_emb = model.encode([persona_job], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, sent_embs)[0]
        
        sorted_sims = sorted(sims, reverse=True)
        sent_threshold = max(0.02, min(0.15, sorted_sims[min(3, len(sorted_sims)-1)]))  # Optimal baseline to get a accurate output
        
        pairs = sorted(zip(sentences, sims), key=lambda x: -x[1])
        top_subsections = []
        
        for sent, score in pairs[:topn]:
            if score >= sent_threshold or len(top_subsections) < 2:
                top_subsections.append(sent[:char_limit])
        
        return top_subsections
    except Exception as e:
        print(f"Error in sentence processing: {e}")
        # Fallback: return something rather than nothing
        return [sent[:char_limit] for sent in sentences[:topn]]


def process(pdf_files, persona, job_to_be_done):
    model = get_miniLM()
    all_sections = []


    # Step 1: Extracting sections from all PDFs
    print("Extracting sections...")
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        sections = extract_sections(pdf_file)
        for section in sections:
            section['document'] = pdf_file.name
        all_sections.extend(sections)


    print(f"Total sections extracted: {len(all_sections)}")
    
    if len(all_sections) == 0:
        return {
            "metadata": {
                "input_documents": [p.name for p in pdf_files],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }


    # Step 2: Embed all sections
    print("Creating embeddings...")
    embeddings = build_section_embeddings(all_sections, model)
    
    # Query combining persona and job
    persona_query = f"{persona.strip()} {job_to_be_done.strip()}"
    print(f"Query: {persona_query}")
    
    top_sections_with_scores = salient_sections(all_sections, embeddings, persona_query, topn=15)
    print(f"Found {len(top_sections_with_scores)} relevant sections")


    # Step 3: Building output
    extracted_sections = []
    sub_section_analysis = []
    
    print("Analyzing sub-sections...")
    for rank, (section, score) in enumerate(top_sections_with_scores, 1):
        extracted_sections.append({
            "document": section['document'],
            "page": section['page'],
            "section_title": section['section_title'],
            "importance_rank": rank
        })
        
        try:
            top_sents = extract_subsections(section, persona_query, model, topn=5, char_limit=350)  
            for sent in top_sents:
                sub_section_analysis.append({
                    "document": section['document'],
                    "page": section['page'],
                    "refined_text": sent
                })
        except Exception as e:
            print(f"Error processing sub-sections: {e}")
            # FALLBACK: analyze some content as output even if processing fails
            if section["content"]:
                fallback_text = section["content"][:350].strip()
                if fallback_text:
                    sub_section_analysis.append({
                        "document": section['document'],
                        "page": section['page'],
                        "refined_text": fallback_text
                    })
            continue


    # Minimum output quantities
    if len(extracted_sections) < 3 and len(all_sections) >= 3:
        # Add more sections if too few numbers
        for i, section in enumerate(all_sections[:10]):
            if len(extracted_sections) >= 10:
                break
            section_key = (section['document'], section['page'], section['section_title'])
            existing_keys = [(s['document'], s['page'], s['section_title']) for s in extracted_sections]
            if section_key not in existing_keys:
                extracted_sections.append({
                    "document": section['document'],
                    "page": section['page'],
                    "section_title": section['section_title'],
                    "importance_rank": len(extracted_sections) + 1
                })


    if len(sub_section_analysis) < 5 and len(all_sections) > 0:
        # sub-sections quantity
        for section in all_sections[:5]:
            if len(sub_section_analysis) >= 20:
                break
            if section["content"]:
                fallback_text = section["content"][:350].strip()
                if fallback_text:
                    sub_section_analysis.append({
                        "document": section['document'],
                        "page": section['page'],
                        "refined_text": fallback_text
                    })


    # Step 4: Format output
    output = {
        "metadata": {
            "input_documents": [p.name for p in pdf_files],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }
    
    print(f"Final output contains {len(extracted_sections)} sections and {len(sub_section_analysis)} sub-sections")
    return output


def load_input_config(collection_path):
    """
    Load configuration from challenge1b_input.json file in the collection directory
    """
    try:
        input_json_path = collection_path / "challenge1b_input.json"
        with open(input_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract persona
        persona = config['persona']['role']
        
        # Extract job to be done
        job_to_be_done = config['job_to_be_done']['task']
        
        # Extract document filenames
        document_filenames = [doc['filename'] for doc in config['documents']]
        
        return persona, job_to_be_done, document_filenames, config
        
    except Exception as e:
        print(f"Error loading input JSON from {collection_path}: {e}")
        sys.exit(1)


def find_pdf_files(document_filenames, collection_path):
    """
    Find PDF files in the PDFs subdirectory of the collection
    """
    pdfs_dir = collection_path / "PDFs"
    found_files = []
    
    if not pdfs_dir.exists():
        print(f"Error: PDFs directory not found: {pdfs_dir}")
        return found_files
    
    for filename in document_filenames:
        pdf_path = pdfs_dir / filename
        if pdf_path.exists():
            found_files.append(pdf_path)
        else:
            print(f"Warning: Could not find PDF file: {filename} in {pdfs_dir}")
    
    return found_files


def main():
    if len(sys.argv) != 2:
        print("Usage: python adobe_1b.py <collection_name>")
        print('Example: python adobe_1b.py "Collection 1"')
        sys.exit(1)
    
    collection_name = sys.argv[1]
    collection_path = Path(collection_name)
    
    if not collection_path.exists():
        print(f"Error: Collection directory '{collection_name}' not found!")
        sys.exit(1)
    
    # Load configuration from collection directory
    persona, job_to_be_done, document_filenames, config = load_input_config(collection_path)
    
    print(f"Loaded configuration from {collection_name}:")
    print(f"Persona: {persona}")
    print(f"Job to be done: {job_to_be_done}")
    print(f"Documents: {len(document_filenames)} files")
    
    # Find PDF files in the collection's PDFs directory
    pdf_files = find_pdf_files(document_filenames, collection_path)
    
    if len(pdf_files) == 0:
        print("Error: No PDF files found!")
        print(f"Make sure PDF files are in the '{collection_path}/PDFs/' folder")
        sys.exit(1)
    
    if len(pdf_files) < 3:
        print("Warning: Less than 3 PDFs found. Processing will continue but results may be limited.")
    
    if len(pdf_files) > 10:
        print("Warning: More than 10 PDFs found. Only the first 10 will be processed.")
        pdf_files = pdf_files[:10]
    
    print(f"Processing {len(pdf_files)} PDFs...")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Process the documents
    output = process(pdf_files, persona, job_to_be_done)
    
    # Save JSON output to the collection directory
    output_file = collection_path / "challenge1b_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Done. Output written to {output_file}")


if __name__ == "__main__":
    main()
