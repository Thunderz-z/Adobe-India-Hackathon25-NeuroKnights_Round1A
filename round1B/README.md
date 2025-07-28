# Adobe Hackathon Round 1B - Persona-Driven Document Intelligence

## Project Overview

This solution implements a **Persona-Driven Document Intelligence System** that processes collections of PDF documents to extract and rank relevant sections based on a specific persona and job-to-be-done. The system combines advanced text processing, semantic embeddings, and intelligent ranking algorithms to provide targeted document analysis tailored to different user roles and objectives.

## Project Structure

```
Challenge_1b/
├── Collection 1/                    # Academic Research
│   ├── PDFs/                       # Research papers on Graph Neural Networks for Drug Discovery
│   ├── challenge1b_input.json      # PhD Researcher in Computational Biology configuration
│   └── challenge1b_output.json     # Literature review analysis results
├── Collection 2/                    # Business Analysis
│   ├── PDFs/                       # Annual reports from competing tech companies (2022-2024)
│   ├── challenge1b_input.json      # Investment Analyst configuration
│   └── challenge1b_output.json     # Revenue trends and market positioning analysis
├── Collection 3/                    # Educational Content
│   ├── PDFs/                       # Organic chemistry textbook chapters
│   ├── challenge1b_input.json      # Undergraduate Chemistry Student configuration
│   └── challenge1b_output.json     # Key concepts and mechanisms for exam preparation
├── adobe_1b.py                     # Main processing script
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container configuration
└── README.md                        # This file
```

## Methodology and Approach

The core methodology of this persona-driven document intelligence system revolves around semantic understanding and contextual relevance matching between user personas, their specific objectives, and document content. The system begins by extracting structured sections from PDF documents using PyMuPDF's advanced text parsing capabilities, which preserves font information, layout details, and hierarchical structure. This extraction process employs sophisticated heuristics to identify section boundaries through font size analysis, formatting patterns, and common academic or business document structures.

Once sections are extracted, the system leverages the pre-trained sentence transformer model "all-MiniLM-L6-v2" to create dense vector representations of both the document sections and the combined persona-job query. This lightweight but effective model generates 384-dimensional embeddings that capture semantic meaning and contextual relationships in the text. The choice of this particular model balances computational efficiency with semantic accuracy, making it suitable for CPU-only execution while maintaining high-quality results across diverse document types and domains.

The ranking and relevance determination process uses cosine similarity calculations between the persona-job query embeddings and section embeddings to identify the most relevant content. The system implements an adaptive threshold mechanism that ensures substantial output regardless of document collection size or content diversity. This approach prevents situations where overly strict similarity thresholds might result in insufficient relevant sections, while also maintaining quality by filtering out clearly irrelevant content.

Sub-section analysis represents another crucial component of the methodology, where the system performs sentence-level tokenization using NLTK and applies the same semantic matching principles at a more granular level. This dual-level analysis allows for both broad section identification and detailed content extraction, providing users with both structural overview and specific textual insights. The system intelligently handles various sentence boundary detection challenges and implements fallback mechanisms to ensure robust performance across different document formats and writing styles.

The persona-driven aspect of the system is implemented through query construction that combines the user's role description with their specific task objectives, creating a comprehensive context vector that guides the relevance matching process. This approach ensures that the same document collection can yield different results when analyzed from different perspectives, such as an investment analyst focusing on financial metrics versus a researcher examining methodological approaches. The system's architecture supports extensibility across various domains and use cases while maintaining consistent performance and output quality standards.

## Models and Libraries Used

### Core Libraries
- **PyMuPDF (fitz) 1.26.3** - Advanced PDF parsing and text extraction with layout preservation
- **sentence-transformers 5.0.0** - Semantic embeddings using the all-MiniLM-L6-v2 model
- **scikit-learn 1.5.2** - Cosine similarity calculations for relevance ranking
- **nltk 3.9.1** - Natural language processing and sentence tokenization
- **langdetect 1.0.9** - Language detection for multilingual document support
- **tqdm 4.67.1** - Progress tracking for batch processing operations

### Pre-trained Models
- **all-MiniLM-L6-v2** - A 384-dimensional sentence transformer model (approximately 90MB)
  - Optimized for semantic similarity tasks
  - Supports multilingual content understanding
  - CPU-efficient architecture suitable for production deployment

## System Requirements

- **Architecture**: AMD64 (x86_64) CPU-only execution
- **Memory**: Minimum 8GB RAM, 16GB recommended
- **Processing**: Multi-core CPU support for parallel document processing
- **Network**: Offline execution capability (no internet required at runtime)

## How to Build and Run

### Prerequisites
- Docker installed and configured for AMD64 platform
- Minimum system requirements as specified above

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t persona-intelligence:round1b .
```
- you can change the name and identifier as you like. Above is just an example

### Running the Docker image

For each collection, execute the following command structure:

### Execution Method


```bash
docker run --rm -v "${PWD}:/app" --network none persona-intelligence:round1b python adobe_1b.py "Collection 1"
```


#### Collection 1 (Academic Research):
```bash
docker run --rm -v "${PWD}:/app" --network none persona-intelligence:round1b python adobe_1b.py "Collection 1"
```

#### Collection 2 (Business Analysis):
```bash
docker run --rm -v "${PWD}:/app" --network none persona-intelligence:round1b python adobe_1b.py "Collection 2"
```

#### Collection 3 (Educational Content):
```bash
docker run --rm -v "${PWD}:/app" --network none persona-intelligence:round1b python adobe_1b.py "Collection 3"
```

If you wish to create a new folder and add new sample pdf files to analyze. Follow the steps below:
- Create a new folder called "Collection 4" (It must follow the same naming convention) 
- Create a Folder named PDFs and place your input pdf files there
- Create the json file with the name: challenge1b_input.json. (The exact name as specified is required)
- Essentially all new Collections must follow the project structure and steps mentioned above
- Run the Command mentioned in the Main Execution step but simply change the collection number at the end (For example if you wish to run Collection 4):

```bash
docker run --rm -v "${PWD}:/app" --network none persona-intelligence:round1b python adobe_1b.py "Collection 4"
```
The output will be in the respective Collection folder.




## Output Format

The system generates structured JSON output containing:

- **Metadata**: Input documents, persona definition, job-to-be-done, and processing timestamp
- **Extracted Sections**: Ranked list of relevant sections with importance scoring
- **Sub-section Analysis**: Detailed sentence-level analysis with refined text extraction

Each collection produces a `challenge1b_output.json` file in its respective directory with comprehensive analysis results tailored to the specified persona and objectives.

## Technical Notes

- All models and dependencies are pre-installed during Docker build phase
- Execution requires no internet connectivity (fully offline operation)
- System automatically handles 3-10 PDF documents per collection
- Processing time optimized for 60-second execution constraint
- Cross-platform compatibility through standardized containerization
- Building the image may take ~8-10 minutes the first time, also depending on the hardware specs
