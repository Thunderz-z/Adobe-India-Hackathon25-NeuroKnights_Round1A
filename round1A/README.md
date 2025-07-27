# Adobe Hackathon Round 1A Submission

## Overview and Approach

This project implements an automated system to extract structured outlines from PDF documents across multiple languages with advanced multilingual support. The core logic involves parsing PDF text spans using PyMuPDF (fitz) to access low-level font and layout details, followed by statistical analysis on font sizes to infer heading hierarchy rather than relying on hardcoded font thresholds or clustering algorithms.

The solution incorporates multilingual pattern recognition to detect numbered sections, keywords, and culturally relevant heading indicators across English, Spanish, French, German, Italian, Portuguese, Russian, Arabic, Chinese, Japanese, Korean, and Hindi. Language detection using langdetect improves pattern matching and weighting of heading likelihood. The system combines multiple features including font size, bold/italic style, textual patterns, uppercase detection, and language-specific keywords to probabilistically score text spans as headings.

Finally, the approach constructs a clean hierarchical outline with heading levels (H1-H4), implementing duplicate removal and title extraction heuristics. This methodology ensures robust, adaptable performance across diverse documents with varied layouts and languages, as often encountered in real-world PDF corpora.

## Models and Libraries Used

The solution uses the following libraries:

- **PyMuPDF (fitz)** – A powerful PDF parsing library for extracting text with font and layout metadata
- **numpy & scipy** – For numeric and statistical operations to analyze font size distributions
- **langdetect** – To identify text languages and support multilingual heuristics
- **tqdm** – For displaying progress bars during batch processing
- **Python Standard Library** – e.g., `re`, `json`, `pathlib`, `collections`

No pretrained machine learning models are used, so there is no concern with model size constraints. The entire solution is based on heuristic, language-aware text processing.

## How to Build and Run The Solution

### Prerequisites

- Docker installed on your system (Windows, macOS, Linux)
- At least 8 CPU cores and 16 GB RAM recommended (as per hackathon constraints)

### Building the Docker Image

Run the following command in your project directory, which contains your `Dockerfile`, `requirements.txt`, and code:

```bash
docker build --platform linux/amd64 -t solutionname:round1a .
```
- you can change the name and identifier as you like. Above is just an example

This will:
- Use a Python 3.11 slim base image
- Install required system and Python packages
- Prepare all dependencies inside the image

### Running the Docker Container

Prepare your local directories:
- Create an `input` folder and place all PDF files you want to process inside it
- Create an empty `output` folder where JSON outputs will be saved. If a output folder does not exist, it will automatically be created at runtime

To run the container and process PDFs offline (network disabled as required), execute:

#### On Linux/macOS:

```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none solutionname:round1a
```

#### On Windows Command Prompt:

```cmd
docker run --rm -v "%cd%\input:/app/input" -v "%cd%\output:/app/output" --network none solutionname:round1a
```

#### On Windows PowerShell:

```powershell
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none solutionname:round1a
```

### Expected Behavior

- The container automatically processes all PDF files in `/app/input`
- For each `filename.pdf`, it generates a corresponding `filename.json` outline in `/app/output`
- The output JSON contains a well-structured hierarchical outline with multilingual language tags and heading levels

