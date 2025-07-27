# Adobe India Hackathon Round 1A - PDF Outline Extractor

## Problem Statement

You're handed a PDF — but instead of simply reading it, you're tasked with making sense of it like a machine would. Your job is to extract a structured outline of the document — essentially the Title, and headings like H1, H2, and H3 — in a clean, hierarchical format.

This outline will be the foundation for the rest of your hackathon journey.

### Why This Matters

PDFs are everywhere — but machines don't naturally understand their structure. By building an outline extractor, you're enabling smarter document experiences, like semantic search, recommendation systems, and insight generation.

### What You Need to Build

You must build a solution that:

• **Accepts a PDF file** (up to 50 pages)
• **Extracts:**
  - Title
  - Headings: H1, H2, H3 (with level and page number)
• **Outputs a valid JSON file** in the format below:

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

## Our Solution

This repository contains our complete solution - an **Enhanced PDF Outline Extractor** that combines machine learning with intelligent heuristics to accurately extract document structure.

### Key Features

🤖 **Hybrid AI Approach**: Combines a fine-tuned BERT-mini model with carefully crafted heuristic rules for maximum accuracy

📊 **Rich Text Analysis**: Analyzes font sizes, styles, positioning, and content patterns to identify headings

🔄 **Batch Processing**: Processes entire directories of PDFs automatically

📈 **Smart Post-Processing**: Ensures proper hierarchical structure and removes duplicates

🐳 **Containerized**: Fully dockerized for easy deployment and testing

### Technical Architecture

#### 1. Text Extraction (`extract_text_blocks`)
- Uses PyMuPDF to extract text blocks with comprehensive metadata
- Captures font sizes, styles (bold/italic), positioning, and content features
- Applies intelligent filtering to remove page numbers, references, and noise

#### 2. Heuristic Classification (`apply_heuristic_rules`)
- Pattern-based detection for numbered sections, chapter headings
- Font size analysis using percentiles and ratios
- Position-based rules (titles typically appear early and high on pages)

#### 3. ML Classification (`classify_text_blocks`)
- Fine-tuned BERT-mini model for sequence classification
- 5-class classification: Title, H1, H2, H3, Other
- Rich feature engineering combining text with structural metadata

#### 4. Post-Processing (`post_process_predictions`)
- Hierarchical structure validation and correction
- Duplicate removal with fuzzy matching
- Confidence-based filtering with level-specific thresholds

### Project Structure

```
Adobe-1a-bert/
├── main.py               # Core extraction logic with PDFOutlineExtractor class
├── test.py               # Batch processing script for testing
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── models/
│   └── bert-mini/       # Fine-tuned BERT model and vocabulary
├── input/               # Directory for input PDF files
├── output/              # Directory for output JSON files
└── explanation.md       # This file - problem statement and solution overview
```

### Usage

#### Quick Start
```bash
# Clone and setup
git clone https://github.com/HansujaB/Adobe-1a-bert.git
cd Adobe-1a-bert
pip install -r requirements.txt

# Place PDFs in input/ directory
cp your-pdfs/*.pdf input/

# Run extraction
python main.py

# Check results in output/ directory
```

#### Docker Usage
```bash
docker build -t adobe-1a-bert .
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output adobe-1a-bert
```

### Performance Highlights

✅ **High Accuracy**: Hybrid ML + heuristics approach achieves superior results over pure ML or rule-based methods

✅ **Robust Processing**: Handles various PDF formats, fonts, and layouts

✅ **Scalable**: Batch processing with comprehensive error handling and logging

✅ **Hierarchical Validation**: Ensures proper H1→H2→H3 structure in output

### Output Format

Each processed PDF generates a JSON file with:
- **title**: Extracted document title
- **outline**: Array of headings with level, text, and page number
- **metadata**: Processing information including timestamps and statistics


This solution successfully addresses the hackathon challenge by providing a production-ready PDF outline extraction system that combines the best of machine learning and traditional rule-based approaches.
