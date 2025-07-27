# Adobe-1a

## Enhanced PDF Outline Extractor

This repository contains an advanced PDF outline extraction tool, originally built for the Adobe India Hackathon Round 1A. The tool leverages an optimized BERT-based model and heuristic rules to extract structured outlines (Title, H1, H2, H3) from PDF documents, outputting results in JSON format.

### Time-Optimized Implementation

This implementation has been optimized to process documents in under 10 seconds per file. The optimizations include:

- Caching of extracted text blocks and classification results
- Limited page processing (only first 10 pages by default)
- Selective block classification based on heuristic filtering
- Larger batch sizes for more efficient processing
- Reduced sequence length for faster tokenization

---

## Features

- **Automatic Extraction**: Extracts document title and headings (H1, H2, H3) using ML and heuristics.
- **Time-Optimized**: Processes documents in under 10 seconds per file.
- **Batch Processing**: Supports processing all PDFs in a directory.
- **Rich Metadata**: Outputs additional metadata such as font size, style, position, and more for each text block.
- **Error Handling**: Skips non-text elements, page numbers, references, and logs errors gracefully.
- **Summary Report**: Generates a summary of all processed files.

---

## Directory Structure

```
Adobe-1a-bert/
├── main.py                # Main CLI and extraction logic (optimized BERT-based)
├── test.py                # Batch test runner for optimized BERT model
├── run_comparison.sh      # Script to run the optimized model
├── requirements.txt       # Python dependencies
├── Dockerfile             # Containerization setup
├── models/
│   └── bert-mini/         # Custom BERT model files and vocabulary
│       └── vocab.txt
├── input/                 # Place your PDF files here
├── output/                # Output JSON files will be saved here
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HansujaB/Adobe-1a-bert.git
cd Adobe-1a-bert
```

### 2. Python Environment

Ensure you have Python 3.10+ installed. It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

#### Key dependencies:

- `transformers`
- `torch`
- `PyMuPDF` (`fitz`)
- `numpy`
- `matplotlib` (for visualization)

---

## Usage

### 1. Prepare Input

- Place all PDF files you want to process in the `input/` directory.

### 2. Run the Extractor

#### With Speed Optimization (Default)

Processes documents in under 10 seconds per file:

```bash
python main.py --optimize
```

#### Without Speed Optimization

For maximum accuracy (slower processing):

```bash
python main.py
```

#### Run Batch Test

```bash
./run_comparison.sh
```

or

```bash
python test.py --optimize
```

#### Example Output

For each processed PDF, a corresponding JSON file will be created in the `output/` directory containing:

- Title
- Outline structure (H1/H2/H3 headings)
- Metadata (filename, timestamp, count, processing time)

A processing summary is saved as `_processing_summary.json`.

---

## Performance

The optimized BERT model processes documents in under 10 seconds per file, typically achieving:

- 5-10 seconds per file with optimization enabled
- Comparable accuracy to the non-optimized version

---

## Docker Usage

A Dockerfile is provided for easy containerization:

```bash
docker build -t adobe-1a-bert .
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output adobe-1a-bert
```

- The default container command runs `python test.py`.

---

## Model Details

### Optimized BERT Model

- Uses the BERT-mini model with custom classification head
- Combines ML-based classification with heuristic rules
- Implements caching and selective processing for speed
- Processes only the most relevant text blocks

---

## Customization

- You can adjust the optimization parameters in `main.py` to balance speed and accuracy.
- Heuristic rules and classification patterns are defined in the Python files and can be adjusted for different document structures.

---

## Logging & Troubleshooting

- All major actions, errors, and summaries are logged using Python's `logging` module.
- If a PDF fails to process, details are captured in the summary file.

---
