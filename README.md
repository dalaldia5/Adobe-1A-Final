# Adobe-1a

## Enhanced PDF Outline Extractor

This repository contains an advanced PDF outline extraction tool, originally built for the Adobe India Hackathon Round 1A. The tool leverages a fine-tuned BERT-based model and heuristic rules to extract structured outlines (Title, H1, H2, H3) from PDF documents, outputting results in JSON format.

---

## Features

- **Automatic Extraction**: Extracts document title and headings (H1, H2, H3) using ML and heuristics.
- **Batch Processing**: Supports processing all PDFs in a directory.
- **Custom Model Support**: Uses a mini-BERT model (`prajjwal1/bert-mini`) for sequence classification.
- **Rich Metadata**: Outputs additional metadata such as font size, style, position, and more for each text block.
- **Error Handling**: Skips non-text elements, page numbers, references, and logs errors gracefully.
- **Summary Report**: Generates a summary of all processed files.

---

## Directory Structure

```
Adobe-1a-bert/
├── main.py               # Main CLI and extraction logic
├── model_trainer.py      # (If provided) Model training scripts
├── test.py               # Batch test runner
├── requirements.txt      # Python dependencies
├── Dockerfile            # Containerization setup
├── models/
│   └── bert-mini/        # Custom BERT model files and vocabulary
│       └── vocab.txt
├── input/                # Place your PDF files here
├── output/               # Output JSON files will be saved here
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
pip install --upgrade pip
pip install -r requirements.txt
```

#### Key dependencies:

- `transformers`
- `torch`
- `PyMuPDF` (`fitz`)
- `numpy`
- `argparse`
- `logging`

---

## Usage

### 1. Prepare Input

- Place all PDF files you want to process in the `input/` directory.

### 2. Run the Extractor

#### Command Line

```bash
python main.py --input input --output output --model models/bert-mini
```

- `--input`: Path to the directory containing PDFs (default: `input`)
- `--output`: Path to the directory for saving JSON outputs (default: `output`)
- `--model`: Model path or Hugging Face model name (default: `prajjwal1/bert-mini` or local `models/bert-mini`)

#### Example Output

For each processed PDF, a corresponding JSON file will be created in the `output/` directory containing:
- Title
- Outline structure (H1/H2/H3 headings)
- Metadata (filename, timestamp, count)

A processing summary is saved as `_processing_summary.json`.

#### Batch Test

You can also run the batch test script:

```bash
python test.py
```
This will process all files in `input/` and save results in `output/`.

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

- The BERT model (mini version) is stored in `models/bert-mini/` and includes a custom vocabulary (`vocab.txt`).
- The model is loaded via Hugging Face `transformers` in `main.py`.

---

## Customization

- You can retrain or fine-tune the model using scripts in `model_trainer.py` if provided.
- Heuristic rules and classification patterns are defined in `main.py` and can be adjusted for different document structures.

---

## Logging & Troubleshooting

- All major actions, errors, and summaries are logged using Python's `logging` module.
- If a PDF fails to process, details are captured in the summary file.

---


