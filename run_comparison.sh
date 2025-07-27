#!/bin/bash

# Run optimized BERT model for PDF outline extraction

# Check if input directory exists
if [ ! -d "input" ]; then
  echo "Error: 'input' directory not found. Please create it and add PDF files."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

echo "===== Running optimized BERT model ====="
python test.py

echo "===== Done! ====="
echo "Results are saved in the 'output' directory."
echo "Processing summary is saved as 'output/_processing_summary.json'" 