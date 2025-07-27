import os
import time
from main import PDFOutlineExtractor
import json
import argparse

def test_batch_pdf_outline(input_folder="input", output_folder="output", optimize_speed=True):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load extractor with speed optimization
    start_time = time.time()
    extractor = PDFOutlineExtractor(optimize_speed=optimize_speed)
    print(f"Model loading time: {time.time() - start_time:.2f} seconds")

    # Process all PDFs in input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    total_time = 0
    successful = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        json_filename = os.path.splitext(pdf_file)[0] + ".json"
        output_path = os.path.join(output_folder, json_filename)

        print(f"Processing: {pdf_file} with optimized BERT (speed optimization: {optimize_speed})")

        try:
            # Time the processing
            start_time = time.time()
            result = extractor.extract_outline(pdf_path)
            processing_time = time.time() - start_time
            total_time += processing_time
            successful += 1
            
            # Add metadata
            result["metadata"] = {
                "filename": pdf_file,
                "model": "BERT-optimized",
                "optimize_speed": optimize_speed,
                "processing_time_seconds": round(processing_time, 2),
                "outline_count": len(result["outline"])
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved to {output_path} in {processing_time:.2f} seconds")

        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "title": "", 
                    "outline": [],
                    "metadata": {
                        "filename": pdf_file,
                        "model": "BERT-optimized",
                        "optimize_speed": optimize_speed,
                        "error": str(e)
                    }
                }, f, indent=2)
    
    # Print summary
    if successful > 0:
        avg_time = total_time / successful
        print(f"\nProcessed {successful} files in {total_time:.2f} seconds")
        print(f"Average processing time: {avg_time:.2f} seconds per file")
    else:
        print("\nNo files were processed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test optimized BERT PDF outline extraction")
    parser.add_argument("--input", default="input", help="Input directory containing PDFs")
    parser.add_argument("--output", default="output", help="Output directory for JSON files")
    parser.add_argument("--no-optimize", action="store_true", help="Disable speed optimization")
    
    args = parser.parse_args()
    
    test_batch_pdf_outline(args.input, args.output, not args.no_optimize)
