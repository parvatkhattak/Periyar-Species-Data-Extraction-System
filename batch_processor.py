#!/usr/bin/env python3
"""
Simplified Batch PDF Processor for Periyar Species Extraction
============================================================

This script provides a simple interface for processing multiple PDFs
and generating species data tables efficiently.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
import argparse

# Import the main extractor
try:
    from periyar_extractor import PeriyarSpeciesExtractor
except ImportError:
    print("Error: periyar_extractor.py not found in the same directory!")
    print("Please ensure both files are in the same folder.")
    sys.exit(1)

class BatchProcessor:
    """Simplified batch processor for multiple PDFs"""
    
    def __init__(self, api_key: str = None):
        self.extractor = PeriyarSpeciesExtractor(api_key)
        self.results = []
        self.failed_files = []
        
    def setup(self, api_key: str = None):
        """Setup the processor"""
        print("Setting up Periyar Species Extractor...")
        try:
            self.extractor.setup_gemini(api_key)
            print("✓ Gemini API configured successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to setup Gemini API: {e}")
            return False
    
    def process_single_pdf(self, pdf_path: str) -> bool:
        """Process a single PDF file"""
        try:
            print(f"Processing: {os.path.basename(pdf_path)}")
            species_data = self.extractor.process_pdf(pdf_path)
            
            if species_data:
                self.results.extend(species_data)
                print(f"✓ Extracted {len(species_data)} species")
                return True
            else:
                print("⚠ No species data found")
                self.failed_files.append((pdf_path, "No species data extracted"))
                return False
                
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {e}")
            self.failed_files.append((pdf_path, str(e)))
            return False
    
    def process_directory(self, directory_path: str, file_pattern: str = "*.pdf"):
        """Process all PDFs in a directory"""
        if not os.path.exists(directory_path):
            print(f"Error: Directory '{directory_path}' not found!")
            return False
        
        # Find all PDF files
        pdf_files = []
        for file in os.listdir(directory_path):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(directory_path, file))
        
        if not pdf_files:
            print("No PDF files found in the directory!")
            return False
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print("-" * 50)
        
        # Process each PDF
        successful = 0
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] ", end="")
            if self.process_single_pdf(pdf_path):
                successful += 1
            print()
        
        print("-" * 50)
        print(f"Processing complete: {successful}/{len(pdf_files)} files successful")
        
        if self.failed_files:
            print(f"Failed files: {len(self.failed_files)}")
            for failed_file, error in self.failed_files:
                print(f"  - {os.path.basename(failed_file)}: {error}")
        
        return successful > 0
    
    def save_results(self, output_dir: str = "output", base_name: str = None):
        """Save results to CSV and Excel files"""
        if not self.results:
            print("No data to save!")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base filename if not provided
        if not base_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"periyar_species_{timestamp}"
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['species_name', 'reference'])
        df = df.sort_values(['species_name', 'reference'])
        df = df.reset_index(drop=True)
        
        # Save files
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
        
        try:
            self.extractor.save_to_csv(df, csv_path)
            self.extractor.save_to_excel(df, excel_path)
            
            print(f"Results saved:")
            print(f"  CSV: {csv_path}")
            print(f"  Excel: {excel_path}")
            
            # Print summary
            self.print_summary(df)
            
            return True
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def print_summary(self, df: pd.DataFrame):
        """Print data summary"""
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(f"Total species extracted: {len(df)}")
        
        # Flora/Fauna breakdown
        flora_count = len(df[df['flora_or_fauna'].str.contains('Flora', case=False, na=False)])
        fauna_count = len(df[df['flora_or_fauna'].str.contains('Fauna', case=False, na=False)])
        unknown_count = len(df) - flora_count - fauna_count
        
        print(f"Flora species: {flora_count}")
        print(f"Fauna species: {fauna_count}")
        if unknown_count > 0:
            print(f"Unknown classification: {unknown_count}")
        
        # Location data
        with_coords = len(df[(df['latitude'].notna()) & (df['longitude'].notna()) & 
                            (df['latitude'] != '') & (df['longitude'] != '')])
        print(f"Species with coordinates: {with_coords}")
        
        # Date information
        with_dates = len(df[(df['date_of_observation'].notna()) & 
                           (df['date_of_observation'] != '')])
        print(f"Species with observation dates: {with_dates}")
        
        # Reference files
        unique_refs = df['reference'].nunique()
        print(f"Source files processed: {unique_refs}")
        
        print("="*50)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Batch process PDFs for Periyar species extraction"
    )
    parser.add_argument(
        "input_path", 
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "-o", "--output", 
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "-n", "--name",
        help="Base name for output files (auto-generated if not provided)"
    )
    parser.add_argument(
        "-k", "--api-key",
        help="Gemini API key (will prompt if not provided)"
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Save only CSV format"
    )
    parser.add_argument(
        "--excel-only",
        action="store_true", 
        help="Save only Excel format"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Setup API
    if not processor.setup(args.api_key):
        return 1
    
    # Process input
    if os.path.isfile(args.input_path):
        # Single file
        print(f"Processing single PDF: {args.input_path}")
        if not processor.process_single_pdf(args.input_path):
            print("Failed to process the PDF file.")
            return 1
    elif os.path.isdir(args.input_path):
        # Directory
        print(f"Processing directory: {args.input_path}")
        if not processor.process_directory(args.input_path):
            print("Failed to process PDFs in directory.")
            return 1
    else:
        print(f"Error: '{args.input_path}' is not a valid file or directory!")
        return 1
    
    # Save results
    if processor.results:
        # Modify save method based on format preferences
        if args.csv_only or args.excel_only:
            processor.save_results_custom(
                output_dir=args.output,
                base_name=args.name,
                save_csv=not args.excel_only,
                save_excel=not args.csv_only
            )
        else:
            processor.save_results(args.output, args.name)
    else:
        print("No species data extracted from any files.")
        return 1
    
    return 0

# Add custom save method to BatchProcessor class
def save_results_custom(self, output_dir: str = "output", base_name: str = None, 
                       save_csv: bool = True, save_excel: bool = True):
    """Save results with format options"""
    if not self.results:
        print("No data to save!")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename if not provided
    if not base_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"periyar_species_{timestamp}"
    
    # Create DataFrame
    df = pd.DataFrame(self.results)
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['species_name', 'reference'])
    df = df.sort_values(['species_name', 'reference'])
    df = df.reset_index(drop=True)
    
    # Save files based on preferences
    saved_files = []
    
    if save_csv:
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        try:
            self.extractor.save_to_csv(df, csv_path)
            saved_files.append(f"CSV: {csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    if save_excel:
        excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
        try:
            self.extractor.save_to_excel(df, excel_path)
            saved_files.append(f"Excel: {excel_path}")
        except Exception as e:
            print(f"Error saving Excel: {e}")
    
    if saved_files:
        print("Results saved:")
        for file_info in saved_files:
            print(f"  {file_info}")
        
        # Print summary
        self.print_summary(df)
        return True
    else:
        print("Failed to save any files!")
        return False

# Add the custom method to BatchProcessor class
BatchProcessor.save_results_custom = save_results_custom

def interactive_mode():
    """Interactive mode for easy usage"""
    print("="*60)
    print("PERIYAR SPECIES EXTRACTION SYSTEM - INTERACTIVE MODE")
    print("="*60)
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Setup API
    print("\n1. Setting up Gemini API...")
    if not processor.setup():
        print("Failed to setup API. Exiting.")
        return
    
    # Get input path
    print("\n2. Select input source:")
    while True:
        input_path = input("Enter path to PDF file or directory: ").strip().strip('"')
        if os.path.exists(input_path):
            break
        print("Path not found. Please try again.")
    
    # Process files
    print("\n3. Processing files...")
    if os.path.isfile(input_path):
        processor.process_single_pdf(input_path)
    else:
        processor.process_directory(input_path)
    
    if not processor.results:
        print("No species data extracted. Exiting.")
        return
    
    # Output options
    print("\n4. Output options:")
    output_dir = input("Output directory (press Enter for 'output'): ").strip()
    if not output_dir:
        output_dir = "output"
    
    base_name = input("Output filename base (press Enter for auto-generated): ").strip()
    
    format_choice = input("Output format (csv/excel/both) [both]: ").strip().lower()
    if not format_choice:
        format_choice = "both"
    
    # Save results
    print("\n5. Saving results...")
    if format_choice == "both":
        processor.save_results(output_dir, base_name)
    else:
        processor.save_results_custom(
            output_dir=output_dir,
            base_name=base_name,
            save_csv=(format_choice in ["csv", "both"]),
            save_excel=(format_choice in ["excel", "both"])
        )
    
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No command line arguments - run interactive mode
        interactive_mode()
    else:
        # Command line arguments provided
        sys.exit(main())