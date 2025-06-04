#!/usr/bin/env python3
"""
Periyar Species Data Extraction System
====================================

This system extracts species data from PDF files and converts them into structured CSV/Excel format
using Gemini 2.0 Flash for AI-powered species identification and data extraction.

Requirements:
- PyPDF2 or pdfplumber for PDF text extraction
- google-generativeai for Gemini API
- pandas for data manipulation
- openpyxl for Excel output
- geopy for geocoding (optional)
"""

import os
import re
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging


import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

def extract_text_with_ocr(self, pdf_path):
    """
    Extract text from PDF using OCR for scanned documents
    """
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        text_content = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # First try normal text extraction
            page_text = page.get_text()
            
            if page_text.strip():
                # If text exists, use it
                text_content += page_text + "\n"
            else:
                # If no text, use OCR on page image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                ocr_text = pytesseract.image_to_string(img)
                text_content += ocr_text + "\n"
        
        doc.close()
        return text_content
        
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""
    

# PDF processing libraries
try:
    import pdfplumber
    PDF_LIB = 'pdfplumber'
except ImportError:
    try:
        import PyPDF2
        PDF_LIB = 'PyPDF2'
    except ImportError:
        print("Please install either pdfplumber or PyPDF2: pip install pdfplumber")
        exit(1)

# Google Gemini API
try:
    import google.generativeai as genai
except ImportError:
    print("Please install Google Generative AI: pip install google-generativeai")
    exit(1)

# Optional: Geopy for location enhancement
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("Geopy not available. Install with: pip install geopy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PeriyarSpeciesExtractor:
    """Main class for extracting species data from PDFs"""
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the extractor
        
        Args:
            gemini_api_key: Google Gemini API key (leave blank to input later)
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        self.geocoder = Nominatim(user_agent="periyar_species_extractor") if GEOPY_AVAILABLE else None
        
        # Species classification patterns
        self.flora_indicators = [
            'plant', 'tree', 'shrub', 'herb', 'fern', 'moss', 'algae', 'grass',
            'orchid', 'flower', 'leaf', 'seed', 'fruit', 'root', 'stem'
        ]
        
        self.fauna_indicators = [
            'bird', 'fish', 'mammal', 'reptile', 'amphibian', 'insect', 'spider',
            'butterfly', 'moth', 'beetle', 'ant', 'bee', 'fly', 'mosquito',
            'snake', 'lizard', 'frog', 'toad', 'turtle', 'crocodile',
            'tiger', 'elephant', 'deer', 'monkey', 'bat', 'rodent'
        ]
        
    def setup_gemini(self, api_key: str = None):
        """Setup Gemini API"""
        if api_key:
            self.gemini_api_key = api_key
        elif not self.gemini_api_key:
            self.gemini_api_key = input("Please enter your Gemini API key: ").strip()
        
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            if PDF_LIB == 'pdfplumber':
                return self._extract_with_pdfplumber(pdf_path)
            else:
                return self._extract_with_pypdf2(pdf_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_species_with_gemini(self, text: str, pdf_filename: str) -> List[Dict]:
        """Use Gemini to extract species information from text"""
        if not self.gemini_model:
            raise Exception("Gemini model not initialized. Call setup_gemini() first.")
        
        prompt = f"""
        Please analyze the following scientific text and extract all species information. 
        For each species found, provide the information in JSON format with these fields:
        
        - species_name: Scientific name (genus species) or common name if scientific name not available
        - flora_or_fauna: "Flora" or "Fauna" (classify based on the species type)
        - date_of_observation: Extract any dates mentioned (format as YYYY-MM-DD or YYYY-MM or YYYY, or date range)
        - location_name: Any specific location names mentioned
        - latitude: Extract latitude if mentioned (decimal degrees)
        - longitude: Extract longitude if mentioned (decimal degrees)
        - additional_info: Any other relevant information about the species
        
        # Text to analyze:
        # {text[:4000]}  # Limit text to avoid token limits
        
        Please return ONLY a valid JSON array of species objects. Do not include any other text or explanations.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                species_data = json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire response as JSON
                species_data = json.loads(response_text)
            
            # Add reference information (filename without extension)
            filename_without_ext = os.path.splitext(pdf_filename)[0]
            for species in species_data:
                species['reference'] = filename_without_ext
                
            return species_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
            logger.error(f"Raw response: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return []
        
    def enhance_location_data(self, location_name: str) -> Tuple[Optional[float], Optional[float]]:
        """Enhance location data using geocoding"""
        if not self.geocoder or not location_name:
            return None, None
        
        try:
            # Add "Kerala, India" to improve geocoding accuracy
            search_query = f"{location_name}, Kerala, India"
            location = self.geocoder.geocode(search_query)
            
            if location:
                return location.latitude, location.longitude
            else:
                # Try without Kerala, India
                location = self.geocoder.geocode(location_name)
                if location:
                    return location.latitude, location.longitude
                    
        except Exception as e:
            logger.warning(f"Geocoding failed for {location_name}: {e}")
        
        return None, None
    
    def classify_species_type(self, species_name: str, additional_info: str = "") -> str:
        """Classify species as Flora or Fauna based on name and context"""
        text_to_analyze = f"{species_name} {additional_info}".lower()
        
        # Check for fauna indicators
        for indicator in self.fauna_indicators:
            if indicator in text_to_analyze:
                return "Fauna"
        
        # Check for flora indicators
        for indicator in self.flora_indicators:
            if indicator in text_to_analyze:
                return "Flora"
        
        # Default classification based on common patterns
        if any(word in species_name.lower() for word in ['bird', 'fish', 'mammal']):
            return "Fauna"
        elif any(word in species_name.lower() for word in ['tree', 'plant', 'flower']):
            return "Flora"
        
        return "Unknown"
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a single PDF file and extract species data"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return []
        
        # Extract species data using Gemini
        pdf_filename = os.path.basename(pdf_path)
        species_data = self.extract_species_with_gemini(text, pdf_filename)
        
        # Enhance and validate data
        enhanced_data = []
        filename_without_ext = os.path.splitext(pdf_filename)[0]
        
        for species in species_data:
            # Ensure required fields exist
            enhanced_species = {
                'species_name': species.get('species_name', ''),
                'flora_or_fauna': species.get('flora_or_fauna', ''),
                'date_of_observation': species.get('date_of_observation', ''),
                'location_name': species.get('location_name', ''),
                'latitude': species.get('latitude', ''),
                'longitude': species.get('longitude', ''),
                'reference': filename_without_ext,  # Use filename without extension
                'additional_info': species.get('additional_info', '')
            }
            
            # Auto-classify if not classified
            if not enhanced_species['flora_or_fauna'] or enhanced_species['flora_or_fauna'] == 'Unknown':
                enhanced_species['flora_or_fauna'] = self.classify_species_type(
                    enhanced_species['species_name'], 
                    enhanced_species.get('additional_info', '')
                )
            
            # Enhance location data if needed
            if enhanced_species['location_name'] and not enhanced_species['latitude']:
                lat, lon = self.enhance_location_data(enhanced_species['location_name'])
                if lat and lon:
                    enhanced_species['latitude'] = lat
                    enhanced_species['longitude'] = lon
            
            enhanced_data.append(enhanced_species)
        
        logger.info(f"Extracted {len(enhanced_data)} species from {filename_without_ext}")
        return enhanced_data
        
    def process_multiple_pdfs(self, pdf_directory: str) -> pd.DataFrame:
        """Process multiple PDF files in a directory"""
        all_species_data = []
        
        # Get all PDF files in directory
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            species_data = self.process_pdf(pdf_path)
            all_species_data.extend(species_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_species_data)
        
        # Clean and format the data
        if not df.empty:
            # Remove duplicates based on species name and reference
            df = df.drop_duplicates(subset=['species_name', 'reference'])
            
            # Sort by species name
            df = df.sort_values('species_name')
            
            # Reset index
            df = df.reset_index(drop=True)
        
        return df
    
    def save_to_excel(self, df: pd.DataFrame, output_path: str):
        """Save DataFrame to Excel file"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Species_Data', index=False)
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Species_Data']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"Data saved to Excel: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save Excel file: {e}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """Save DataFrame to CSV file"""
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Data saved to CSV: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV file: {e}")
            raise

def main():
    """Main function to demonstrate usage"""
    print("Periyar Species Data Extraction System")
    print("=====================================")
    
    # Initialize extractor
    extractor = PeriyarSpeciesExtractor()
    
    # Setup Gemini API
    try:
        extractor.setup_gemini()
    except Exception as e:
        print(f"Failed to setup Gemini API: {e}")
        return
    
    # Get input directory
    pdf_directory = input("Enter the path to the directory containing PDF files: ").strip()
    if not os.path.exists(pdf_directory):
        print(f"Directory not found: {pdf_directory}")
        return
    
    # Process PDFs
    try:
        print("Processing PDF files...")
        df = extractor.process_multiple_pdfs(pdf_directory)
        
        if df.empty:
            print("No species data extracted from PDF files.")
            return
        
        print(f"Successfully extracted data for {len(df)} species")
        
        # Get output preferences
        output_format = input("Choose output format (csv/excel/both): ").strip().lower()
        output_name = input("Enter output filename (without extension): ").strip()
        
        if not output_name:
            output_name = f"periyar_species_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save files
        if output_format in ['csv', 'both']:
            csv_path = f"{output_name}.csv"
            extractor.save_to_csv(df, csv_path)
        
        if output_format in ['excel', 'both']:
            excel_path = f"{output_name}.xlsx"
            extractor.save_to_excel(df, excel_path)
        
        print("Processing completed successfully!")
        
        # Display summary
        print(f"\nSummary:")
        print(f"Total species extracted: {len(df)}")
        print(f"Flora species: {len(df[df['flora_or_fauna'] == 'Flora'])}")
        print(f"Fauna species: {len(df[df['flora_or_fauna'] == 'Fauna'])}")
        print(f"Species with location data: {len(df[(df['latitude'] != '') & (df['longitude'] != '')])}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        logger.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()