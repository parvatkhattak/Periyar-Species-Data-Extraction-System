#!/usr/bin/env python3
"""
Streamlit Interface for Periyar Species Extractor with OCR Support
================================================================

Web interface for processing PDFs and extracting species data with OCR for scanned documents.
"""

import streamlit as st
import os
import pandas as pd
import tempfile
from datetime import datetime
import io
import re
import logging
from typing import List, Dict, Any


from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# PDF processing libraries
import fitz  # PyMuPDF
from PIL import Image

# AI and OCR
import google.generativeai as genai
import base64

# Page configuration
st.set_page_config(
    page_title="Periyar Species Data Extractor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

class PeriyarSpeciesExtractor:
    """Enhanced Species Extractor with OCR capabilities"""
    
    def __init__(self):
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def setup_gemini(self, api_key: str):
        """Setup Gemini AI model"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            # Test the connection
            test_response = self.model.generate_content("Hello")
            self.logger.info("Gemini API configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup Gemini: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with fallback to OCR"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Try normal text extraction first
                page_text = page.get_text().strip()
                
                if page_text:
                    full_text += page_text + "\n\n"
                else:
                    # Fallback to OCR using Gemini Vision
                    self.logger.info(f"Using OCR for page {page_num + 1}")
                    ocr_text = self.extract_text_with_gemini_vision(page)
                    if ocr_text:
                        full_text += ocr_text + "\n\n"
            
            doc.close()
            return full_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_reference_info(self, text: str) -> tuple:
        """
        Extract author names and citation year from the document text
        
        Returns:
            tuple: (author_names, citation_year)
        """
        author_names = ""
        citation_year = ""
        
        # Common patterns for academic citations and references
        patterns = [
            # Pattern 1: "Author, A. & Author, B. (YYYY)"
            r'([A-Z][a-z]+(?:,\s*[A-Z]\.?\s*&?\s*[A-Z][a-z]+)*)\s*\((\d{4})\)',
            # Pattern 2: "Author et al. (YYYY)"
            r'([A-Z][a-z]+\s+et\s+al\.?)\s*\((\d{4})\)',
            # Pattern 3: "Author, A., Author, B., & Author, C. YYYY"
            r'([A-Z][a-z]+(?:,\s*[A-Z]\.?,?\s*(?:&\s*)?[A-Z][a-z]+)*)\s*(\d{4})',
            # Pattern 4: "By: Author Name" or "Authors: Author Name"
            r'(?:By|Authors?|Author\(s\)):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # Pattern 5: Look for year in brackets or parentheses
            r'\b(\d{4})\b',
        ]
        
        # Try to find author and year patterns
        text_sample = text[:2000]  # Search in first 2000 characters
        
        for i, pattern in enumerate(patterns[:4]):  # Skip the year-only pattern for now
            matches = re.findall(pattern, text_sample, re.IGNORECASE)
            if matches:
                match = matches[0]
                if isinstance(match, tuple) and len(match) == 2:
                    author_names = match[0].strip()
                    citation_year = match[1].strip()
                    break
                elif i == 3:  # "By: Author Name" pattern
                    author_names = match.strip()
        
        # If no author found, try to extract from document header/title area
        if not author_names:
            lines = text_sample.split('\n')[:10]  # First 10 lines
            for line in lines:
                # Look for lines that might contain author names
                if any(keyword in line.lower() for keyword in ['by', 'author', 'written']):
                    # Extract potential author name after the keyword
                    for keyword in ['by', 'author', 'authors', 'written by']:
                        if keyword in line.lower():
                            parts = line.lower().split(keyword)
                            if len(parts) > 1:
                                potential_author = parts[1].strip(' :,-').title()
                                if len(potential_author) > 3 and len(potential_author) < 100:
                                    author_names = potential_author
                                    break
        
        # Try to find year if not found yet
        if not citation_year:
            year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', text_sample)
            if year_matches:
                # Take the most recent year or the first one found
                citation_year = max(year_matches)
        
        # Clean up author names
        if author_names:
            # Remove common unwanted text
            author_names = re.sub(r'\s*\(.*?\)\s*', '', author_names)  # Remove parentheses
            author_names = re.sub(r'[^\w\s,&.]', '', author_names)     # Keep only letters, spaces, commas, &, and dots
            author_names = author_names.strip()
        
        return author_names, citation_year


    def extract_text_with_gemini_vision(self, page) -> str:
        """Extract text from PDF page using Gemini Vision API"""
        try:
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to base64 for Gemini
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Prepare image for Gemini
            image_part = {
                "mime_type": "image/png",
                "data": img_base64
            }
            
            prompt = """
            Extract all text from this image. This appears to be from a scientific document about species in Periyar.
            Please transcribe all visible text accurately, maintaining the original formatting as much as possible.
            Focus on species names, scientific names, locations, dates, and any other relevant information.
            """
            
            response = self.model.generate_content([prompt, image_part])
            return response.text if response.text else ""
            
        except Exception as e:
            self.logger.error(f"Gemini Vision OCR failed: {e}")
            return ""
    
    def extract_species_from_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Extract species information from text using Gemini AI"""
        if not text.strip():
            return []
        
        # Extract reference information from the document
        author_names, citation_year = self.extract_reference_info(text)
        
        # Create reference string
        if author_names and citation_year:
            reference_string = f"{author_names} ({citation_year})"
        elif author_names:
            reference_string = author_names
        elif citation_year:
            reference_string = f"Unknown Author ({citation_year})"
        else:
            # Fallback to filename if no reference info found
            reference_string = os.path.splitext(source_file)[0]
        
        prompt = f"""
        Analyze the following text from a scientific document about species in Periyar and extract all species information.

        For each species mentioned, extract:
        1. species_name (scientific name or common name)
        2. location_name (specific location within study area)
        3. latitude (decimal degrees format like 9.458333)
        4. longitude (decimal degrees format like 77.140000)
        5. sampling_period_from_month (start month name)
        6. sampling_period_from_year (start year)
        7. sampling_period_to_month (end month name)
        8. sampling_period_to_year (end year)
        9. sampling_season (Pre-Monsoon, Monsoon, Post-monsoon)
        10. order_family_species (taxonomic hierarchy in Order/Family/Species format)
        11. threat_status (conservation status with abbreviations like EN, DD, LR)
        12. relative_abundance (Very common, Common, Moderate, Rare, Very rare)
        13. endemism (endemism status with abbreviations like EN-K, EN-WG, EN-I, WD)
        14. flora_or_fauna (Flora or Fauna)
        15. reference (use this exact format: "{reference_string}")
        16. remarks (additional notes or information)

        Return the data as a JSON array of objects. If information is not available, use null.

        Text to analyze:
        {text[:8000]}
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            if response.text:
                # Try to extract JSON from response
                json_text = self.extract_json_from_response(response.text)
                if json_text:
                    import json
                    species_data = json.loads(json_text)
                    
                    # Ensure reference field uses the extracted reference string
                    if isinstance(species_data, list):
                        for species in species_data:
                            species['reference'] = reference_string
                        return species_data
                    return []
            
        except Exception as e:
            self.logger.error(f"Error extracting species data: {e}")
        
        return []
    
    def extract_json_from_response(self, text: str) -> str:
        """Extract JSON from AI response text"""
        # Look for JSON array in the response
        json_pattern = r'\[[\s\S]*\]'
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)
        
        # If no array found, look for JSON object and wrap it
        obj_pattern = r'\{[\s\S]*\}'
        match = re.search(obj_pattern, text)
        if match:
            return f"[{match.group(0)}]"
        
        return ""
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF file"""
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            self.logger.warning(f"No text extracted from {pdf_path}")
            return []
        
        # Extract species data
        source_file = os.path.basename(pdf_path)
        species_data = self.extract_species_from_text(text, source_file)
        
        self.logger.info(f"Extracted {len(species_data)} species from {source_file}")
        return species_data
    
    def process_pdf_with_filename(self, pdf_path: str, original_filename: str) -> List[Dict[str, Any]]:
        """Process a single PDF file with original filename for reference"""
        self.logger.info(f"Processing PDF: {original_filename}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            self.logger.warning(f"No text extracted from {original_filename}")
            return []
        
        # Extract species data using original filename
        species_data = self.extract_species_from_text(text, original_filename)
        
        # Ensure proper reference format
        filename_without_ext = os.path.splitext(original_filename)[0]
        for species in species_data:
            if 'reference' not in species or not species['reference']:
                species['reference'] = filename_without_ext
        
        self.logger.info(f"Extracted {len(species_data)} species from {original_filename}")
        return species_data
# Initialize session state
if 'extractor' not in st.session_state:
    st.session_state.extractor = None
if 'api_tested' not in st.session_state:
    st.session_state.api_tested = False
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []
if 'results' not in st.session_state:
    st.session_state.results = None

def log_message(message):
    """Add message to processing log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.processing_log.append(log_entry)

def clear_log():
    """Clear processing log"""
    st.session_state.processing_log = []

def test_api_connection(api_key):
    """Test the Gemini API connection"""
    try:
        with st.spinner("Testing API connection..."):
            extractor = PeriyarSpeciesExtractor()
            extractor.setup_gemini(api_key)
            st.session_state.extractor = extractor
            st.session_state.api_tested = True
            log_message("✓ Gemini API connection successful")
            return True
    except Exception as e:
        log_message(f"✗ API test failed: {e}")
        st.session_state.api_tested = False
        return False

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files"""
    if not st.session_state.extractor:
        st.error("Please test your API key first!")
        return None
    
    results = []
    total_files = len(uploaded_files)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    log_message(f"Starting processing of {total_files} files...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})")
            
            log_message(f"[{i+1}/{total_files}] Processing {uploaded_file.name}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process the PDF - pass the original filename instead of temp path
                species_data = st.session_state.extractor.process_pdf_with_filename(tmp_path, uploaded_file.name)
                results.extend(species_data)
                log_message(f"  ✓ Extracted {len(species_data)} species")
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            log_message(f"  ✗ Error processing {uploaded_file.name}: {e}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if results:
        log_message(f"Processing completed! Total species extracted: {len(results)}")
        return results
    else:
        log_message("No species data extracted from any files")
        return None
    
def create_download_files(results, filename_base, output_format):
    """Create download files based on format selection"""
    if not results:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=['species_name', 'reference'])
    
    # Reorder columns according to the required format
    column_order = [
        'species_name', 'location_name', 'latitude', 'longitude',
        'sampling_period_from_month', 'sampling_period_from_year',
        'sampling_period_to_month', 'sampling_period_to_year',
        'sampling_season', 'order_family_species', 'threat_status',
        'relative_abundance', 'endemism', 'flora_or_fauna', 'reference', 'remarks'
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder columns
    df = df[column_order]
    df = df.sort_values(['species_name', 'reference'])
    df = df.reset_index(drop=True)
    
    download_files = {}
    
    if output_format in ['csv', 'both']:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        download_files['csv'] = {
            'data': csv_buffer.getvalue(),
            'filename': f"{filename_base}.csv",
            'mime': 'text/csv'
        }
    
    if output_format in ['excel', 'both']:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Species_Data', index=False)
            
            # Add summary sheet with updated metrics
            summary_data = {
                'Metric': [
                    'Total Species',
                    'Flora Species',
                    'Fauna Species',
                    'Species with Coordinates',
                    'Species with Sampling Period',
                    'Species with Season Data',
                    'Species with Abundance Data',
                    'Species with Threat Status',
                    'Species with Endemism Data',
                    'Source Files'
                ],
                'Count': [
                    len(df),
                    len(df[df['flora_or_fauna'].str.contains('Flora', case=False, na=False)]),
                    len(df[df['flora_or_fauna'].str.contains('Fauna', case=False, na=False)]),
                    len(df[(df['latitude'].notna()) & (df['longitude'].notna()) & 
                          (df['latitude'] != '') & (df['longitude'] != '')]),
                    len(df[(df['sampling_period_from_month'].notna()) | 
                          (df['sampling_period_from_year'].notna()) |
                          (df['sampling_period_to_month'].notna()) |
                          (df['sampling_period_to_year'].notna())]),
                    len(df[df['sampling_season'].notna() & (df['sampling_season'] != '')]),
                    len(df[df['relative_abundance'].notna() & (df['relative_abundance'] != '')]),
                    len(df[df['threat_status'].notna() & (df['threat_status'] != '')]),
                    len(df[df['endemism'].notna() & (df['endemism'] != '')]),
                    df['reference'].nunique()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        download_files['excel'] = {
            'data': excel_buffer.getvalue(),
            'filename': f"{filename_base}.xlsx",
            'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
    
    return download_files, df

def show_results_summary(df):
    """Display results summary"""
    st.markdown("### 📊 Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Species", len(df))
    
    with col2:
        flora_count = len(df[df['flora_or_fauna'].str.contains('Flora', case=False, na=False)])
        st.metric("Flora Species", flora_count)
    
    with col3:
        fauna_count = len(df[df['flora_or_fauna'].str.contains('Fauna', case=False, na=False)])
        st.metric("Fauna Species", fauna_count)
    
    with col4:
        coord_count = len(df[(df['latitude'].notna()) & (df['longitude'].notna()) & 
                            (df['latitude'] != '') & (df['longitude'] != '')])
        st.metric("With Coordinates", coord_count)
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        sampling_count = len(df[(df['sampling_period_from_month'].notna()) | 
                               (df['sampling_period_from_year'].notna()) |
                               (df['sampling_period_to_month'].notna()) |
                               (df['sampling_period_to_year'].notna())])
        st.metric("With Sampling Period", sampling_count)

    with col6:
        abundance_count = len(df[df['relative_abundance'].notna() & (df['relative_abundance'] != '')])
        st.metric("With Abundance Data", abundance_count)

    with col7:
        threat_count = len(df[df['threat_status'].notna() & (df['threat_status'] != '')])
        st.metric("With Threat Status", threat_count)

    with col8:
        endemic_count = len(df[df['endemism'].notna() & (df['endemism'] != '')])
        st.metric("With Endemism Data", endemic_count)
    
    # Additional row for more metrics
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        season_count = len(df[df['sampling_season'].notna() & (df['sampling_season'] != '')])
        st.metric("With Season Data", season_count)
    
    with col10:
        taxonomy_count = len(df[df['order_family_species'].notna() & (df['order_family_species'] != '')])
        st.metric("With Taxonomy Data", taxonomy_count)
    
    with col11:
        location_count = len(df[df['location_name'].notna() & (df['location_name'] != '')])
        st.metric("With Location Data", location_count)
    
    with col12:
        source_count = df['reference'].nunique()
        st.metric("Source Files", source_count)

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Periyar Species Data Extractor</h1>
        <p>Extract species data from PDF documents using AI with OCR support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Auto-initialize API on first run
    if not st.session_state.api_tested and API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        with st.spinner("Initializing API connection..."):
            if test_api_connection(API_KEY):
                st.sidebar.success("🟢 API Ready")
            else:
                st.sidebar.error("❌ API connection failed!")
    elif API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        st.sidebar.error("⚠️ Please set your API key in the code")
    else:
        st.sidebar.success("🟢 API Ready")
    
    # Output options
    st.sidebar.subheader("📁 Output Options")
    
    output_format = st.sidebar.selectbox(
        "Output Format",
        ["both", "csv", "excel"],
        help="Choose the output file format"
    )
    
    filename_base = st.sidebar.text_input(
        "Output Filename",
        value="",
        help="Leave empty for automatic naming",
        label_visibility="visible"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload PDF Files")
        
        uploaded_files = st.file_uploader(
            "Select PDF files to process",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files containing species data (supports both text and scanned PDFs)",
            label_visibility="visible"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} PDF file(s) uploaded")
            
            # Show file list
            with st.expander("📋 Uploaded Files"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name} ({file.size:,} bytes)")
        
        # Process button
        if st.button("🚀 Process PDFs", type="primary", disabled=not st.session_state.api_tested):
            if not uploaded_files:
                st.warning("Please upload PDF files first!")
            else:
                with st.spinner("Processing PDFs with OCR support..."):
                    results = process_uploaded_files(uploaded_files)
                    st.session_state.results = results
                
                if results:
                    st.success(f"✅ Processing completed! Extracted {len(results)} species records.")
                else:
                    st.error("❌ No species data could be extracted from the uploaded files.")
    
    with col2:
        st.header("📊 Processing Log")
        
        # Log display
        log_container = st.container()
        with log_container:
            if st.session_state.processing_log:
                log_text = "\n".join(st.session_state.processing_log[-20:])  # Show last 20 entries
                st.text_area("Processing Log", value=log_text, height=300, disabled=True, label_visibility="collapsed")
            else:
                st.info("Processing log will appear here...")
        
        # Clear log button
        if st.button("🗑️ Clear Log"):
            clear_log()
            st.rerun()
    
    # Results section
    if st.session_state.results:
        st.header("📈 Results")
        
        # Generate filename
        if not filename_base:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"periyar_species_{timestamp}"
        
        # Create download files
        download_files, df = create_download_files(
            st.session_state.results, 
            filename_base, 
            output_format
        )
        
        # Show summary
        show_results_summary(df)
        
        # Download section
        st.markdown("### 💾 Download Results")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            if 'csv' in download_files:
                st.download_button(
                    label="📊 Download CSV",
                    data=download_files['csv']['data'],
                    file_name=download_files['csv']['filename'],
                    mime=download_files['csv']['mime']
                )
        
        with download_col2:
            if 'excel' in download_files:
                st.download_button(
                    label="📈 Download Excel",
                    data=download_files['excel']['data'],
                    file_name=download_files['excel']['filename'],
                    mime=download_files['excel']['mime']
                )
        
        # Data preview
        if st.checkbox("👀 Preview Data"):
            st.dataframe(df.head(100), use_container_width=True)
            if len(df) > 100:
                st.info(f"Showing first 100 rows of {len(df)} total records")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 Built with ❤️ using Streamlit & Google Gemini AI | "
        "Supports both text-based and scanned PDF documents with OCR"
    )

if __name__ == "__main__":
    main()
