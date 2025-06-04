# Periyar Species Data Extraction System

🌿 An AI-powered system for extracting species data from PDF documents using Google Gemini AI with OCR support for scanned documents.

## Overview

This system processes PDF files containing species information from Periyar and converts them into structured CSV/Excel format. It uses Google Gemini 2.0 Flash for intelligent species identification and data extraction, with support for both text-based and scanned PDFs.

## Features

- **AI-Powered Extraction**: Uses Google Gemini AI for intelligent species identification
- **OCR Support**: Handles scanned PDFs using Gemini Vision API
- **Multiple Interfaces**: 
  - Command-line batch processor
  - Interactive terminal mode
  - Web-based Streamlit GUI
- **Flexible Output**: CSV and Excel formats with customizable naming
- **Data Enhancement**: Automatic species classification and geocoding support
- **Batch Processing**: Process multiple PDFs simultaneously
- **Data Validation**: Duplicate removal and data cleaning

## Installation

### Prerequisites

- Python 3.7 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Install Required Packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas openpyxl pdfplumber google-generativeai streamlit pymupdf pillow geopy
```

### Optional Dependencies

For enhanced OCR support:
```bash
pip install pytesseract
```

## Project Structure

```
periyar-species-extractor/
├── periyar_extractor.py          # Core extraction engine
├── batch_processor.py            # Batch processing for multiple PDFs
├── streamlit_periyar_gui.py      # Web GUI interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

### 1. Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run streamlit_periyar_gui.py
```

Features:
- Drag-and-drop PDF upload
- Real-time processing log
- Data preview and download
- OCR support for scanned documents

### 2. Batch Processing (Command Line)

Process multiple PDFs from command line:

```bash
python batch_processor.py /path/to/pdf/directory
```

**Options:**
```bash
python batch_processor.py [INPUT_PATH] [OPTIONS]

Options:
  -o, --output DIR        Output directory (default: output)
  -n, --name NAME         Base name for output files
  -k, --api-key KEY       Gemini API key
  --csv-only             Save only CSV format
  --excel-only           Save only Excel format
```

**Examples:**
```bash
# Process all PDFs in a directory
python batch_processor.py ./pdfs/ -o ./results/

# Process single PDF with custom name
python batch_processor.py species_report.pdf -n "periyar_analysis"

# Save only CSV format
python batch_processor.py ./pdfs/ --csv-only
```

### 3. Interactive Mode

Run without arguments for interactive setup:

```bash
python batch_processor.py
```

### 4. Direct Python Usage

```python
from periyar_extractor import PeriyarSpeciesExtractor

# Initialize extractor
extractor = PeriyarSpeciesExtractor()
extractor.setup_gemini("your-api-key")

# Process single PDF
species_data = extractor.process_pdf("species_report.pdf")

# Process multiple PDFs
df = extractor.process_multiple_pdfs("./pdf_directory/")

# Save results
extractor.save_to_csv(df, "results.csv")
extractor.save_to_excel(df, "results.xlsx")
```

## Configuration

### API Key Setup

Set your Gemini API key in one of these ways:

1. **Environment Variable:**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

2. **Command Line:**
   ```bash
   python batch_processor.py --api-key "your-api-key-here" ./pdfs/
   ```

3. **Interactive Input:** The system will prompt for API key if not provided

4. **Web Interface:** Modify the API_KEY variable in `streamlit_periyar_gui.py`

## Output Format

The system extracts the following information for each species:

| Field | Description |
|-------|-------------|
| `species_name` | Common name of the species |
| `scientific_name` | Latin/binomial scientific name |
| `flora_or_fauna` | Classification as Flora or Fauna |
| `family` | Taxonomic family |
| `habitat` | Habitat description |
| `location_name` | Specific location within Periyar |
| `latitude` | GPS latitude (if available) |
| `longitude` | GPS longitude (if available) |
| `date_of_observation` | Date of observation |
| `reference` | Source PDF filename |
| `additional_info` | Other relevant notes |

## Advanced Features

### OCR Support

The system automatically detects scanned PDFs and uses OCR:
- Primary: Gemini Vision API for intelligent text extraction
- Fallback: Tesseract OCR (if installed)
- Hybrid approach: Combines normal text extraction with OCR

### Data Enhancement

- **Automatic Classification**: Species are classified as Flora/Fauna based on context
- **Geocoding**: Location names are converted to coordinates (requires geopy)
- **Duplicate Removal**: Automatic deduplication based on species name and source
- **Data Validation**: Ensures data quality and consistency

### Batch Processing Features

- **Progress Tracking**: Real-time progress updates
- **Error Handling**: Continues processing even if individual files fail
- **Detailed Logging**: Comprehensive processing logs
- **Resume Capability**: Can handle large batches efficiently

## Troubleshooting

### Common Issues

**1. API Key Error**
```
Error: Failed to configure Gemini API
```
- Verify your API key is correct
- Check internet connection
- Ensure API key has proper permissions

**2. PDF Processing Error**
```
Error: No text extracted from PDF
```
- PDF might be image-based (OCR will be attempted)
- File might be corrupted
- Try with different PDF files

**3. Import Errors**
```
ImportError: No module named 'pdfplumber'
```
- Install missing dependencies: `pip install -r requirements.txt`

**4. Empty Results**
```
No species data extracted
```
- Check if PDF contains species information
- Try with different PDF files
- Verify API key is working

### Performance Tips

- **Large PDFs**: Process in smaller batches for better performance
- **OCR Processing**: Scanned PDFs take longer to process
- **API Limits**: Be aware of Gemini API rate limits
- **Memory Usage**: Close unnecessary applications when processing large batches

## Dependencies

### Core Dependencies
- `pandas` - Data manipulation and analysis
- `openpyxl` - Excel file handling
- `pdfplumber` - PDF text extraction
- `google-generativeai` - Gemini AI API
- `streamlit` - Web interface
- `pymupdf` - PDF processing with OCR support
- `pillow` - Image processing

### Optional Dependencies
- `geopy` - Geocoding support
- `pytesseract` - Alternative OCR engine

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Open an issue on GitHub

## Changelog

### Version 1.0.0
- Initial release with basic PDF processing
- Gemini AI integration
- Batch processing support

### Version 1.1.0
- Added OCR support for scanned PDFs
- Streamlit web interface
- Enhanced error handling
- Improved data validation

### Version 1.2.0
- Gemini Vision API integration
- Interactive processing mode
- Advanced geocoding features
- Performance optimizations

---

**Made with ❤️ for Periyar species research**
