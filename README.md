# Periyar Species Data Extractor

## 🌿 Project Overview

**Periyar Species Data Extractor** is an advanced AI-powered tool designed to extract comprehensive species data from PDF documents using OCR support and intelligent data processing. The application specifically focuses on biodiversity data from the Periyar Tiger Reserve region in Kerala, India, providing researchers and conservationists with an automated solution for digitizing species information from scientific documents.

## 🎯 Key Features

### Core Functionality
- **AI-Powered Text Extraction**: Uses Google's Gemini 2.0 Flash model for intelligent text processing
- **OCR Support**: Advanced Optical Character Recognition for scanned documents
- **Species Validation**: Comprehensive validation against GBIF and marine species databases
- **Coordinate Mapping**: Automatic GPS coordinate extraction and validation for Periyar region
- **Multi-Author Attribution**: Intelligent author identification and citation management
- **Data Deduplication**: Smart merging of duplicate species records
- **Export Options**: CSV and Excel output formats with detailed summaries

### Advanced Processing
- **Large PDF Handling**: Optimized processing for multi-page documents with progress tracking
- **Location Database**: Built-in Periyar location database with coordinates
- **Taxonomic Validation**: Order/Family/Species classification verification
- **Conservation Status**: IUCN threat status recognition and standardization
- **Endemic Species Detection**: Western Ghats endemism classification

## 🏗️ Architecture & State Flow

### Application State Flow Diagram

```
┌─────────────────┐
│   Application   │
│   Startup       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Initialize     │
│  Session State  │
│  - extractor    │
│  - api_tested   │
│  - results      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   API Key       │
│   Input &       │
│   Validation    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   Test API      │─────▶│   API Failed    │
│   Connection    │      │   Show Error    │
└─────────┬───────┘      └─────────────────┘
          │ ✓
          ▼
┌─────────────────┐
│   File Upload   │
│   Interface     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Processing    │
│   Settings      │
│   Configuration │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   PDF File(s)   │─────▶│   Single PDF    │
│   Analysis      │      │   Processing    │
└─────────┬───────┘      └─────────┬───────┘
          │                        │
          ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│   Multi-PDF     │      │   Text          │
│   Batch         │      │   Extraction    │
│   Processing    │      └─────────┬───────┘
└─────────┬───────┘                │
          │                        ▼
          └──────────────────────▶ ┌─────────────────┐
                                   │   Author        │
                                   │   Extraction    │
                                   └─────────┬───────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │   Coordinate    │
                                   │   Mapping       │
                                   └─────────┬───────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │   Species       │
                                   │   Extraction    │
                                   └─────────┬───────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │   Data          │
                                   │   Validation    │
                                   └─────────┬───────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │   Deduplication │
                                   │   & Merging     │
                                   └─────────┬───────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │   Results       │
                                   │   Display       │
                                   └─────────┬───────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │   Export        │
                                   │   Options       │
                                   └─────────────────┘
```

### Detailed Processing Flow

#### 1. **Initialization Phase**
```
┌─────────────────┐
│   App Start     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Load Session   │
│  State          │
│  - extractor    │
│  - api_tested   │
│  - logs         │
│  - results      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Configure      │
│  Streamlit UI   │
│  - Page config  │
│  - CSS styles   │
│  - Sidebar      │
└─────────────────┘
```

#### 2. **API Configuration Flow**
```
┌─────────────────┐
│   API Key       │
│   Input         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   Validate      │─────▶│   Invalid Key   │
│   Key Format    │      │   Show Error    │
└─────────┬───────┘      └─────────────────┘
          │ ✓
          ▼
┌─────────────────┐
│   Initialize    │
│   Gemini API    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   Test API      │─────▶│   Connection    │
│   Connection    │      │   Failed        │
└─────────┬───────┘      └─────────────────┘
          │ ✓
          ▼
┌─────────────────┐
│   API Ready     │
│   Enable UI     │
└─────────────────┘
```

#### 3. **File Processing Pipeline**
```
┌─────────────────┐
│   File Upload   │
│   Multiple PDFs │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   For Each PDF  │
│   Create Temp   │
│   File          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   Check PDF     │─────▶│   Large PDF     │
│   Size/Pages    │      │   (>3 pages)    │
└─────────┬───────┘      └─────────┬───────┘
          │                        │
          │ Small PDF              ▼
          ▼                ┌─────────────────┐
┌─────────────────┐        │   Enhanced      │
│   Standard      │        │   Processing    │
│   Processing    │        │   with Progress │
└─────────┬───────┘        └─────────┬───────┘
          │                          │
          └─────────┬────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Extract Full  │
          │   Document Text │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Extract       │
          │   Authors       │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Build         │
          │   Coordinate    │
          │   Mapping       │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Process       │
          │   Each Page     │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Merge         │
          │   Duplicates    │
          └─────────────────┘
```

#### 4. **Species Extraction Flow**
```
┌─────────────────┐
│   Page Text     │
│   Input         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Gemini AI     │
│   Processing    │
│   with Enhanced │
│   Prompt        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   Extract JSON  │─────▶│   Parse Error   │
│   from Response │      │   Retry Logic   │
└─────────┬───────┘      └─────────────────┘
          │ ✓
          ▼
┌─────────────────┐
│   For Each      │
│   Species       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Clean &       │
│   Validate Data │
│   - Names       │
│   - Coordinates │
│   - Dates       │
│   - Status      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Author        │
│   Attribution   │
│   Analysis      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Coordinate    │
│   Enrichment    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Enhanced      │
│   Remarks       │
│   Generation    │
└─────────────────┘
```

#### 5. **Validation Pipeline**
```
┌─────────────────┐
│   Species List  │
│   Input         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   For Each      │
│   Species       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   Name          │─────▶│   GBIF API      │
│   Validation    │      │   Query         │
└─────────┬───────┘      └─────────┬───────┘
          │                        │
          ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│   Taxonomy      │      │   Cache Result  │
│   Validation    │      └─────────────────┘
└─────────┬───────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│   Geographic    │─────▶│   Occurrence    │
│   Validation    │      │   Check India   │
└─────────┬───────┘      └─────────────────┘
          │
          ▼
┌─────────────────┐
│   Periyar       │
│   Database      │
│   Check         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Calculate     │
│   Confidence    │
│   Score         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Generate      │
│   Validation    │
│   Notes         │
└─────────────────┘
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **Streamlit**: Web application framework
- **Google Generative AI (Gemini 2.0)**: AI text processing
- **PyMuPDF (fitz)**: PDF text extraction
- **PIL (Pillow)**: Image processing for OCR

### Data Processing Libraries
- **pandas**: Data manipulation and analysis
- **openpyxl**: Excel file generation
- **fuzzywuzzy**: Fuzzy string matching
- **regex (re)**: Pattern matching and text processing

### External APIs
- **GBIF API**: Species validation and taxonomic data
- **WORMS API**: Marine species validation
- **Google Gemini API**: AI-powered text extraction

### Additional Dependencies
- **python-dotenv**: Environment variable management
- **requests**: HTTP API calls
- **base64**: Image encoding for OCR
- **tempfile**: Temporary file handling
- **logging**: Application logging

## 📁 Project Structure

```
periyar-species-extractor/
│
├── streamlit_periyar_gui.py          # Main application file
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (API keys)
├── README.md                         # This file

```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Cloud API access with Gemini API enabled
- Stable internet connection for API calls

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/periyar-species-extractor.git
   cd periyar-species-extractor
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

5. **Run the Application**
   ```bash
   streamlit run streamlit_periyar_gui.py
   ```

### Requirements.txt
```txt
streamlit>=1.28.0
google-generativeai>=0.3.0
PyMuPDF>=1.23.0
Pillow>=10.0.0
pandas>=2.0.0
openpyxl>=3.1.0
python-dotenv>=1.0.0
requests>=2.31.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.20.0
```

## 🎛️ Configuration Options

### Processing Settings
- **Delay Between Pages**: 1.0-10.0 seconds (default: 3.0)
- **Max Retries**: 2-7 attempts (default: 3)
- **Extraction Mode**: Standard/Detailed/Maximum
- **Coordinate Validation**: Enable/Disable Periyar region validation
- **Name Cleaning**: Enable/Disable scientific name standardization

### Validation Settings
- **Species Validation**: Enable/Disable external database validation
- **Confidence Threshold**: 0.0-1.0 (default: 0.5)
- **Show Invalid Species**: Include/Exclude failed validations

### Output Settings
- **Export Format**: CSV, Excel, or Both
- **Include Validation Data**: Add validation columns to output
- **Summary Sheet**: Include processing summary in Excel files

## 📊 Data Schema

### Core Species Data Structure
```json
{
  "species_name": "Exact species name from document",
  "location_name": "Specific location within Periyar",
  "latitude": 9.458333,
  "longitude": 77.140000,
  "sampling_period_from_month": "January",
  "sampling_period_from_year": 2020,
  "sampling_period_to_month": "December", 
  "sampling_period_to_year": 2020,
  "sampling_season": "Pre-Monsoon",
  "order_family_species": "Primates/Cercopithecidae/Macaca radiata",
  "threat_status": "EN",
  "relative_abundance": "Common",
  "endemism": "EN-WG",
  "flora_or_fauna": "Fauna",
  "reference": "Author citation",
  "remarks": "Additional notes and context"
}
```

### Validation Data Structure
```json
{
  "is_valid": true,
  "confidence_score": 0.85,
  "corrected_name": "Standardized species name",
  "validation_details": {
    "name_validation": {},
    "taxonomic_validation": {},
    "geographic_validation": {},
    "format_validation": {},
    "periyar_validation": {}
  },
  "validation_notes": "Human-readable validation summary"
}
```

## 🔍 Core Components Deep Dive

### 1. PeriyarSpeciesExtractor Class

**Primary Functions:**
- `setup_gemini()`: Initialize Gemini AI model
- `process_large_pdf()`: Handle multi-page documents with progress tracking
- `extract_species_from_page()`: Process individual pages with AI
- `extract_location_coordinate_mapping()`: Build coordinate database from document
- `merge_duplicate_species()`: Intelligent deduplication and merging

**Key Features:**
- Multi-author attribution system
- Coordinate enrichment from document context
- Smart text cleaning and validation
- Retry logic for API failures

### 2. SpeciesValidator Class

**Validation Pipeline:**
- **Name Validation**: GBIF database verification
- **Taxonomic Validation**: Order/Family/Species structure check
- **Geographic Validation**: India/Periyar occurrence verification
- **Format Validation**: Scientific name format compliance
- **Periyar Validation**: Local species database check

**Confidence Scoring:**
- Weighted validation results (30% name, 25% geographic, 20% taxonomic, 15% format, 10% Periyar)
- 0.0-1.0 confidence scale
- Configurable threshold filtering

### 3. PeriyarLocationDatabase Class

**Built-in Locations:**
- Major landmarks (Periyar Lake, Thekkady, Kumily)
- Trails and routes (Nature Trail, Bamboo Rafting)
- Forest compartments (Thannikudy, Manakkavala)
- Research stations and accommodations

**Coordinate Features:**
- Fuzzy location matching
- Elevation data integration
- Bounds validation for Periyar region (9.0-10.0°N, 76.5-77.5°E)

## 🔄 Processing Workflows

### Single PDF Processing
1. **Text Extraction**: PyMuPDF + OCR fallback
2. **Author Detection**: Pattern matching for citations
3. **Coordinate Mapping**: GPS data extraction from full document
4. **Species Extraction**: AI-powered information extraction
5. **Data Cleaning**: Standardization and validation
6. **Deduplication**: Smart merging of duplicate records

### Batch PDF Processing
1. **File Queue Management**: Sequential processing with progress tracking
2. **Memory Optimization**: Temporary file handling
3. **Error Recovery**: Per-file error handling with continuation
4. **Results Aggregation**: Cross-document deduplication
5. **Validation Pipeline**: Batch species validation
6. **Export Generation**: Multi-format output creation

### Validation Workflow
1. **Input Preprocessing**: Species list preparation
2. **External API Calls**: GBIF and WORMS database queries
3. **Local Database Check**: Periyar species reference matching
4. **Confidence Calculation**: Multi-factor scoring algorithm
5. **Results Filtering**: Threshold-based inclusion/exclusion
6. **Note Generation**: Human-readable validation summaries

## 🎯 Use Cases & Applications

### Research Applications
- **Biodiversity Surveys**: Digitize field survey reports
- **Literature Reviews**: Extract species data from research papers
- **Meta-Analysis**: Aggregate species information across studies
- **Database Population**: Automated data entry for biodiversity databases

### Conservation Applications
- **Species Monitoring**: Track species occurrence over time
- **Protected Area Management**: Inventory species within reserves
- **Threat Assessment**: Compile conservation status information
- **Endemic Species Tracking**: Identify and catalog endemic flora/fauna

### Academic Applications
- **Thesis Data Extraction**: Process research document repositories
- **Curriculum Development**: Create species reference materials
- **Student Projects**: Provide datasets for analysis
- **Publication Support**: Standardize species reporting

## 🚨 Error Handling & Troubleshooting

### Common Issues

#### API Connection Problems
- **Symptom**: API test fails
- **Solution**: Verify API key, check internet connection
- **Prevention**: Use environment variables, implement retry logic

#### PDF Processing Errors
- **Symptom**: No text extracted from PDF
- **Solution**: Enable OCR mode, check PDF quality
- **Prevention**: Pre-process scanned documents

#### Memory Issues
- **Symptom**: Application crashes with large PDFs
- **Solution**: Increase delay between pages, reduce batch size
- **Prevention**: Implement pagination and cleanup

#### Validation Failures
- **Symptom**: All species marked as invalid
- **Solution**: Lower confidence threshold, check internet connectivity
- **Prevention**: Cache validation results, implement offline fallback

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export DEBUG=True
```

## 📈 Performance Optimization

### Processing Speed
- **Page Delays**: Balance extraction quality vs. speed
- **API Caching**: Reduce redundant validation calls
- **Batch Processing**: Optimize memory usage for large files
- **Retry Logic**: Smart exponential backoff

### Memory Management
- **Temporary Files**: Automatic cleanup after processing
- **Session State**: Efficient data structure management
- **Image Processing**: Optimized resolution for OCR
- **Data Streaming**: Process results incrementally

### Accuracy Improvements
- **Enhanced Prompts**: Detailed AI instructions for better extraction
- **Validation Layers**: Multiple verification systems
- **Context Analysis**: Surrounding text for better attribution
- **Coordinate Validation**: Geographic bounds checking

## 🔒 Security & Privacy

### API Key Management
- Environment variable storage
- No hardcoded credentials
- Secure session handling

### Data Privacy
- Temporary file processing
- No permanent data storage
- Local processing priority

### Input Validation
- File type restrictions
- Size limitations
- Content sanitization

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest tests/`
5. Submit pull request

### Code Standards
- PEP 8 compliance
- Type hints for public methods
- Comprehensive docstrings
- Unit test coverage >80%

### Feature Requests
- Use GitHub Issues template
- Provide detailed use case description
- Include sample data if applicable


## 🙏 Acknowledgments

- **Periyar Tiger Reserve** for conservation inspiration
- **Google Gemini AI** for advanced text processing capabilities
- **GBIF** for species validation services
- **Streamlit** for the web application framework
- **Open Source Community** for supporting libraries
