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
import requests
from fuzzywuzzy import fuzz
import json
import time
from typing import Dict, List, Tuple, Optional
import logging
from typing import List, Dict, Any
import time
import asyncio

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

# In your sidebar - REPLACE existing settings section
st.sidebar.markdown("### ⚙️ Processing Settings")
delay_between_pages = st.sidebar.slider(
    "Delay between pages (seconds)", 
    min_value=1.0, 
    max_value=10.0, 
    value=3.0,  # Increased default for better extraction
    step=0.5,
    help="Longer delays allow more thorough extraction"
)

max_retries = st.sidebar.selectbox(
    "Max retries per page",
    options=[2, 3, 5, 7],
    index=1,  # Default to 3 retries
    help="More retries improve extraction completeness"
)

# ADD NEW EXTRACTION SETTINGS
st.sidebar.markdown("### 🎯 Extraction Settings")

extraction_mode = st.sidebar.radio(
    "Extraction thoroughness",
    ["Standard", "Detailed", "Maximum"],
    index=1,  # Default to Detailed
    help="Higher settings extract more details but take longer"
)

validate_coordinates = st.sidebar.checkbox(
    "Validate coordinates for Periyar region",
    value=True,
    help="Only accept coordinates within Periyar bounds"
)

clean_scientific_names = st.sidebar.checkbox(
    "Clean and standardize scientific names", 
    value=True,
    help="Remove formatting issues from species names"
)

# Add to sidebar settings
st.sidebar.markdown("### 🔍 Validation Settings")
enable_validation = st.sidebar.checkbox(
    "Enable species validation",
    value=True,
    help="Validate extracted species against scientific databases"
)

confidence_threshold = st.sidebar.slider(
    "Validation confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Minimum confidence score for species inclusion"
)

show_invalid_species = st.sidebar.checkbox(
    "Show invalid species in results",
    value=False,
    help="Include species that failed validation in output"
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



class SpeciesValidator:
    """Comprehensive species validation system"""
    
    def __init__(self):
        self.gbif_base_url = "https://api.gbif.org/v1"
        self.worms_base_url = "https://www.marinespecies.org/rest"
        self.cache = {}  # Cache for API responses
        self.periyar_species_list = self.load_periyar_species_reference()
        
    def load_periyar_species_reference(self) -> Dict[str, Dict]:
        """Load known Periyar species reference data"""
        # This would ideally load from a curated database
        # For now, we'll use a basic structure that can be expanded
        return {
            # Flora examples
            "shorea_robusta": {
                "scientific_name": "Shorea robusta",
                "common_names": ["Sal", "Sakhua"],
                "family": "Dipterocarpaceae",
                "type": "Flora",
                "status": "verified",
                "periyar_occurrence": True
            },
            "tectona_grandis": {
                "scientific_name": "Tectona grandis",
                "common_names": ["Teak"],
                "family": "Lamiaceae",
                "type": "Flora",
                "status": "verified",
                "periyar_occurrence": True
            },
            # Fauna examples
            "panthera_tigris": {
                "scientific_name": "Panthera tigris",
                "common_names": ["Tiger", "Royal Bengal Tiger"],
                "family": "Felidae",
                "type": "Fauna",
                "status": "verified",
                "periyar_occurrence": True
            },
            "elephas_maximus": {
                "scientific_name": "Elephas maximus",
                "common_names": ["Asian Elephant", "Indian Elephant"],
                "family": "Elephantidae",
                "type": "Fauna",
                "status": "verified",
                "periyar_occurrence": True
            }
        }
    
    def validate_species_batch(self, species_list: List[Dict]) -> List[Dict]:
        """Validate a batch of species with comprehensive checks"""
        validated_species = []
        
        for i, species in enumerate(species_list):
            print(f"Validating species {i+1}/{len(species_list)}: {species.get('species_name', 'Unknown')}")
            
            # Perform validation
            validation_result = self.validate_single_species(species)
            
            # Add validation results to species record
            species_with_validation = species.copy()
            species_with_validation.update(validation_result)
            
            validated_species.append(species_with_validation)
            
            # Rate limiting for API calls
            time.sleep(0.5)
        
        return validated_species
    
    def validate_single_species(self, species: Dict) -> Dict:
        """Comprehensive validation for a single species"""
        species_name = species.get('species_name', '').strip()
        if not species_name:
            return self.create_validation_result(False, "No species name provided")
        
        validation_results = {
            'name_validation': self.validate_species_name(species_name),
            'taxonomic_validation': self.validate_taxonomy(species_name, species.get('order_family_species')),
            'geographic_validation': self.validate_geographic_occurrence(species_name, species.get('location_name')),
            'format_validation': self.validate_name_format(species_name),
            'periyar_validation': self.validate_periyar_occurrence(species_name)
        }
        
        # Calculate overall confidence score
        confidence_score = self.calculate_confidence_score(validation_results)
        
        # Determine overall validation status
        is_valid = confidence_score >= 0.7  # 70% threshold
        
        return {
            'is_valid': is_valid,
            'confidence_score': confidence_score,
            'validation_details': validation_results,
            'corrected_name': validation_results['name_validation'].get('corrected_name'),
            'validation_notes': self.generate_validation_notes(validation_results)
        }
    
    def validate_species_name(self, species_name: str) -> Dict:
        """Validate species name against GBIF database"""
        try:
            # Check cache first
            cache_key = f"gbif_{species_name.lower()}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Search GBIF species API
            search_url = f"{self.gbif_base_url}/species/match"
            params = {
                'name': species_name,
                'verbose': True
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    'status': 'found' if data.get('matchType') != 'NONE' else 'not_found',
                    'match_type': data.get('matchType'),
                    'confidence': data.get('confidence', 0),
                    'corrected_name': data.get('scientificName'),
                    'kingdom': data.get('kingdom'),
                    'phylum': data.get('phylum'),
                    'class': data.get('class'),
                    'order': data.get('order'),
                    'family': data.get('family'),
                    'genus': data.get('genus'),
                    'species': data.get('species'),
                    'taxonomic_status': data.get('taxonomicStatus'),
                    'synonym': data.get('synonym', False)
                }
                
                # Cache the result
                self.cache[cache_key] = result
                return result
            else:
                return {'status': 'api_error', 'error': f"API returned {response.status_code}"}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def validate_taxonomy(self, species_name: str, taxonomy_string: str) -> Dict:
        """Validate taxonomic classification"""
        try:
            if not taxonomy_string:
                return {'status': 'missing', 'message': 'No taxonomic information provided'}
            
            # Parse taxonomy string (expected format: Order/Family/Species)
            parts = taxonomy_string.split('/')
            if len(parts) != 3:
                return {'status': 'invalid_format', 'message': 'Taxonomy should be in Order/Family/Species format'}
            
            order, family, species = [part.strip() for part in parts]
            
            # Validate against GBIF data if available
            gbif_data = self.validate_species_name(species_name)
            if gbif_data.get('status') == 'found':
                gbif_family = gbif_data.get('family', '').lower()
                gbif_order = gbif_data.get('order', '').lower()
                
                family_match = gbif_family == family.lower() if gbif_family else False
                order_match = gbif_order == order.lower() if gbif_order else False
                
                return {
                    'status': 'validated',
                    'family_match': family_match,
                    'order_match': order_match,
                    'gbif_order': gbif_data.get('order'),
                    'gbif_family': gbif_data.get('family'),
                    'provided_order': order,
                    'provided_family': family
                }
            
            return {'status': 'no_reference', 'message': 'Could not validate against reference database'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def validate_geographic_occurrence(self, species_name: str, location: str) -> Dict:
        """Validate if species occurs in the given geographic region"""
        try:
            # Check against known Periyar species
            periyar_check = self.validate_periyar_occurrence(species_name)
            
            # Additional GBIF occurrence check for India/Kerala
            cache_key = f"occurrence_{species_name.lower()}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Search for occurrences in India
            occurrence_url = f"{self.gbif_base_url}/occurrence/search"
            params = {
                'scientificName': species_name,
                'country': 'IN',  # India
                'limit': 5
            }
            
            response = requests.get(occurrence_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                
                result = {
                    'status': 'checked',
                    'india_occurrences': count,
                    'found_in_india': count > 0,
                    'periyar_known': periyar_check.get('found', False),
                    'confidence': min(1.0, count / 10) if count > 0 else 0.0
                }
                
                self.cache[cache_key] = result
                return result
            else:
                return {'status': 'api_error', 'message': 'Could not check occurrence data'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def validate_name_format(self, species_name: str) -> Dict:
        """Validate scientific name format"""
        try:
            # Basic format checks
            checks = {
                'has_two_parts': len(species_name.split()) >= 2,
                'starts_with_capital': species_name[0].isupper() if species_name else False,
                'no_numbers': not any(char.isdigit() for char in species_name),
                'valid_characters': all(c.isalpha() or c.isspace() or c in '.-' for c in species_name),
                'not_all_caps': not species_name.isupper(),
                'genus_capitalized': species_name.split()[0][0].isupper() if len(species_name.split()) > 0 else False
            }
            
            # Calculate format score
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            format_score = passed_checks / total_checks
            
            return {
                'status': 'valid' if format_score >= 0.8 else 'questionable',
                'format_score': format_score,
                'checks': checks,
                'issues': [check for check, passed in checks.items() if not passed]
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def validate_periyar_occurrence(self, species_name: str) -> Dict:
        """Check if species is known to occur in Periyar"""
        species_key = species_name.lower().replace(' ', '_')
        
        if species_key in self.periyar_species_list:
            return {
                'found': True,
                'reference_data': self.periyar_species_list[species_key],
                'confidence': 1.0
            }
        
        # Fuzzy matching for similar names
        best_match_score = 0
        best_match = None
        
        for key, data in self.periyar_species_list.items():
            scientific_name = data.get('scientific_name', '')
            score = fuzz.ratio(species_name.lower(), scientific_name.lower())
            
            if score > best_match_score and score > 80:  # 80% similarity threshold
                best_match_score = score
                best_match = data
        
        if best_match:
            return {
                'found': False,
                'similar_species': best_match,
                'similarity_score': best_match_score / 100,
                'confidence': best_match_score / 100
            }
        
        return {
            'found': False,
            'confidence': 0.0
        }
    
    def calculate_confidence_score(self, validation_results: Dict) -> float:
        """Calculate overall confidence score from validation results"""
        scores = []
        weights = {
            'name_validation': 0.3,
            'taxonomic_validation': 0.2,
            'geographic_validation': 0.25,
            'format_validation': 0.15,
            'periyar_validation': 0.1
        }
        
        for check, weight in weights.items():
            result = validation_results.get(check, {})
            
            if check == 'name_validation':
                if result.get('status') == 'found':
                    scores.append(result.get('confidence', 0) * weight)
                else:
                    scores.append(0)
            
            elif check == 'taxonomic_validation':
                if result.get('status') == 'validated':
                    family_score = 1.0 if result.get('family_match') else 0.5
                    order_score = 1.0 if result.get('order_match') else 0.5
                    scores.append(((family_score + order_score) / 2) * weight)
                else:
                    scores.append(0.3 * weight)  # Partial credit for having taxonomy
            
            elif check == 'geographic_validation':
                confidence = result.get('confidence', 0)
                scores.append(confidence * weight)
            
            elif check == 'format_validation':
                format_score = result.get('format_score', 0)
                scores.append(format_score * weight)
            
            elif check == 'periyar_validation':
                confidence = result.get('confidence', 0)
                scores.append(confidence * weight)
        
        return sum(scores)
    
    def generate_validation_notes(self, validation_results: Dict) -> str:
        """Generate human-readable validation notes"""
        notes = []
        
        # Name validation notes
        name_result = validation_results.get('name_validation', {})
        if name_result.get('status') == 'found':
            confidence = name_result.get('confidence', 0)
            if confidence >= 90:
                notes.append("✓ Species name verified in GBIF database")
            elif confidence >= 70:
                notes.append("⚠ Species name found but with moderate confidence")
            else:
                notes.append("⚠ Species name found but with low confidence")
        else:
            notes.append("✗ Species name not found in GBIF database")
        
        # Taxonomy validation notes
        tax_result = validation_results.get('taxonomic_validation', {})
        if tax_result.get('status') == 'validated':
            if tax_result.get('family_match') and tax_result.get('order_match'):
                notes.append("✓ Taxonomic classification verified")
            else:
                notes.append("⚠ Taxonomic classification partially verified")
        
        # Geographic validation notes
        geo_result = validation_results.get('geographic_validation', {})
        if geo_result.get('found_in_india'):
            notes.append("✓ Species occurrence confirmed in India")
        
        # Periyar validation notes
        periyar_result = validation_results.get('periyar_validation', {})
        if periyar_result.get('found'):
            notes.append("✓ Species known to occur in Periyar")
        elif periyar_result.get('similar_species'):
            notes.append("⚠ Similar species found in Periyar reference")
        
        return "; ".join(notes) if notes else "No validation notes available"
    
    def create_validation_result(self, is_valid: bool, message: str) -> Dict:
        """Create a standard validation result structure"""
        return {
            'is_valid': is_valid,
            'confidence_score': 1.0 if is_valid else 0.0,
            'validation_details': {'message': message},
            'validation_notes': message
        }


class PeriyarSpeciesExtractor:
    """Enhanced Species Extractor with OCR capabilities"""
    
    def __init__(self):
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.validator = SpeciesValidator() 

    def validate_extracted_species(self, species_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate all extracted species"""
        if not species_list:
            return []
        
        self.logger.info(f"Starting validation of {len(species_list)} species")
        
        # Perform batch validation
        validated_species = self.validator.validate_species_batch(species_list)
        
        # Log validation summary
        valid_count = sum(1 for s in validated_species if s.get('is_valid', False))
        self.logger.info(f"Validation completed: {valid_count}/{len(validated_species)} species validated")
        
        return validated_species

    def filter_valid_species(self, validated_species: List[Dict[str, Any]], min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Filter species based on validation confidence"""
        filtered = [
            species for species in validated_species 
            if species.get('confidence_score', 0) >= min_confidence
        ]
        
        self.logger.info(f"Filtered species: {len(filtered)}/{len(validated_species)} meet confidence threshold of {min_confidence}")
        return filtered



    def setup_gemini(self, api_key: str):
        """Setup Gemini AI model"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            # Test the connection
            test_response = self.model.generate_content("Hello")
            self.logger.info("Gemini API configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup Gemini: {e}")
            raise
    

    def process_large_pdf(self, pdf_path: str, original_filename: str, delay_between_pages: float = 1.0, validate_species: bool = True) -> List[Dict[str, Any]]:
        """Process large PDF with optional species validation"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            all_species_data = []
            
            self.logger.info(f"Processing {total_pages} pages from {original_filename}")
            
            # STEP 1: Extract full text from entire document for coordinate mapping
            full_text = ""
            for i in range(total_pages):  # Read ALL pages for coordinate mapping
                page = doc[i]
                page_text = page.get_text().strip()
                if page_text:
                    full_text += page_text + "\n\n"
            
            # STEP 2: Extract comprehensive location-coordinate mapping
            location_coords_map = self.extract_location_coordinate_mapping(full_text)
            self.logger.info(f"Found coordinates for {len(location_coords_map)} locations: {list(location_coords_map.keys())}")
            
            # STEP 3: Extract all authors (as before)
            author_sample = full_text[:5000]  # First 5000 chars for author detection
            all_authors = self.extract_all_authors_from_text(author_sample)
            self.logger.info(f"Found {len(all_authors)} authors: {list(all_authors.values())}")
            
            # STEP 4: Process each page with coordinate enrichment
            for page_num in range(total_pages):
                try:
                    self.logger.info(f"Processing page {page_num + 1}/{total_pages}")
                    
                    page = doc[page_num]
                    page_text = page.get_text().strip()
                    
                    if not page_text:
                        page_text = self.extract_text_with_gemini_vision(page)
                    
                    if page_text and len(page_text.strip()) > 50:
                        # Process page with coordinate enrichment
                        page_species = self.extract_species_from_page_with_coordinates(
                            page_text, original_filename, all_authors, location_coords_map, page_num + 1
                        )
                        
                        if page_species:
                            all_species_data.extend(page_species)
                            self.logger.info(f"Page {page_num + 1}: Found {len(page_species)} species")
                    
                    # Rate limiting
                    if page_num < total_pages - 1:
                        time.sleep(delay_between_pages)
                        
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            # Merge and deduplicate results
            if all_species_data:
                deduplicated_data = self.merge_duplicate_species_multi_author(all_species_data)
                self.logger.info(f"Total species after deduplication: {len(deduplicated_data)}")
                
                # Add validation step
                if validate_species:
                    validated_data = self.validate_extracted_species(deduplicated_data)
                    return validated_data
                else:
                    return deduplicated_data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error processing large PDF {original_filename}: {e}")
            return []



    def extract_species_from_page(self, page_text: str, source_file: str, 
                                all_authors: Dict[str, str], page_number: int) -> List[Dict[str, Any]]:
        """Extract species information from a single page with multi-author attribution"""
        if not page_text.strip():
            return []
        
        # Get primary author as fallback
        primary_authors = {k: v for k, v in all_authors.items() if k.startswith('primary_')}
        fallback_reference = list(primary_authors.values())[0] if primary_authors else list(all_authors.values())[0] if all_authors else os.path.splitext(source_file)[0]
        
        # Enhanced prompt with multi-author context
        prompt = f"""
        You are extracting species data from page {page_number} of a scientific document.
        
        IMPORTANT: This document may mention species observed by DIFFERENT AUTHORS. 
        Pay attention to author attribution in the text.
        
        AVAILABLE AUTHORS IN THIS DOCUMENT:
        {chr(10).join([f"- {ref}" for ref in all_authors.values()])}
        
        AUTHOR ATTRIBUTION RULES:
        1. Look for phrases like "According to Smith (1998)", "Smith et al. observed", "reported by Jones"
        2. If a species is mentioned with specific author attribution, use that author
        3. If no specific attribution, use the primary document author: {fallback_reference}
        4. Include surrounding context to determine correct author attribution
        
        EXTRACTION REQUIREMENTS:
        For each species, extract all details AND determine the correct author who observed it.
        
        Look for attribution patterns:
        - "According to [Author] ([Year])" 
        - "[Author] ([Year]) observed/found/reported"
        - "observed by [Author]"
        - "In [Author]'s study"
        - "[Author] mentions/states"
        
        OUTPUT FORMAT - JSON array with author-specific attribution:
        [
        {{
            "species_name": "exact species name",
            "location_name": "specific location",
            "latitude": null,
            "longitude": null,
            "sampling_period_from_month": null,
            "sampling_period_from_year": null,
            "sampling_period_to_month": null,
            "sampling_period_to_year": null,
            "sampling_season": null,
            "order_family_species": null,
            "threat_status": null,
            "relative_abundance": null,
            "endemism": null,
            "flora_or_fauna": "Flora/Fauna",
            "reference": "EXACT author who observed this species",
            "remarks": "include attribution context if any",
            "context_text": "surrounding text that indicates author attribution"
        }}
        ]
        
        CRITICAL: The "reference" field must be the EXACT author who observed the species, not always the document author.
        
        Page {page_number} text:
        {page_text}
        """
        
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    if response.text:
                        json_text = self.extract_json_from_response(response.text)
                        if json_text:
                            import json
                            species_data = json.loads(json_text)
                            
                            if isinstance(species_data, list):
                                processed_species = []
                                for species in species_data:
                                    # Post-process to ensure correct author attribution
                                    processed_species_record = self.post_process_author_attribution(
                                        species, page_text, all_authors, fallback_reference
                                    )
                                    if processed_species_record:
                                        processed_species.append(processed_species_record)
                                return processed_species
                            return []
                            
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    else:
                        self.logger.warning(f"API call failed (attempt {attempt + 1}), retrying: {e}")
                        time.sleep(3 + (attempt * 2))
                        
        except Exception as e:
            self.logger.error(f"Error extracting species from page {page_number}: {e}")
            return []
        
        return []

    def extract_species_from_page_with_coordinates(self, page_text: str, source_file: str, 
                                                all_authors: Dict[str, str], 
                                                location_coords_map: Dict[str, Dict[str, float]], 
                                                page_number: int) -> List[Dict[str, Any]]:
        """Enhanced species extraction with better coordinate detection"""
        
        # Initialize location database
        if not hasattr(self, 'location_db'):
            self.location_db = PeriyarLocationDatabase()
        
        # Extract species using existing method
        species_list = self.extract_species_from_page(page_text, source_file, all_authors, page_number)
        
        enriched_species = []
        for species in species_list:
            enriched_record = species.copy()
            
            # Check current coordinates
            current_lat = species.get('latitude')
            current_lon = species.get('longitude')
            has_valid_coords = self.validate_coordinates(current_lat, current_lon)
            
            coordinate_source = 'document'  # Default source
            
            if not has_valid_coords:
                location_name = species.get('location_name', '')
                if location_name:
                    # Try to find coordinates
                    found_lat, found_lon = self.find_coordinates_for_location(location_name, location_coords_map)
                    
                    if found_lat is not None and found_lon is not None:
                        enriched_record['latitude'] = found_lat
                        enriched_record['longitude'] = found_lon
                        
                        # Determine coordinate source
                        if location_name in location_coords_map:
                            coordinate_source = 'mapping'
                        else:
                            coordinate_source = 'database'
                        
                        enriched_record['coordinate_source'] = coordinate_source
            
            # Enhanced remarks processing
            context_text = species.get('context_text', '') or page_text[max(0, page_text.lower().find(species.get('species_name', '').lower()) - 200):
                                                                        page_text.lower().find(species.get('species_name', '').lower()) + 200]
            
            enhanced_remarks = self.clean_and_enhance_remarks(enriched_record, context_text)
            if enhanced_remarks:
                enriched_record['remarks'] = enhanced_remarks
            
            enriched_species.append(enriched_record)
        
        return enriched_species

    def validate_coordinates(self, lat, lon) -> bool:
        """Validate if coordinates are valid for Periyar region"""
        if lat is None or lon is None:
            return False
        
        try:
            lat_float = float(lat)
            lon_float = float(lon)
            
            # Periyar region bounds with some tolerance
            return (9.0 <= lat_float <= 10.0) and (76.5 <= lon_float <= 77.5)
        except (ValueError, TypeError):
            return False



    def post_process_author_attribution(self, species_data: Dict, page_text: str, 
                                    all_authors: Dict[str, str], fallback_reference: str) -> Dict[str, Any]:
        """Post-process species record to ensure correct author attribution"""
        
        species_name = species_data.get('species_name', '')
        context_text = species_data.get('context_text', '')
        
        if not species_name:
            return None
        
        # Find the species mention in the page text
        species_context = ""
        if species_name in page_text:
            # Get ±150 characters around the species mention
            species_pos = page_text.lower().find(species_name.lower())
            if species_pos != -1:
                start = max(0, species_pos - 150)
                end = min(len(page_text), species_pos + len(species_name) + 150)
                species_context = page_text[start:end]
        
        # Use context_text from AI if available, otherwise use extracted context
        analysis_context = context_text if context_text else species_context
        
        # Determine the correct author attribution
        attributed_author = self.extract_contextual_author(analysis_context, all_authors)
        
        # Clean and validate the species data
        cleaned_species = self.clean_and_validate_species_data(species_data, 1, attributed_author)
        
        # Add context information to remarks
        if cleaned_species and analysis_context:
            current_remarks = cleaned_species.get('remarks', '') or ''
            context_note = f"Context: {analysis_context[:100]}..." if len(analysis_context) > 100 else f"Context: {analysis_context}"
            if current_remarks:
                cleaned_species['remarks'] = f"{current_remarks}; {context_note}"
            else:
                cleaned_species['remarks'] = context_note
        
        return cleaned_species

    def extract_conservation_notes(self, text: str) -> str:
        """
        Extract conservation-related information from context text.
        Returns a concise sentence or empty string if none found.
        """
        if not text:
            return ""
        
        patterns = [
            r'(?:conservation status|protected|endangered|threatened|vulnerable|critically endangered)\s+([^.!?]+?)(?:[.!?]|$)',
            r'(?:IUCN|status)\s*:\s*(EN|VU|CR|NT|LC|DD|EX|EW)',
            r'(?:under protection|in danger|at risk)\s+([^.!?]+?)(?:[.!?]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                snippet = match.group(1).strip()
                sentence = snippet[0].upper() + snippet[1:].rstrip('.')
                return f"Conservation notes: {sentence}."
        
        return ""

    def clean_and_validate_species_data(self, species_data: Dict[str, Any], page_number: int, reference_string: str) -> Dict[str, Any]:
        """Clean, validate and standardize extracted species data"""
        
        # Initialize cleaned data
        cleaned = {}
        
        # 1. SPECIES NAME CLEANING
        species_name = str(species_data.get('species_name', '')).strip()
        if not species_name or species_name.lower() in ['null', 'none', 'nan', '']:
            return None  # Skip records without species name
        
        # Clean species name formatting
        species_name = re.sub(r'\s+', ' ', species_name)  # Remove extra spaces
        species_name = species_name.replace('"', '').replace("'", '')  # Remove quotes
        cleaned['species_name'] = species_name
        
        # 2. LOCATION NAME STANDARDIZATION
        location = str(species_data.get('location_name', '')).strip()
        if location and location.lower() not in ['null', 'none', 'nan', '']:
            # Standardize common location names
            location_mapping = {
                'periyar lake': 'Periyar Lake',
                'kumily': 'Kumily',
                'thekkady': 'Thekkady', 
                'mangaladevi': 'Mangaladevi',
                'sabarimala': 'Sabarimala',
                'periyar tiger reserve': 'Periyar Tiger Reserve'
            }
            location_lower = location.lower()
            for key, value in location_mapping.items():
                if key in location_lower:
                    location = value
                    break
            cleaned['location_name'] = location
        else:
            cleaned['location_name'] = None
        
        # 3. COORDINATE VALIDATION AND CONVERSION
        def clean_coordinate(coord_str, coord_type='lat'):
            if not coord_str or str(coord_str).lower() in ['null', 'none', 'nan', '']:
                return None
                
            try:
                # Handle different coordinate formats
                coord_str = str(coord_str).strip()
                
                # If already decimal
                if '°' not in coord_str:
                    coord_val = float(coord_str)
                    # Validate range for Periyar region
                    if coord_type == 'lat' and 9.0 <= coord_val <= 10.0:
                        return coord_val
                    elif coord_type == 'lon' and 76.5 <= coord_val <= 77.5:
                        return coord_val
                    else:
                        return None
                
                # Convert degrees/minutes/seconds to decimal
                # Pattern: 9°27'30"N or 77°8'24"E
                dms_pattern = r"(\d+)°(\d+)'(\d+(?:\.\d+)?)\""
                match = re.search(dms_pattern, coord_str)
                if match:
                    degrees = float(match.group(1))
                    minutes = float(match.group(2))
                    seconds = float(match.group(3))
                    decimal = degrees + minutes/60 + seconds/3600
                    
                    # Validate range
                    if coord_type == 'lat' and 9.0 <= decimal <= 10.0:
                        return round(decimal, 6)
                    elif coord_type == 'lon' and 76.5 <= decimal <= 77.5:
                        return round(decimal, 6)
                
                return None
            except (ValueError, AttributeError):
                return None
        
        cleaned['latitude'] = clean_coordinate(species_data.get('latitude'), 'lat')
        cleaned['longitude'] = clean_coordinate(species_data.get('longitude'), 'lon')
        
        # 4. DATE/TIME STANDARDIZATION
        def clean_month(month_str):
            if not month_str or str(month_str).lower() in ['null', 'none', 'nan', '']:
                return None
            
            month_str = str(month_str).strip().lower()
            month_mapping = {
                'jan': 'January', 'january': 'January', '1': 'January',
                'feb': 'February', 'february': 'February', '2': 'February',
                'mar': 'March', 'march': 'March', '3': 'March',
                'apr': 'April', 'april': 'April', '4': 'April',
                'may': 'May', '5': 'May',
                'jun': 'June', 'june': 'June', '6': 'June',
                'jul': 'July', 'july': 'July', '7': 'July',
                'aug': 'August', 'august': 'August', '8': 'August',
                'sep': 'September', 'september': 'September', '9': 'September',
                'oct': 'October', 'october': 'October', '10': 'October',
                'nov': 'November', 'november': 'November', '11': 'November',
                'dec': 'December', 'december': 'December', '12': 'December'
            }
            return month_mapping.get(month_str, None)
        
        def clean_year(year_str):
            if not year_str or str(year_str).lower() in ['null', 'none', 'nan', '']:
                return None
            try:
                year = int(float(str(year_str)))
                if 1900 <= year <= 2030:  # Reasonable year range
                    return year
            except (ValueError, TypeError):
                pass
            return None
        
        cleaned['sampling_period_from_month'] = clean_month(species_data.get('sampling_period_from_month'))
        cleaned['sampling_period_from_year'] = clean_year(species_data.get('sampling_period_from_year'))
        cleaned['sampling_period_to_month'] = clean_month(species_data.get('sampling_period_to_month'))
        cleaned['sampling_period_to_year'] = clean_year(species_data.get('sampling_period_to_year'))
        
        # 5. SEASON STANDARDIZATION
        season = str(species_data.get('sampling_season', '')).strip()
        if season and season.lower() not in ['null', 'none', 'nan', '']:
            season_mapping = {
                'pre-monsoon': 'Pre-Monsoon',
                'premonsoon': 'Pre-Monsoon',
                'monsoon': 'Monsoon',
                'post-monsoon': 'Post-monsoon',
                'postmonsoon': 'Post-monsoon',
                'winter': 'Winter',
                'summer': 'Summer',
                'dry': 'Dry Season',
                'wet': 'Wet Season'
            }
            season_lower = season.lower()
            cleaned['sampling_season'] = season_mapping.get(season_lower, season)
        else:
            cleaned['sampling_season'] = None
        
        # 6. TAXONOMY CLEANING
        taxonomy = str(species_data.get('order_family_species', '')).strip()
        if taxonomy and taxonomy.lower() not in ['null', 'none', 'nan', '']:
            # Ensure proper format: Order/Family/Species
            cleaned['order_family_species'] = taxonomy
        else:
            cleaned['order_family_species'] = None
        
        # 7. THREAT STATUS STANDARDIZATION
        threat = str(species_data.get('threat_status', '')).strip().upper()
        valid_statuses = ['EN', 'VU', 'NT', 'LC', 'DD', 'CR', 'EW', 'EX', 'LR']
        if threat in valid_statuses:
            cleaned['threat_status'] = threat
        else:
            cleaned['threat_status'] = None
        
        # 8. ABUNDANCE STANDARDIZATION
        abundance = str(species_data.get('relative_abundance', '')).strip()
        if abundance and abundance.lower() not in ['null', 'none', 'nan', '']:
            abundance_mapping = {
                'very common': 'Very common',
                'common': 'Common',
                'moderate': 'Moderate',
                'rare': 'Rare',
                'very rare': 'Very rare',
                'occasional': 'Occasional',
                'frequent': 'Frequent',
                'abundant': 'Common',
                'scarce': 'Rare'
            }
            abundance_lower = abundance.lower()
            cleaned['relative_abundance'] = abundance_mapping.get(abundance_lower, abundance)
        else:
            cleaned['relative_abundance'] = None
        
        # 9. ENDEMISM STANDARDIZATION  
        endemism = str(species_data.get('endemism', '')).strip().upper()
        valid_endemism = ['EN-K', 'EN-WG', 'EN-I', 'WD', 'ENDEMIC', 'NON-ENDEMIC']
        if endemism in valid_endemism:
            cleaned['endemism'] = endemism
        elif 'ENDEMIC' in endemism.upper():
            cleaned['endemism'] = 'ENDEMIC'
        else:
            cleaned['endemism'] = None
        
        # 10. FLORA/FAUNA CLASSIFICATION
        flora_fauna = str(species_data.get('flora_or_fauna', '')).strip()
        if flora_fauna and flora_fauna.lower() not in ['null', 'none', 'nan', '']:
            if flora_fauna.lower() in ['flora', 'plant', 'tree', 'shrub', 'herb']:
                cleaned['flora_or_fauna'] = 'Flora'
            elif flora_fauna.lower() in ['fauna', 'animal', 'mammal', 'bird', 'reptile', 'amphibian', 'fish', 'insect']:
                cleaned['flora_or_fauna'] = 'Fauna'
            else:
                cleaned['flora_or_fauna'] = flora_fauna
        else:
            cleaned['flora_or_fauna'] = None
        
        # 11. REFERENCE AND REMARKS - FINAL UPDATED VERSION
        cleaned['reference'] = reference_string

        remarks = str(species_data.get('remarks', '')).strip()
        if remarks and remarks.lower() not in ['null', 'none', 'nan', '']:
            # Clean page numbers from remarks
            cleaned_remarks = self.clean_page_numbers_from_remarks(remarks)
            cleaned['remarks'] = cleaned_remarks
        else:
            cleaned['remarks'] = None

        # ADD THIS MISSING RETURN STATEMENT:
        return cleaned



    def clean_page_numbers_from_remarks(self, remarks: str) -> str:
        """
        Remove page number references from remarks
        Examples: 
        - "Page 1; Found in forest" -> "Found in forest"
        - "Page 5" -> None
        """
        if not remarks or str(remarks).strip() == '':
            return None
        
        remarks = str(remarks).strip()
        
        # Remove patterns like "Page X;" or "Page X,"
        cleaned = re.sub(r'^Page\s+\d+\s*[;,]\s*', '', remarks, flags=re.IGNORECASE)
        
        # Remove patterns like "Page X" at the beginning
        cleaned = re.sub(r'^Page\s+\d+\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove patterns like "; Page X" at the end
        cleaned = re.sub(r'\s*[;,]\s*Page\s+\d+\s*$', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned = cleaned.strip()
        
        # If only page number was there, return None
        if not cleaned or cleaned.lower() in ['null', 'none', 'nan']:
            return None
        
        return cleaned



    def merge_duplicate_species(self, species_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Intelligently merge duplicate species records by combining information
        from multiple sources to create the most complete record possible.
        """
        if not species_list:
            return []
        
        # Group species by name (case-insensitive)
        species_groups = {}
        
        for species in species_list:
            # Handle None values properly
            species_name_raw = species.get('species_name')
            if species_name_raw is None:
                species_name = ''
            else:
                species_name = str(species_name_raw).strip().lower()
                
            if not species_name:
                continue
                
            if species_name not in species_groups:
                species_groups[species_name] = []
            species_groups[species_name].append(species)
        
        merged_species = []
        
        for species_name, duplicates in species_groups.items():
            if len(duplicates) == 1:
                # No duplicates, keep as is
                merged_species.append(duplicates[0])
            else:
                # Merge duplicates intelligently
                merged_record = self.merge_species_records(duplicates)
                merged_species.append(merged_record)
        
        return merged_species
    
    def merge_duplicate_species_multi_author(self, species_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate species while preserving different author attributions"""
        
        if not species_list:
            return []
        
        # Group by species name and author reference
        species_groups = {}
        
        for species in species_list:
            species_name = species.get('species_name', '').strip().lower()
            reference = species.get('reference', '').strip()
            
            # Create composite key: species_name + reference
            key = f"{species_name}||{reference}"
            
            if key not in species_groups:
                species_groups[key] = []
            species_groups[key].append(species)
        
        merged_species = []
        
        for group_key, group_records in species_groups.items():
            if len(group_records) == 1:
                merged_species.append(group_records[0])
            else:
                # Merge multiple records for same species-author combination
                merged_record = self.merge_species_records(group_records)
                merged_species.append(merged_record)
        
        # Sort by species name and then by reference
        merged_species.sort(key=lambda x: (x.get('species_name', '').lower(), x.get('reference', '')))
        
        return merged_species



    def merge_species_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple records of the same species into a single comprehensive record.
        Priority: Most complete record first, then merge missing information.
        """
        if not records:
            return {}
        
        if len(records) == 1:
            return records[0]
        
        # Define column priorities and merging strategies
        merge_strategies = {
            'species_name': 'most_complete',
            'location_name': 'combine_unique',
            'latitude': 'first_valid',
            'longitude': 'first_valid',
            'sampling_period_from_month': 'earliest_or_first',
            'sampling_period_from_year': 'earliest_or_first',
            'sampling_period_to_month': 'latest_or_first',
            'sampling_period_to_year': 'latest_or_first',
            'sampling_season': 'combine_unique',
            'order_family_species': 'most_complete',
            'threat_status': 'most_complete',
            'relative_abundance': 'most_complete',
            'endemism': 'most_complete',
            'flora_or_fauna': 'most_complete',
            'reference': 'first_valid',  # **UPDATED: Changed from 'combine_all' to 'first_valid'**
            'remarks': 'combine_unique'  # **UPDATED: Changed from 'combine_all' to 'combine_unique'**
        }

        
        # Sort records by completeness (most complete first)
        sorted_records = sorted(records, key=lambda x: self.calculate_completeness_score(x), reverse=True)
        
        # Start with the most complete record
        merged = sorted_records[0].copy()
        
        # Merge information from other records
        for record in sorted_records[1:]:
            for field, strategy in merge_strategies.items():
                merged[field] = self.apply_merge_strategy(
                    merged.get(field), 
                    record.get(field), 
                    strategy,
                    field
                )
        
        return merged

    def calculate_completeness_score(self, record: Dict[str, Any]) -> int:
        """
        Calculate a completeness score for a species record.
        Higher score means more complete information.
        """
        score = 0
        important_fields = [
            'species_name', 'location_name', 'latitude', 'longitude',
            'sampling_period_from_month', 'sampling_period_from_year',
            'order_family_species', 'threat_status', 'relative_abundance',
            'endemism', 'flora_or_fauna'
        ]
        
        for field in important_fields:
            value = record.get(field)
            if value and str(value).strip() and str(value).lower() not in ['null', 'none', 'nan', '']:
                if field in ['latitude', 'longitude']:
                    # Give extra weight to coordinate data
                    score += 3
                elif field in ['species_name', 'order_family_species']:
                    # Give extra weight to taxonomic data
                    score += 2
                else:
                    score += 1
        
        return score

    def apply_merge_strategy(self, current_value, new_value, strategy: str, field_name: str):
        """
        Apply the specified merge strategy to combine two field values.
        """
        def is_valid_value(val):
            return val is not None and str(val).strip() != '' and str(val).lower() not in ['null', 'none', 'nan']
        
        current_valid = is_valid_value(current_value)
        new_valid = is_valid_value(new_value)
        
        if not current_valid and not new_valid:
            return None
        elif not current_valid:
            return new_value
        elif not new_valid:
            return current_value
        
        # Both values are valid, apply strategy
        if strategy == 'most_complete':
            # Return the longer/more detailed value
            if len(str(new_value)) > len(str(current_value)):
                return new_value
            return current_value
        
        elif strategy == 'combine_unique':
            # Combine unique values separated by semicolon
            current_parts = [part.strip() for part in str(current_value).split(';') if part.strip()]
            new_parts = [part.strip() for part in str(new_value).split(';') if part.strip()]
            
            # Remove duplicates while preserving order
            combined = []
            seen = set()
            for part in current_parts + new_parts:
                part_lower = part.lower()
                if part_lower not in seen:
                    combined.append(part)
                    seen.add(part_lower)
            
            return '; '.join(combined) if combined else current_value
        
        elif strategy == 'combine_all':
            # Combine all values separated by semicolon
            parts = [str(current_value).strip(), str(new_value).strip()]
            unique_parts = []
            seen = set()
            for part in parts:
                if part and part.lower() not in seen:
                    unique_parts.append(part)
                    seen.add(part.lower())
            return '; '.join(unique_parts)
        
        elif strategy == 'first_valid':
            # Return the first valid value (current takes precedence)
            return current_value
        
        elif strategy == 'earliest_or_first':
            # For dates/years, return the earliest; otherwise first valid
            if field_name.endswith('_year'):
                try:
                    current_year = int(current_value)
                    new_year = int(new_value)
                    return min(current_year, new_year)
                except (ValueError, TypeError):
                    return current_value
            elif field_name.endswith('_month'):
                # For months, we need to consider the year context, but for now return first valid
                return current_value
            else:
                return current_value
        
        elif strategy == 'latest_or_first':
            # For dates/years, return the latest; otherwise first valid
            if field_name.endswith('_year'):
                try:
                    current_year = int(current_value)
                    new_year = int(new_value)
                    return max(current_year, new_year)
                except (ValueError, TypeError):
                    return current_value
            elif field_name.endswith('_month'):
                # For months, we need to consider the year context, but for now return first valid
                return current_value
            else:
                return current_value
        
        # Default: return current value
        return current_value

    def remove_duplicates_and_merge(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicates and merge species information intelligently.
        This replaces the simple drop_duplicates approach.
        """
        if not results:
            return []
        
        self.logger.info(f"Starting deduplication process with {len(results)} records")
        
        # First, merge duplicate species
        merged_results = self.merge_duplicate_species(results)
        
        self.logger.info(f"After merging duplicates: {len(merged_results)} unique species")
        
        # Additional cleanup: remove records with no species name
        cleaned_results = []
        for record in merged_results:
            species_name = record.get('species_name', '').strip()
            if species_name and species_name.lower() not in ['null', 'none', 'nan', '']:
                cleaned_results.append(record)
        
        self.logger.info(f"After cleanup: {len(cleaned_results)} valid species records")
        
        return cleaned_results

    def extract_all_authors_from_text(self, text: str) -> Dict[str, str]:
        """Enhanced author extraction with better pattern recognition"""
        authors_found = {}
        
        # Enhanced primary author patterns
        primary_patterns = [
            # Standard citation format
            r'([A-Z][a-z]+(?:\s+et\s+al\.?))\s*\((\d{4})\)',
            r'([A-Z][a-z]+(?:\s*,\s*[A-Z]\.?[A-Z]?\.?)*(?:\s*,\s*[A-Z][a-z]+(?:\s*,\s*[A-Z]\.?[A-Z]?\.?)*)*\s*(?:&|and)\s*[A-Z][a-z]+)\s*\((\d{4})\)',
            
            # Document header patterns
            r'(?:By|Authors?|Author\(s\)|Written\s+by):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:\(?(\d{4})\)?)?',
            r'(?:Compiled\s+by|Prepared\s+by|Edited\s+by):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:\(?(\d{4})\)?)?',
            
            # Title page patterns
            r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)*(?:\s+[A-Z][a-z]+)*)\s*$\s*^\d{4}$',  # Multiline pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n\s*(\d{4})',
        ]
        
        # Enhanced citation patterns within text
        citation_patterns = [
            # Attribution phrases
            r'(?:according\s+to|as\s+per|following|based\s+on)\s+([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4})\)',
            r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4})\)\s*(?:observed|reported|found|noted|documented|recorded|saw|sighted)',
            r'([A-Z][a-z]+\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][a-z]+))\s*\((\d{4})\)\s*(?:observed|reported|found|noted|documented|recorded)',
            
            # Passive attribution
            r'(?:observed|reported|found|noted|documented|recorded|seen|collected)\s+by\s+([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            r'(?:study\s+by|research\s+by|work\s+by|survey\s+by)\s+([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            
            # Active attribution
            r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4})\)\s*(?:mentions?|states?|suggests?|indicates?|reports?|describes?)',
            r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s+(?:and\s+(?:his|her|their)\s+)?(?:team|colleagues|co-workers)\s*\(?(\d{4})?\)?',
            
            # Possessive patterns
            r'(?:in\s+)?([A-Z][a-z]+)\'?s\s+(?:study|work|research|observations?|findings?|data|survey|census)\s*\(?(\d{4})?\)?',
            
            # Reference patterns
            r'(?:as\s+)?(?:reported|cited|mentioned|described)\s+(?:in|by)\s+([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            r'(?:see|cf\.|compare)\s+([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            
            # Communication patterns
            r'(?:personal\s+communication|pers\.?\s+comm\.?)\s+(?:with|from)?\s*([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
        ]
        
        # Search for primary document authors first (first 3000 chars)
        text_sample = text[:3000]
        for pattern in primary_patterns:
            matches = re.findall(pattern, text_sample, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        author_name = match[0].strip()
                        year = match[1].strip() if match[1] else ""
                        
                        if self.is_valid_author_name(author_name):
                            formatted_author = self.format_author_names(author_name)
                            key = f"primary_{formatted_author}_{year}"
                            authors_found[key] = f"{formatted_author} {year}" if year else formatted_author
                            break  # Take first primary author found
        
        # Search for cited authors throughout the text
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 1:
                    author_name = match[0].strip()
                    year = match[1].strip() if len(match) > 1 and match[1] else ""
                    
                    if self.is_valid_author_name(author_name):
                        formatted_author = self.format_author_names(author_name)
                        key = f"cited_{formatted_author}_{year}"
                        reference = f"{formatted_author} {year}" if year else formatted_author
                        authors_found[key] = reference
        
        return authors_found

    def is_valid_author_name(self, author_name: str) -> bool:
        """Validate if extracted text is likely an author name"""
        if not author_name or len(author_name) < 3:
            return False
        
        # Remove common false positives
        invalid_names = {
            'the', 'this', 'that', 'they', 'we', 'our', 'his', 'her', 'its',
            'and', 'or', 'but', 'also', 'however', 'therefore', 'thus',
            'species', 'genus', 'family', 'order', 'class', 'kingdom',
            'study', 'research', 'work', 'paper', 'article', 'book',
            'periyar', 'kerala', 'india', 'western', 'ghats',
            'forest', 'wildlife', 'sanctuary', 'reserve', 'park'
        }
        
        if author_name.lower() in invalid_names:
            return False
        
        # Check for valid author name patterns
        # Should start with capital letter and contain mostly letters
        if not re.match(r'^[A-Z][a-z]+', author_name):
            return False
        
        # Should not be mostly numbers or special characters
        if len(re.findall(r'[0-9]', author_name)) > len(author_name) * 0.3:
            return False
        
        return True


    def extract_location_coordinate_mapping(self, full_text: str) -> Dict[str, Dict[str, float]]:
        """Enhanced location-coordinate mapping with comprehensive pattern recognition"""
        location_coords = {}
        
        # Comprehensive coordinate patterns
        coordinate_patterns = [
            # Pattern 1: Location with decimal coordinates
            r'([A-Za-z][\w\s]{2,50}?)\s*[:\(,]?\s*(\d{1,2}\.\d{3,6})°?\s*N?,?\s*(\d{1,2}\.\d{3,6})°?\s*E?\s*[\)\]]?',
            
            # Pattern 2: GPS format variations
            r'([A-Za-z][\w\s]{2,50}?)\s*[-–]?\s*GPS\s*:?\s*(\d{1,2}\.\d{3,6})[,\s]+(\d{1,2}\.\d{3,6})',
            r'([A-Za-z][\w\s]{2,50}?)\s*[-–]?\s*(?:coordinates?|coords?)\s*:?\s*(\d{1,2}\.\d{3,6})[,\s]+(\d{1,2}\.\d{3,6})',
            
            # Pattern 3: Labeled coordinates
            r'([A-Za-z][\w\s]{2,50}?)\s*[-–]?\s*(?:Lat|Latitude)\s*:?\s*(\d{1,2}\.\d{3,6}),?\s*(?:Lon|Long|Longitude)\s*:?\s*(\d{1,2}\.\d{3,6})',
            r'([A-Za-z][\w\s]{2,50}?)\s*[-–]?\s*(?:N|North)\s*:?\s*(\d{1,2}\.\d{3,6}),?\s*(?:E|East)\s*:?\s*(\d{1,2}\.\d{3,6})',
            
            # Pattern 4: Coordinates first, then location
            r'(\d{1,2}\.\d{3,6})°?\s*N?,?\s*(\d{1,2}\.\d{3,6})°?\s*E?\s*[-–]\s*([A-Za-z][\w\s]{2,50})',
            r'(\d{1,2}\.\d{3,6})[,\s]+(\d{1,2}\.\d{3,6})\s*[-–]\s*([A-Za-z][\w\s]{2,50})',
            
            # Pattern 5: DMS format (Degrees Minutes Seconds)
            r'([A-Za-z][\w\s]{2,50}?)\s*[:\(]\s*(\d{1,2})°(\d{1,2})\'(\d{1,2}(?:\.\d+)?)\"?\s*N,?\s*(\d{1,2})°(\d{1,2})\'(\d{1,2}(?:\.\d+)?)\"?\s*E',
            
            # Pattern 6: Table format variations
            r'([A-Za-z][\w\s]{2,50}?)\s*[|\t]\s*(\d{1,2}\.\d{3,6})\s*[|\t]\s*(\d{1,2}\.\d{3,6})',
            r'([A-Za-z][\w\s]{2,50}?)\s+(\d{1,2}\.\d{3,6})\s+(\d{1,2}\.\d{3,6})\s*(?:\n|$)',
            
            # Pattern 7: Elevation with coordinates
            r'([A-Za-z][\w\s]{2,50}?)\s*\(\s*(\d{1,2}\.\d{3,6})\s*N,\s*(\d{1,2}\.\d{3,6})\s*E,?\s*\d+\s*m?\s*\)',
            
            # Pattern 8: Scientific notation
            r'([A-Za-z][\w\s]{2,50}?)\s*[:\(]\s*(\d{1,2}\.\d{3,6})\s*(?:×10|x10|E)?\s*,?\s*(\d{1,2}\.\d{3,6})',
            
            # Pattern 9: Nested parentheses
            r'([A-Za-z][\w\s]{2,50}?)\s*\(\s*(\d{1,2}\.\d{3,6})\s*,\s*(\d{1,2}\.\d{3,6})\s*\)',
            
            # Pattern 10: Colon separated
            r'([A-Za-z][\w\s]{2,50}?)\s*:\s*(\d{1,2}\.\d{3,6})\s*:\s*(\d{1,2}\.\d{3,6})',
        ]
        
        for pattern_idx, pattern in enumerate(coordinate_patterns):
            matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                try:
                    if pattern_idx in [3, 4]:  # Coordinates first patterns
                        lat, lon, location = match
                        lat, lon = float(lat), float(lon)
                    elif pattern_idx == 4:  # DMS format
                        location, lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec = match
                        lat = float(lat_deg) + float(lat_min)/60 + float(lat_sec)/3600
                        lon = float(lon_deg) + float(lon_min)/60 + float(lon_sec)/3600
                    else:  # Standard format
                        location, lat, lon = match
                        lat, lon = float(lat), float(lon)
                    
                    # Enhanced validation for Periyar region
                    if self.is_valid_periyar_coordinate(lat, lon):
                        cleaned_location = self.standardize_location_name(location)
                        
                        # Additional validation for location name
                        if self.is_valid_location_name(cleaned_location):
                            location_coords[cleaned_location] = {
                                'latitude': round(lat, 6),
                                'longitude': round(lon, 6),
                                'source_pattern': pattern_idx + 1,
                                'confidence': self.calculate_coordinate_confidence(location, lat, lon)
                            }
                            
                except (ValueError, IndexError) as e:
                    continue
        
        return location_coords

    def is_valid_periyar_coordinate(self, lat: float, lon: float) -> bool:
        """Enhanced coordinate validation for Periyar region"""
        # Expanded bounds with buffer for nearby areas
        return (8.8 <= lat <= 10.2) and (76.3 <= lon <= 77.7)

    def is_valid_location_name(self, location: str) -> bool:
        """Validate if extracted text is a valid location name"""
        if not location or len(location) < 2:
            return False
        
        # Remove invalid location patterns
        invalid_patterns = [
            r'^\d+$',  # Only numbers
            r'^[^a-zA-Z]*$',  # No letters
            r'^(and|or|the|at|in|on|by|for|with|from)$',  # Common words
            r'^(page|fig|figure|table|chart|graph)$',  # Document elements
        ]
        
        location_lower = location.lower()
        for pattern in invalid_patterns:
            if re.match(pattern, location_lower):
                return False
        
        return True

    def calculate_coordinate_confidence(self, location: str, lat: float, lon: float) -> float:
        """Calculate confidence score for extracted coordinates"""
        confidence = 0.8  # Base confidence
        
        # Location name quality
        if len(location.split()) >= 2:
            confidence += 0.1
        
        # Coordinate precision
        lat_precision = len(str(lat).split('.')[-1]) if '.' in str(lat) else 0
        lon_precision = len(str(lon).split('.')[-1]) if '.' in str(lon) else 0
        
        if lat_precision >= 4 and lon_precision >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)


    def standardize_location_name(self, location: str) -> str:
        """Enhanced location name standardization with better pattern recognition"""
        if not location:
            return ""
        
        location = location.strip()
        
        # Remove common prefixes and suffixes with more comprehensive patterns
        prefixes_to_remove = [
            r'^(?:at|near|in|from|around|close to|vicinity of|area of|region of)\s+',
            r'^(?:the|a|an)\s+',
            r'^(?:upper|lower|north|south|east|west|northern|southern|eastern|western)\s+',
            r'^(?:site|location|place|spot)\s+(?:at|near|of)\s+',
        ]
        
        suffixes_to_remove = [
            r'\s+(?:area|region|zone|camp|trail|station|point|site|location|spot|place)$',
            r'\s+(?:forest|reserve|sanctuary|park|garden|plantation)$',
            r'\s+(?:range|hills|mountain|peak|valley|river|stream|lake|pond)$',
            r'\s+(?:division|section|block|compartment|beat)$',
        ]
        
        # Apply removals
        for prefix in prefixes_to_remove:
            location = re.sub(prefix, '', location, flags=re.IGNORECASE)
        for suffix in suffixes_to_remove:
            location = re.sub(suffix, '', location, flags=re.IGNORECASE)
        
        # Enhanced location mappings with more variations
        location_mappings = {
            # Lake variations
            'periyar lake': 'Periyar Lake',
            'periyar reservoir': 'Periyar Lake', 
            'periyar dam': 'Periyar Lake',
            'periyar water body': 'Periyar Lake',
            'lake periyar': 'Periyar Lake',
            
            # Town variations
            'kumily town': 'Kumily',
            'kumily market': 'Kumily',
            'kumily bus stand': 'Kumily',
            'thekkady town': 'Thekkady',
            'thekkady market': 'Thekkady',
            'thekkady boat landing': 'Thekkady',
            
            # Temple variations
            'mangaladevi temple': 'Mangaladevi',
            'mangala devi': 'Mangaladevi',
            'mangaladevi hill': 'Mangaladevi',
            'sabarimala temple': 'Sabarimala',
            'sabarimala shrine': 'Sabarimala',
            
            # Reserve variations
            'periyar tiger reserve': 'Periyar Tiger Reserve',
            'ptr': 'Periyar Tiger Reserve',
            'periyar reserve': 'Periyar Tiger Reserve',
            'periyar national park': 'Periyar Tiger Reserve',
            'periyar wildlife sanctuary': 'Periyar Tiger Reserve',
            
            # Trail and route variations
            'periyar nature trail': 'Periyar Nature Trail',
            'nature trail': 'Periyar Nature Trail',
            'bamboo rafting': 'Bamboo Rafting Route',
            'bamboo rafting area': 'Bamboo Rafting Route',
            'jungle patrol': 'Jungle Patrol Route',
            'patrol route': 'Jungle Patrol Route',
            'elephant junction': 'Elephant Junction',
            
            # Forest compartments and beats
            'thannikudy': 'Thannikudy',
            'thannikudi': 'Thannikudy',
            'manakkavala': 'Manakkavala',
            'mullaperiyar': 'Mullaperiyar',
            'vallakkadavu': 'Vallakkadavu',
            'vallakadavu': 'Vallakkadavu',
            
            # Research and accommodation
            'aranya nivas': 'Aranya Nivas',
            'spice village': 'Spice Village',
            'lake palace': 'Lake Palace',
            'forest rest house': 'Forest Rest House',
            'interpretation centre': 'Wildlife Interpretation Centre',
            'research station': 'Forest Research Station',
        }
        
        location_lower = location.lower().strip()
        
        # Direct mapping first
        if location_lower in location_mappings:
            return location_mappings[location_lower]
        
        # Partial matching for complex location names
        for variant, standard in location_mappings.items():
            if variant in location_lower or location_lower in variant:
                # Check if it's a meaningful match (not just single character)
                if len(set(variant.split()) & set(location_lower.split())) >= 1:
                    return standard
        
        # Clean and title case for unmatched locations
        cleaned = re.sub(r'\s+', ' ', location).strip()
        return ' '.join(word.capitalize() for word in cleaned.split() if word)



    def find_coordinates_for_location(self, location_name: str, location_coords_map: Dict[str, Dict[str, float]]) -> tuple:
        """Enhanced coordinate finding with automatic detection"""
        if not location_name:
            return None, None
        
        # Initialize location database
        if not hasattr(self, 'location_db'):
            self.location_db = PeriyarLocationDatabase()
        
        location_clean = self.standardize_location_name(location_name)
        
        # Step 1: Check document-extracted coordinates first
        if location_coords_map:
            for mapped_location, coords in location_coords_map.items():
                if self.locations_match(location_clean, mapped_location):
                    return coords['latitude'], coords['longitude']
        
        # Step 2: Check known Periyar location database
        lat, lon = self.location_db.get_coordinates(location_clean)
        if lat and lon:
            return lat, lon
        
        # Step 3: Try variations and partial matches
        location_variations = self.generate_location_variations(location_clean)
        for variation in location_variations:
            lat, lon = self.location_db.get_coordinates(variation)
            if lat and lon:
                return lat, lon
        
        return None, None

    def locations_match(self, location1: str, location2: str) -> bool:
        """Check if two location names refer to the same place"""
        if not location1 or not location2:
            return False
        
        loc1_clean = location1.lower().strip()
        loc2_clean = location2.lower().strip()
        
        # Direct match
        if loc1_clean == loc2_clean:
            return True
        
        # Substring match
        if loc1_clean in loc2_clean or loc2_clean in loc1_clean:
            return True
        
        # Word overlap match (at least 60% words in common)
        words1 = set(loc1_clean.split())
        words2 = set(loc2_clean.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        overlap = len(words1 & words2)
        min_words = min(len(words1), len(words2))
        
        return overlap / min_words >= 0.6

    def generate_location_variations(self, location: str) -> List[str]:
        """Generate possible variations of a location name"""
        variations = [location]
        
        # Add common suffixes
        suffixes = ['Area', 'Region', 'Zone', 'Trail', 'Point', 'Station']
        for suffix in suffixes:
            variations.append(f"{location} {suffix}")
        
        # Add common prefixes
        prefixes = ['Upper', 'Lower', 'North', 'South', 'East', 'West']
        for prefix in prefixes:
            variations.append(f"{prefix} {location}")
        
        # Remove common words and try again
        words = location.split()
        if len(words) > 1:
            # Try without first word
            variations.append(' '.join(words[1:]))
            # Try without last word
            variations.append(' '.join(words[:-1]))
        
        return variations



    def extract_contextual_author(self, text_context: str, all_authors: Dict[str, str]) -> str:
        """Enhanced contextual author extraction with better pattern matching"""
        if not text_context or not all_authors:
            return list(all_authors.values())[0] if all_authors else "Unknown Author"
        
        context = text_context.lower()
        
        # Enhanced attribution patterns with more comprehensive matching
        attribution_patterns = [
            # Direct attribution patterns
            r'(?:according\s+to|as\s+per|following|based\s+on)\s+([a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            r'([a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4})\)\s*(?:observed|reported|found|noted|documented|recorded|saw|sighted|mentioned|stated)',
            r'(?:observed|reported|found|noted|documented|seen|recorded|collected)\s+by\s+([a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            
            # Possessive patterns
            r'([a-z]+(?:\s+et\s+al\.?)?)\'?s\s+(?:study|work|research|observations?|findings?|data|survey|census)',
            r'(?:in|from)\s+([a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            
            # Citation patterns
            r'(?:as\s+)?(?:reported|cited|mentioned|described)\s+(?:in|by)\s+([a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            r'([a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4})\)\s*(?:mentions?|states?|suggests?|indicates?|reports?|describes?)',
            
            # Study reference patterns
            r'(?:study\s+by|research\s+by|work\s+by)\s+([a-z]+(?:\s+et\s+al\.?)?)\s*\(?(\d{4})?\)?',
            r'([a-z]+(?:\s+et\s+al\.?)?)\s+(?:and\s+(?:his|her|their)\s+)?(?:team|colleagues|co-workers)',
            
            # Communication patterns
            r'(?:personal\s+communication|pers\.?\s+comm\.?)\s+(?:with|from)?\s*([a-z]+(?:\s+et\s+al\.?)?)',
            r'(?:verbal|oral)\s+(?:communication|report)\s+(?:from|by)\s+([a-z]+(?:\s+et\s+al\.?)?)',
        ]
        
        best_match = None
        best_confidence = 0
        
        for pattern in attribution_patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                if isinstance(match, tuple):
                    author_name = match[0].strip()
                    year = match[1].strip() if len(match) > 1 and match[1] else ""
                else:
                    author_name = match.strip()
                    year = ""
                
                # Skip if too generic
                if len(author_name) < 3 or author_name.lower() in ['the', 'this', 'that', 'they', 'we', 'our', 'his', 'her']:
                    continue
                
                # Find best matching author from available authors
                for key, full_reference in all_authors.items():
                    # Calculate match confidence
                    confidence = self.calculate_author_match_confidence(author_name, year, full_reference)
                    
                    if confidence > best_confidence and confidence > 0.6:  # Minimum 60% confidence
                        best_match = full_reference
                        best_confidence = confidence
        
        # Return best match or fallback to primary author
        if best_match:
            return best_match
        
        # Fallback hierarchy: primary authors first, then any author
        primary_authors = {k: v for k, v in all_authors.items() if k.startswith('primary_')}
        if primary_authors:
            return list(primary_authors.values())[0]
        
        return list(all_authors.values())[0] if all_authors else "Unknown Author"

    def calculate_author_match_confidence(self, extracted_name: str, extracted_year: str, full_reference: str) -> float:
        """Calculate confidence score for author matching"""
        if not extracted_name or not full_reference:
            return 0.0
        
        confidence = 0.0
        extracted_lower = extracted_name.lower()
        reference_lower = full_reference.lower()
        
        # Name matching
        if extracted_lower in reference_lower:
            confidence += 0.6
        elif any(word in reference_lower for word in extracted_lower.split() if len(word) > 2):
            confidence += 0.4
        
        # Year matching (if available)
        if extracted_year and extracted_year in full_reference:
            confidence += 0.3
        
        # Et al. handling
        if 'et al' in extracted_lower and 'et al' in reference_lower:
            confidence += 0.2
        
        # First author matching (most important)
        ref_first_word = reference_lower.split()[0] if reference_lower.split() else ""
        ext_first_word = extracted_lower.split()[0] if extracted_lower.split() else ""
        
        if ref_first_word and ext_first_word and ref_first_word == ext_first_word:
            confidence += 0.4
        
        return min(confidence, 1.0)  # Cap at 1.0



    def format_author_names(self, author_string: str) -> str:
        """
        Format author names to the desired format: 'Zacharias et al. 1996' style
        Input: Various formats like "Zacharias, V.J., Johnsingh, A.J.T. & Goyal, S.P."
        Output: "Zacharias et al." (without year, year will be added separately)
        """
        if not author_string:
            return ""
        
        # Clean the input
        author_string = author_string.strip()
        
        # Remove common prefixes
        prefixes_to_remove = ['by:', 'author:', 'authors:', 'written by:']
        for prefix in prefixes_to_remove:
            if author_string.lower().startswith(prefix):
                author_string = author_string[len(prefix):].strip()
        
        # Handle different author format patterns
        
        # Pattern 1: Already in "et al." format
        if 'et al' in author_string.lower():
            # Extract first author name before "et al"
            match = re.match(r'([A-Z][a-z]+)', author_string)
            if match:
                return f"{match.group(1)} et al."
            return author_string
        
        # Pattern 2: Multiple authors separated by commas and &/and
        # Example: "Zacharias, V.J., Johnsingh, A.J.T. & Goyal, S.P."
        if '&' in author_string or ' and ' in author_string.lower():
            # Split by comma to get individual authors
            authors = re.split(r'\s*,\s*', author_string)
            
            # Extract first author's surname
            first_author = authors[0].strip()
            # Get just the surname (first word)
            surname_match = re.match(r'([A-Z][a-z]+)', first_author)
            if surname_match:
                first_surname = surname_match.group(1)
                
                # Count total number of authors
                # The last author is usually after & or 'and'
                author_count = len(authors)
                
                # If there's an & or 'and' in the last element, it indicates multiple authors
                last_part = authors[-1] if authors else ""
                if '&' in last_part or ' and ' in last_part.lower():
                    author_count += 1  # Add one more for the author after &/and
                
                if author_count > 1:
                    return f"{first_surname} et al."
                else:
                    return first_surname
        
        # Pattern 3: Single author or simple format
        # Just extract the first surname
        surname_match = re.match(r'([A-Z][a-z]+)', author_string)
        if surname_match:
            return surname_match.group(1)
        
        # Fallback: return cleaned original
        return author_string



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
        """Extract species information from text using Gemini AI with enhanced prompting"""
        if not text.strip():
            return []

        # Extract reference information from the document
        author_names, citation_year = self.extract_reference_info(text)

        # Create reference string in desired format
        if author_names and citation_year:
            reference_string = f"{author_names} {citation_year}"
        elif author_names:
            reference_string = author_names
        elif citation_year:
            reference_string = f"Unknown Author {citation_year}"
        else:
            reference_string = os.path.splitext(source_file)[0]

        # Your existing prompt code here...
        prompt = f"""
        You are extracting species data from a scientific document about Periyar, India (Kerala).
        
        DOCUMENT CONTEXT:
        - Location: Periyar Tiger Reserve, Kerala, India
        - Coordinates range: Lat 9.2°-9.7°N, Long 76.9°-77.3°E
        - Ecosystem: Western Ghats tropical forests
        
        EXTRACTION REQUIREMENTS:
        Extract ALL species mentioned with complete details. Look for:
        1. Scientific names (often in italics/parentheses)
        2. Common names 
        3. Location specifics (trails, elevations, zones)
        4. GPS coordinates or degree/minute/second notation
        5. Sampling dates, periods, seasons
        6. Conservation status (EN, VU, NT, LC, DD, CR)
        7. Abundance indicators (common, rare, frequent, etc.)
        8. Endemic status indicators
        9. Taxonomic information (Order, Family)
        10. Habitat and behavioral notes
        
        OUTPUT FORMAT (JSON array):
        [
        {{
            "species_name": "exact name from document",
            "location_name": "specific location within Periyar",
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
            "reference": "{reference_string}",
            "remarks": "habitat and behavior notes"
        }}
        ]
        
        IMPORTANT: Extract EVERY species mentioned, even brief references.
        
        Document text to analyze:
        {text[:10000]}
        """

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                json_text = self.extract_json_from_response(response.text)
                if json_text:
                    import json
                    species_data = json.loads(json_text)
                    
                    if isinstance(species_data, list):
                        # Apply cleaning and validation to each species
                        processed_species = []
                        for species in species_data:
                            cleaned_species = self.clean_and_validate_species_data(species, 1, reference_string)
                            if cleaned_species:
                                processed_species.append(cleaned_species)
                        
                        return processed_species
                    return []
        except Exception as e:
            self.logger.error(f"Error extracting species data: {e}")
            return []
        
        return []  # This should be the final return


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text().strip()
                
                if page_text:
                    text += page_text + "\n\n"
                else:
                    # Fallback to OCR for pages without extractable text
                    ocr_text = self.extract_text_with_gemini_vision(page)
                    if ocr_text:
                        text += ocr_text + "\n\n"
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""

    
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
        """Process a single PDF file with original filename and coordinate mapping"""
        self.logger.info(f"Processing PDF: {original_filename}")
        
        try:
            # Check PDF size/pages to decide processing method
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()
            
            if page_count > 3:  # Use enhanced processing for larger PDFs
                self.logger.info(f"Large PDF detected ({page_count} pages), using enhanced coordinate processing")
                species_data = self.process_large_pdf(pdf_path, original_filename, delay_between_pages=1.5)
            else:
                # Enhanced processing even for small PDFs
                doc = fitz.open(pdf_path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    page_text = page.get_text().strip()
                    if page_text:
                        full_text += page_text + "\n\n"
                doc.close()
                
                # Extract coordinate mapping even for small PDFs
                location_coords_map = self.extract_location_coordinate_mapping(full_text)
                all_authors = self.extract_all_authors_from_text(full_text[:3000])
                
                if not full_text.strip():
                    self.logger.warning(f"No text extracted from {original_filename}")
                    return []
                
                # Use enhanced extraction
                species_data = self.extract_species_from_text_with_coordinates(
                    full_text, original_filename, all_authors, location_coords_map
                )
            
            self.logger.info(f"Extracted {len(species_data)} species from {original_filename}")
            return species_data
            
        except Exception as e:
            self.logger.error(f"Error processing {original_filename}: {e}")
            return []

    def extract_species_from_text_with_coordinates(self, text: str, source_file: str, 
                                                all_authors: Dict[str, str], 
                                                location_coords_map: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Extract species from text with coordinate enrichment"""
        
        # First extract species using existing method
        species_list = self.extract_species_from_text(text, source_file)
        
        # Enrich with coordinates
        enriched_species = []
        
        for species in species_list:
            enriched_record = species.copy()
            
            # Check if coordinates are missing or invalid
            current_lat = species.get('latitude')
            current_lon = species.get('longitude')
            
            has_valid_coords = (current_lat is not None and current_lon is not None and
                            str(current_lat).lower() not in ['null', 'none', 'nan', ''] and
                            str(current_lon).lower() not in ['null', 'none', 'nan', ''])
            
            if not has_valid_coords:
                location_name = species.get('location_name', '')
                if location_name:
                    found_lat, found_lon = self.find_coordinates_for_location(location_name, location_coords_map)
                    
                    if found_lat is not None and found_lon is not None:
                        enriched_record['latitude'] = found_lat
                        enriched_record['longitude'] = found_lon
                        
                        # Add coordinate source to remarks
                        current_remarks = enriched_record.get('remarks', '') or ''
                        coord_note = "Coordinates from document location mapping"
                        enriched_record['remarks'] = f"{current_remarks}; {coord_note}" if current_remarks else coord_note
            
            enriched_species.append(enriched_record)
        
        return enriched_species

    def clean_and_enhance_remarks(self, species_data: Dict, context_text: str = "") -> str:
        """Create clean, essential remarks with complete sentences"""
        remarks_parts = []
        
        # Extract habitat information
        habitat_info = self.extract_habitat_info(context_text)
        if habitat_info:
            remarks_parts.append(habitat_info)
        
        # Extract behavioral information
        behavior_info = self.extract_behavior_info(context_text)
        if behavior_info:
            remarks_parts.append(behavior_info)
        
        # Add coordinate source if applicable
        if species_data.get('latitude') and species_data.get('longitude'):
            coord_source = species_data.get('coordinate_source', 'document')
            if coord_source == 'database':
                remarks_parts.append("Coordinates obtained from Periyar location database.")
            elif coord_source == 'mapping':
                remarks_parts.append("Coordinates mapped from document location data.")
        
        # Add sampling context if available
        sampling_context = self.extract_sampling_context(context_text)
        if sampling_context:
            remarks_parts.append(sampling_context)
        
        # Add conservation notes
        conservation_notes = self.extract_conservation_notes(context_text)
        if conservation_notes:
            remarks_parts.append(conservation_notes)
        
        # Combine and validate sentences
        final_remarks = []
        for remark in remarks_parts:
            cleaned_remark = self.ensure_complete_sentence(remark)
            if cleaned_remark and len(cleaned_remark) > 10:  # Minimum length check
                final_remarks.append(cleaned_remark)
        
        return ' '.join(final_remarks) if final_remarks else None

    def extract_habitat_info(self, text: str) -> str:
        """Extract habitat information and form complete sentence"""
        if not text:
            return ""
        
        habitat_patterns = [
            r'(?:found|observed|seen|recorded|noted)\s+(?:in|at|on|near)\s+([^.!?]+?)(?:[.!?]|$)',
            r'(?:habitat|occurs|lives|inhabits)\s+([^.!?]+?)(?:[.!?]|$)',
            r'(?:canopy|understory|ground|tree|forest|grassland|wetland)\s+([^.!?]+?)(?:[.!?]|$)',
        ]
        
        for pattern in habitat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                habitat_text = matches[0].strip()
                if len(habitat_text) > 5:
                    return f"Species observed in {habitat_text.lower()}."
        
        return ""

    def extract_behavior_info(self, text: str) -> str:
        """Extract behavioral information"""
        behavior_patterns = [
            r'(?:feeding|foraging|nesting|breeding|roosting)\s+([^.!?]+?)(?:[.!?]|$)',
            r'(?:active|inactive|dormant)\s+(?:during|in|at)\s+([^.!?]+?)(?:[.!?]|$)',
            r'(?:behavior|behaviour|activity)\s+([^.!?]+?)(?:[.!?]|$)',
        ]
        
        for pattern in behavior_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                behavior_text = matches[0].strip()
                if len(behavior_text) > 5:
                    return f"Behavioral notes: {behavior_text.lower()}."
        
        return ""

    def extract_sampling_context(self, text: str) -> str:
        """Extract sampling methodology context"""
        sampling_patterns = [
            r'(?:sampled|collected|surveyed|monitored)\s+(?:using|by|through)\s+([^.!?]+?)(?:[.!?]|$)',
            r'(?:method|technique|approach)\s+([^.!?]+?)(?:[.!?]|$)',
            r'(?:transect|quadrat|plot|line)\s+([^.!?]+?)(?:[.!?]|$)',
        ]
        
        for pattern in sampling_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                sampling_text = matches[0].strip()
                if len(sampling_text) > 5:
                    return f"Sampling method: {sampling_text.lower()}."
        
        return ""

    def ensure_complete_sentence(self, text: str) -> str:
        """Ensure text forms a complete sentence"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure sentence ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Remove redundant spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def test_enhanced_location_system():
        """Test the enhanced location extraction system"""
        extractor = PeriyarSpeciesExtractor()
        
        # Test location standardization
        test_locations = [
            "near Periyar lake area",
            "at Thekkady town",
            "Mangala devi temple region",
            "vicinity of Kumily market"
        ]
        
        print("Location Standardization Test:")
        for loc in test_locations:
            standardized = extractor.standardize_location_name(loc)
            print(f"{loc} -> {standardized}")
        
        # Test coordinate detection
        print("\nCoordinate Detection Test:")
        for loc in test_locations:
            standardized = extractor.standardize_location_name(loc)
            lat, lon = extractor.find_coordinates_for_location(standardized, {})
            print(f"{standardized}: {lat}, {lon}")


class PeriyarLocationDatabase:
    """Database of known Periyar locations with coordinates"""
    
    def __init__(self):
        self.locations = {
            # Major landmarks
            'Periyar Lake': {'lat': 9.4667, 'lon': 77.1833},
            'Thekkady': {'lat': 9.5950, 'lon': 77.1590},
            'Kumily': {'lat': 9.5833, 'lon': 77.1667},
            'Mangaladevi': {'lat': 9.4333, 'lon': 77.2000},
            'Sabarimala': {'lat': 9.4417, 'lon': 77.0833},
            
            # Trails and routes
            'Periyar Nature Trail': {'lat': 9.4700, 'lon': 77.1800},
            'Bamboo Rafting Route': {'lat': 9.4600, 'lon': 77.1750},
            'Jungle Patrol Route': {'lat': 9.4500, 'lon': 77.1900},
            'Elephant Junction': {'lat': 9.4800, 'lon': 77.1700},
            
            # Forest areas
            'Thannikudy': {'lat': 9.5167, 'lon': 77.1833},
            'Manakkavala': {'lat': 9.4500, 'lon': 77.2167},
            'Mullaperiyar': {'lat': 9.5000, 'lon': 77.1500},
            'Vallakkadavu': {'lat': 9.4333, 'lon': 77.1833},
            
            # Peaks and viewpoints
            'Kottaimalai': {'lat': 9.4833, 'lon': 77.2000},
            'Pandikuzhi': {'lat': 9.5000, 'lon': 77.2167},
            'Chokkampetti': {'lat': 9.4167, 'lon': 77.1667},
            
            # Research stations
            'Forest Research Station': {'lat': 9.4750, 'lon': 77.1820},
            'Wildlife Interpretation Centre': {'lat': 9.4680, 'lon': 77.1790},
            
            # Accommodation areas
            'Aranya Nivas': {'lat': 9.4650, 'lon': 77.1800},
            'Spice Village': {'lat': 9.5900, 'lon': 77.1600},
            'Lake Palace': {'lat': 9.4700, 'lon': 77.1850},
        }
        
        # Add elevation data where available
        self.elevations = {
            'Mangaladevi': 1337,
            'Kottaimalai': 1200,
            'Pandikuzhi': 1000,
            'Thekkady': 700,
            'Kumily': 900,
        }
    
    def get_coordinates(self, location_name: str) -> tuple:
        """Get coordinates for a location with fuzzy matching"""
        if not location_name:
            return None, None
        
        location_clean = location_name.strip()
        
        # Direct match
        if location_clean in self.locations:
            loc_data = self.locations[location_clean]
            return loc_data['lat'], loc_data['lon']
        
        # Fuzzy matching
        best_match = None
        best_score = 0
        
        for known_location in self.locations.keys():
            # Check for substring matches
            if location_clean.lower() in known_location.lower() or known_location.lower() in location_clean.lower():
                score = len(set(location_clean.lower().split()) & set(known_location.lower().split()))
                if score > best_score:
                    best_score = score
                    best_match = known_location
            
            # Check for word overlap
            location_words = set(location_clean.lower().split())
            known_words = set(known_location.lower().split())
            overlap = len(location_words & known_words)
            if overlap > 0 and overlap > best_score:
                best_score = overlap
                best_match = known_location
        
        if best_match and best_score > 0:
            loc_data = self.locations[best_match]
            return loc_data['lat'], loc_data['lon']
        
        return None, None
    
    def add_location(self, name: str, lat: float, lon: float):
        """Add a new location to the database"""
        if 9.0 <= lat <= 10.0 and 76.5 <= lon <= 77.5:  # Validate Periyar bounds
            self.locations[name] = {'lat': lat, 'lon': lon}


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

def process_uploaded_files(uploaded_files, delay_between_pages=1.5, enable_validation=True, confidence_threshold=0.5):
    """Process uploaded PDF files with validation"""
    if not st.session_state.extractor:
        st.error("Please test your API key first!")
        return None

    results = []
    total_files = len(uploaded_files)
    
    # Create progress containers
    file_progress = st.progress(0)
    page_progress = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update file progress
            file_progress.progress((i + 1) / total_files)
            status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")
            
            # Save temporarily and process
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Get page count for progress tracking
                doc = fitz.open(tmp_path)
                total_pages = doc.page_count
                doc.close()
                
                log_message(f"[{i+1}/{total_files}] Processing {uploaded_file.name} ({total_pages} pages)")
                
                # Process with page-level progress updates
                species_data = st.session_state.extractor.process_large_pdf(
                    tmp_path, uploaded_file.name, delay_between_pages
                )
                
                results.extend(species_data)
                log_message(f" ✓ Extracted {len(species_data)} species")
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            log_message(f" ✗ Error processing {uploaded_file.name}: {e}")
            continue
    
    # Clear progress indicators
    file_progress.empty()
    page_progress.empty()
    status_text.empty()

    
    if results:
        # Apply intelligent deduplication and merging
        deduplicated_results = st.session_state.extractor.remove_duplicates_and_merge(results)
        
        if enable_validation:
            log_message("Starting species validation...")
            validated_results = st.session_state.extractor.validate_extracted_species(deduplicated_results)
            
            # Filter by confidence if requested
            if confidence_threshold > 0:
                filtered_results = st.session_state.extractor.filter_valid_species(
                    validated_results, confidence_threshold
                )
                log_message(f"Filtered to {len(filtered_results)} species above confidence threshold")
                return filtered_results
            else:
                return validated_results
        else:
            return deduplicated_results
    
    return None
    
def create_download_files(results, filename_base, output_format):
    """Create download files with validation information"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # Define column order including validation columns
    column_order = [
        'species_name', 'is_valid', 'confidence_score', 'corrected_name',
        'location_name', 'latitude', 'longitude',
        'sampling_period_from_month', 'sampling_period_from_year',
        'sampling_period_to_month', 'sampling_period_to_year',
        'sampling_season', 'order_family_species', 'threat_status',
        'relative_abundance', 'endemism', 'flora_or_fauna',
        'reference', 'remarks', 'validation_notes'
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

def test_species_validation():
    """Test the species validation system"""
    validator = SpeciesValidator()
    
    test_species = [
        {"species_name": "Panthera tigris", "location_name": "Periyar"},
        {"species_name": "Invalid species name", "location_name": "Periyar"},
        {"species_name": "Shorea robusta", "location_name": "Periyar"}
    ]
    
    results = validator.validate_species_batch(test_species)
    
    for result in results:
        print(f"Species: {result['species_name']}")
        print(f"Valid: {result['is_valid']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Notes: {result['validation_notes']}")
        print("-" * 50)


def show_results_summary(df):
    """Display results summary"""
    st.markdown("### 📊 Processing Summary")
    
    # Add information about original vs deduplicated count
    if 'original_count' in st.session_state:
        st.info(f"🔄 Deduplication: {st.session_state.original_count} → {len(df)} records (removed {st.session_state.original_count - len(df)} duplicates)")
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

def show_validation_summary(df):
    """Display validation summary"""
    if 'is_valid' not in df.columns:
        return
    
    st.markdown("### 🔍 Validation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        valid_count = len(df[df['is_valid'] == True])
        st.metric("Valid Species", valid_count)
    
    with col2:
        invalid_count = len(df[df['is_valid'] == False])
        st.metric("Invalid Species", invalid_count)
    
    with col3:
        avg_confidence = df['confidence_score'].mean() if 'confidence_score' in df.columns else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        high_confidence = len(df[df['confidence_score'] >= 0.8]) if 'confidence_score' in df.columns else 0
        st.metric("High Confidence", high_confidence)
    
    # Show validation details in expandable section
    with st.expander("Validation Details"):
        if 'validation_notes' in df.columns:
            validation_df = df[['species_name', 'is_valid', 'confidence_score', 'validation_notes']].copy()
            st.dataframe(validation_df)


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
