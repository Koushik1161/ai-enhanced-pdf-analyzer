"""
AI-Enhanced PDF Document Analyzer
==================================

A comprehensive PDF analysis tool with AI-powered insights, visual representations,
and intelligent document understanding capabilities.

Features:
- PDF text extraction with layout preservation
- Mathematical expression detection and analysis
- Named Entity Recognition (NER) with 15+ entity types
- AI-powered document summarization
- Visual representations with entity highlighting
- OCR math extraction from embedded images
- OpenAI integration for intelligent insights
- CLI and API interfaces
- Docker containerization

Author: AI-Enhanced PDF Analyzer Team
Version: 2.0 (AI-Enhanced)
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF
import spacy
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PDFAnalyzer:
    """
    Advanced PDF analyzer with AI capabilities.
    
    Implements all 7 core requirements plus bonus features:
    1. PDF text extraction
    2. Mathematical expression detection  
    3. Named Entity Recognition
    4. Document summarization
    5. CLI interface
    6. Clean, modular code
    7. Comprehensive documentation
    """
    
    def __init__(self):
        """Initialize the PDF analyzer with NLP models."""
        print("Initializing PDF Analyzer...")
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Error: spaCy English model not found. Please run:")
            print("python -m spacy download en_core_web_sm")
            sys.exit(1)
        
        # Initialize OpenAI for summarization (with fallback)
        self.llm = None
        self.fallback_summarizer = None
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            try:
                self.llm = OpenAI(
                    api_key=openai_api_key,
                    model_name="gpt-3.5-turbo-instruct",
                    max_tokens=200,
                    temperature=0.3
                )
            except Exception as e:
                print(f"Warning: OpenAI initialization failed: {e}")
        
        # Fallback summarizer
        if not self.llm:
            try:
                self.fallback_summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    max_length=150,
                    min_length=100,
                    do_sample=False
                )
            except Exception as e:
                print(f"Warning: Fallback summarizer initialization failed: {e}")
        
        # Mathematical expression patterns
        self.math_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\\\[[^\]]+\\\]',  # LaTeX display math
            r'\\\([^)]+\\\)',  # LaTeX inline math alternative
            r'[∫∑∏∂∇∆√π∞±×÷≤≥≠≈∈∉⊂⊃∪∩]',  # Mathematical symbols
            r'\b\d+\s*[+\-*/=]\s*\d+',  # Basic arithmetic
            r'\b\w+\s*=\s*\d+',  # Variable assignments
            r'\b\d+\.\d+',  # Decimal numbers in equations
            r'\b\d+/\d+',  # Fractions
            r'\bf\([^)]+\)\s*=\s*[^,\n]+',  # Functions
            r'\b(?:sin|cos|tan|log|ln|exp)\s*\([^)]+\)',  # Mathematical functions
        ]
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Task 1: Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            print(f"Extracting text from: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = ""
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                text += page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n"
            
            doc.close()
            print(f"Successfully extracted text from {page_count} pages")
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_math_expressions(self, text: str) -> List[str]:
        """
        Task 2: Extract mathematical expressions from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of mathematical expressions found
        """
        print("Extracting mathematical expressions...")
        expressions = []
        
        for pattern in self.math_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            expressions.extend(matches)
        
        # Remove duplicates while preserving order
        unique_expressions = []
        seen = set()
        for expr in expressions:
            expr_clean = expr.strip()
            if expr_clean and len(expr_clean) > 1 and expr_clean not in seen:
                unique_expressions.append(expr_clean)
                seen.add(expr_clean)
        
        print(f"Found {len(unique_expressions)} mathematical expressions")
        return unique_expressions
    
    def perform_ner(self, text: str) -> Dict[str, List[str]]:
        """
        Task 3: Perform Named Entity Recognition.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        print("Performing Named Entity Recognition...")
        
        # Limit text size for processing efficiency
        max_chars = 1000000  # 1M characters
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text.strip()
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            if entity_text not in entities[entity_type]:
                entities[entity_type].append(entity_text)
        
        # Sort entities by frequency (most common first)
        for entity_type in entities:
            entities[entity_type] = sorted(entities[entity_type])
        
        total_entities = sum(len(entities[et]) for et in entities)
        print(f"Found {total_entities} entities across {len(entities)} categories")
        
        return entities
    
    def summarize_text(self, text: str) -> str:
        """
        Task 4: Generate document summary (100-150 words).
        
        Args:
            text: Input text to summarize
            
        Returns:
            Generated summary
        """
        print("Generating summary...")
        
        # Limit text for summarization
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        try:
            if self.llm:
                # Use OpenAI LLM
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template="""
                    Please provide a concise summary of the following document in 100-150 words.
                    Focus on the main topics, key findings, and important concepts.
                    
                    Document text:
                    {text}
                    
                    Summary:
                    """
                )
                
                chain = prompt_template | self.llm | StrOutputParser()
                summary = chain.invoke({"text": text})
                
            elif self.fallback_summarizer:
                # Use BART summarizer
                summary_result = self.fallback_summarizer(text)
                summary = summary_result[0]['summary_text']
                
            else:
                # Basic extractive summary
                sentences = text.split('.')[:10]
                summary = '. '.join(sentences[:3]) + '.'
            
            # Ensure summary is within word limit
            words = summary.split()
            if len(words) > 150:
                summary = ' '.join(words[:150]) + '...'
            
            word_count = len(summary.split())
            print(f"Generated summary with {word_count} words")
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary generation failed due to technical issues."
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete PDF analysis combining all features.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing all analysis results
        """
        print("\n" + "="*60)
        print("Starting PDF Analysis")
        print("="*60)
        
        # Extract text
        text = self.extract_text(pdf_path)
        if not text:
            return {"error": "Failed to extract text from PDF"}
        
        # Perform all analyses
        math_expressions = self.extract_math_expressions(text)
        entities = self.perform_ner(text)
        summary = self.summarize_text(text)
        
        # Get PDF metadata
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except:
            total_pages = 0
        
        # Compile results
        results = {
            "pdf_file": os.path.basename(pdf_path),
            "total_pages": total_pages,
            "summary": summary,
            "entities": entities,
            "total_entities": sum(len(entities[et]) for et in entities),
            "math_expressions": math_expressions,
            "total_math_expressions": len(math_expressions)
        }
        
        return results


def create_cli():
    """
    Task 5: Create CLI interface
    """
    parser = argparse.ArgumentParser(
        description="AI-Enhanced PDF Document Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_analyzer.py document.pdf
  python pdf_analyzer.py document.pdf --output results.json
  python pdf_analyzer.py document.pdf --format detailed
  python pdf_analyzer.py document.pdf --no-visualize --no-ai-enhanced
  python pdf_analyzer.py document.pdf --no-ocr-math
        """
    )
    
    parser.add_argument("pdf_file", help="Path to the PDF file to analyze")
    parser.add_argument("-o", "--output", help="Output file for results (JSON format)")
    parser.add_argument("-f", "--format", choices=["simple", "detailed"], 
                       default="detailed", help="Output format (default: detailed)")
    
    # Advanced features enabled by default
    parser.add_argument("--no-visualize", action="store_true", 
                       help="Disable visual representations")
    parser.add_argument("--no-ocr-math", action="store_true", 
                       help="Disable OCR math extraction")
    parser.add_argument("--no-ai-enhanced", action="store_true",
                       help="Disable AI-enhanced analysis")
    parser.add_argument("--output-dir", help="Output directory (default: ./output)")
    parser.add_argument("--no-auto-save", action="store_true",
                       help="Disable automatic output saving")
    
    return parser


def display_results(results: Dict[str, Any], format_type: str = "simple"):
    """
    Display results in console
    """
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nPDF File: {results.get('pdf_file', 'Unknown')}")
    print(f"Total Pages: {results.get('total_pages', 0)}")
    
    print("\n" + "="*60)
    
    # Document Summary
    print("\nDOCUMENT SUMMARY:")
    print("-" * 60)
    print(results.get('summary', 'No summary available'))
    
    print("\n" + "="*60)
    
    # Named Entities
    entities = results.get('entities', {})
    total_entities = results.get('total_entities', 0)
    
    print(f"NAMED ENTITIES (Total: {total_entities}):")
    print("-" * 60)
    
    if format_type == "detailed":
        for entity_type, entity_list in entities.items():
            print(f"\n{entity_type} ({len(entity_list)} found):")
            for i, entity in enumerate(entity_list[:10]):  # Show first 10
                print(f"  - {entity}")
            if len(entity_list) > 10:
                print(f"  ... and {len(entity_list) - 10} more")
    else:
        for entity_type, entity_list in entities.items():
            examples = ", ".join(entity_list[:5])
            remaining = len(entity_list) - 5
            if remaining > 0:
                examples += f" ... and {remaining} more"
            print(f"\n{entity_type} ({len(entity_list)} found):")
            print(f"  {examples}")
    
    print("\n" + "="*60)
    
    # Mathematical Expressions
    math_expressions = results.get('math_expressions', [])
    total_math = results.get('total_math_expressions', 0)
    
    print(f"MATHEMATICAL EXPRESSIONS (Total: {total_math}):")
    print("-" * 60)
    
    if math_expressions:
        for i, expr in enumerate(math_expressions[:10], 1):
            print(f"{i}. {expr}")
        if len(math_expressions) > 10:
            print(f"... and {len(math_expressions) - 10} more")
    else:
        print("No mathematical expressions found")
    
    print("\n" + "="*60 + "\n")


def main():
    """
    Main entry point
    """
    # Parse CLI arguments
    parser = create_cli()
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file '{args.pdf_file}' not found!")
        return 1
    
    # Set up output directory
    output_dir = args.output_dir or "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = PDFAnalyzer()
    
    # Analyze PDF
    results = analyzer.analyze_pdf(args.pdf_file)
    
    # Convert negative flags to positive for easier handling
    args.visualize = not args.no_visualize
    args.ocr_math = not args.no_ocr_math
    args.ai_enhanced = not args.no_ai_enhanced
    args.auto_save = not args.no_auto_save
    
    # Bonus Feature: OCR Math Extraction
    if args.ocr_math:
        try:
            if args.ai_enhanced:
                # Use enhanced OCR with AI analysis
                from enhanced_ocr_extractor import enhanced_extract_math_from_images
                print("\n🔍 AI-enhanced math extraction from images...")
                
                output_dir = args.output_dir or "output"
                ocr_results = enhanced_extract_math_from_images(args.pdf_file, output_dir)
                
                # Add enhanced results
                if ocr_results.get('expressions'):
                    results['math_expressions'].extend(ocr_results['expressions'])
                    results['total_math_expressions'] += len(ocr_results['expressions'])
                    results['enhanced_ocr_results'] = ocr_results
                    print(f"   ✅ Found {len(ocr_results['expressions'])} expressions with AI analysis")
                else:
                    print("   ℹ️ No math expressions found in images")
            else:
                # Use standard OCR
                from ocr_math_extractor import extract_math_from_images
                print("\n🔍 Extracting math expressions from images...")
                
                output_dir = args.output_dir or "output"
                ocr_results = extract_math_from_images(args.pdf_file, output_dir)
                
                # Add OCR results to main results
                if ocr_results['expressions']:
                    ocr_expressions = [expr['text'] for expr in ocr_results['expressions']]
                    results['math_expressions'].extend(ocr_expressions)
                    results['total_math_expressions'] += len(ocr_expressions)
                    results['ocr_math_results'] = ocr_results
                    print(f"   ✅ Found {len(ocr_expressions)} additional math expressions in images")
                else:
                    print("   ℹ️ No math expressions found in images")
                
        except ImportError:
            print("   ⚠️ OCR dependencies not installed. Install with: pip install pytesseract easyocr opencv-python")
        except Exception as e:
            print(f"   ❌ Error in OCR extraction: {e}")
    
    # Display results
    display_results(results, args.format)
    
    # Auto-save results
    if args.auto_save or args.output:
        # Generate output filename if not specified
        if args.output:
            output_file = args.output
        else:
            base_name = os.path.basename(args.pdf_file).replace('.pdf', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.json")
            
        # Save JSON results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Also save formatted text output
        text_output_file = output_file.replace('.json', '.txt')
        with open(text_output_file, 'w') as f:
            # Redirect print output to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            display_results(results, args.format)
            sys.stdout = old_stdout
        print(f"📄 Text output saved to: {text_output_file}")
    
    # Bonus Feature: Visual Representations
    if args.visualize:
        try:
            if args.ai_enhanced:
                # Use enhanced visualizer with AI insights
                from enhanced_visualizer import enhanced_visualize_pdf_analysis
                print("\n🎨 Creating AI-enhanced visual representations...")
                
                output_dir = args.output_dir or "output"
                visual_files = enhanced_visualize_pdf_analysis(args.pdf_file, results, output_dir)
            else:
                # Use standard visualizer
                from visualizer import visualize_pdf_analysis
                print("\n🎨 Creating visual representations...")
                
                output_dir = args.output_dir or "output"
                visual_files = visualize_pdf_analysis(args.pdf_file, results, output_dir)
            
            if visual_files:
                print("   ✅ Created visualization files:")
                for file_type, file_path in visual_files.items():
                    print(f"      {file_type}: {file_path}")
            else:
                print("   ⚠️ Could not create visualizations")
                
        except ImportError:
            print("   ⚠️ Visualization dependencies not installed. Install with: pip install matplotlib Pillow")
        except Exception as e:
            print(f"   ❌ Error creating visualizations: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())