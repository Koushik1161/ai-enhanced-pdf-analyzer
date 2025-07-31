"""
OCR module for extracting mathematical expressions from images in PDFs.
Implements bonus feature for Task 2.
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
import easyocr
import re
from typing import List, Dict, Tuple, Optional
from PIL import Image
import io
import os
from dataclasses import dataclass


@dataclass
class MathExpression:
    """Represents a mathematical expression found in an image."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    page_num: int
    extraction_method: str


class OCRMathExtractor:
    """Extracts mathematical expressions from images using OCR techniques."""
    
    def __init__(self):
        """Initialize OCR engines."""
        self.easyocr_reader = None
        self.math_patterns = [
            r'[∫∑∏∂∇∆√π∞±×÷≤≥≠≈∈∉⊂⊃∪∩]',  # Mathematical symbols
            r'\b\d+\s*[+\-*/=]\s*\d+',           # Basic arithmetic
            r'\b\w+\s*=\s*\d+',                   # Variable assignments
            r'\b\d+\.\d+',                        # Decimal numbers
            r'\b\d+/\d+',                         # Fractions
            r'\b\d+\s*%',                         # Percentages
            r'\$\s*\d+',                          # Currency
            r'\b\d+\s*[°]\s*',                    # Degrees
            r'\b[a-zA-Z]\s*=\s*[a-zA-Z0-9+\-*/()]+',  # Algebraic equations
            r'\b(?:sin|cos|tan|log|ln|exp)\s*\([^)]+\)',  # Functions
        ]
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader lazily."""
        if self.easyocr_reader is None:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
            except Exception as e:
                print(f"Warning: Could not initialize EasyOCR: {e}")
                self.easyocr_reader = False
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract all images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Convert to numpy array for OpenCV
                            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            
                            images.append({
                                'image': img_cv,
                                'pil_image': img_pil,
                                'page_num': page_num,
                                'img_index': img_index,
                                'width': pix.width,
                                'height': pix.height
                            })
                        
                        pix = None  # Clean up
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            return images
            
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
            return []
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy for mathematical content.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Resize image if it's too small (OCR works better on larger images)
        height, width = cleaned.shape
        if height < 50 or width < 50:
            scale_factor = max(2, 100 / min(height, width))
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
    
    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of extracted text with confidence scores
        """
        results = []
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image_for_ocr(image)
            
            # Configure Tesseract for better math recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()[]{}.,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ∫∑∏∂∇∆√π∞±×÷≤≥≠≈∈∉⊂⊃∪∩°$%'
            
            # Extract text with detailed information
            data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter results and combine words into expressions
            text_blocks = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        text_blocks.append({
                            'text': text,
                            'confidence': int(data['conf'][i]),
                            'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                        })
            
            # Combine nearby text blocks into expressions
            if text_blocks:
                combined_text = ' '.join([block['text'] for block in text_blocks])
                avg_confidence = sum([block['confidence'] for block in text_blocks]) / len(text_blocks)
                
                results.append({
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'bbox': (0, 0, image.shape[1], image.shape[0]),
                    'method': 'tesseract'
                })
                
        except Exception as e:
            print(f"Error in Tesseract OCR: {e}")
        
        return results
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of extracted text with confidence scores
        """
        results = []
        
        try:
            self._init_easyocr()
            
            if self.easyocr_reader is False:
                return results
            
            # Preprocess image
            processed_image = self.preprocess_image_for_ocr(image)
            
            # Extract text
            ocr_results = self.easyocr_reader.readtext(processed_image)
            
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.3:  # Confidence threshold
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                    
                    results.append({
                        'text': text.strip(),
                        'confidence': confidence * 100,  # Convert to percentage
                        'bbox': (x, y, w, h),
                        'method': 'easyocr'
                    })
                    
        except Exception as e:
            print(f"Error in EasyOCR: {e}")
        
        return results
    
    def is_mathematical_expression(self, text: str) -> bool:
        """
        Determine if extracted text contains mathematical expressions.
        
        Args:
            text: Extracted text string
            
        Returns:
            True if text appears to contain math
        """
        if not text or len(text.strip()) < 2:
            return False
        
        # Check against mathematical patterns
        for pattern in self.math_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def extract_math_from_pdf_images(self, pdf_path: str) -> List[MathExpression]:
        """
        Extract mathematical expressions from all images in a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of mathematical expressions found
        """
        math_expressions = []
        
        # Extract images from PDF
        images = self.extract_images_from_pdf(pdf_path)
        
        if not images:
            print("No images found in PDF")
            return math_expressions
        
        print(f"Found {len(images)} images in PDF")
        
        for img_data in images:
            try:
                image = img_data['image']
                page_num = img_data['page_num']
                
                # Try both OCR methods
                tesseract_results = self.extract_text_tesseract(image)
                easyocr_results = self.extract_text_easyocr(image)
                
                # Combine and filter results
                all_results = tesseract_results + easyocr_results
                
                for result in all_results:
                    text = result['text']
                    if self.is_mathematical_expression(text):
                        math_expr = MathExpression(
                            text=text,
                            confidence=result['confidence'],
                            bbox=result['bbox'],
                            page_num=page_num,
                            extraction_method=result['method']
                        )
                        math_expressions.append(math_expr)
                        
            except Exception as e:
                print(f"Error processing image from page {img_data['page_num']}: {e}")
                continue
        
        # Remove duplicates and sort by confidence
        unique_expressions = []
        seen_texts = set()
        
        for expr in sorted(math_expressions, key=lambda x: x.confidence, reverse=True):
            if expr.text not in seen_texts:
                unique_expressions.append(expr)
                seen_texts.add(expr.text)
        
        return unique_expressions
    
    def save_extracted_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Save all extracted images from PDF for inspection.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            
        Returns:
            List of saved image paths
        """
        saved_paths = []
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            images = self.extract_images_from_pdf(pdf_path)
            
            for img_data in images:
                filename = f"page_{img_data['page_num']}_img_{img_data['img_index']}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Save the image
                img_data['pil_image'].save(output_path)
                saved_paths.append(output_path)
                
        except Exception as e:
            print(f"Error saving extracted images: {e}")
        
        return saved_paths


def extract_math_from_images(pdf_path: str, output_dir: str = None) -> Dict:
    """
    Main function to extract mathematical expressions from PDF images.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional output directory for saving images
        
    Returns:
        Dictionary containing extracted math expressions and metadata
    """
    extractor = OCRMathExtractor()
    
    # Extract mathematical expressions
    math_expressions = extractor.extract_math_from_pdf_images(pdf_path)
    
    # Prepare results
    results = {
        'pdf_file': os.path.basename(pdf_path),
        'total_images_processed': len(extractor.extract_images_from_pdf(pdf_path)),
        'math_expressions_found': len(math_expressions),
        'expressions': []
    }
    
    for expr in math_expressions:
        results['expressions'].append({
            'text': expr.text,
            'confidence': expr.confidence,
            'page_number': expr.page_num,
            'extraction_method': expr.extraction_method,
            'bbox': expr.bbox
        })
    
    # Save extracted images if output directory specified
    if output_dir:
        saved_images = extractor.save_extracted_images(pdf_path, output_dir)
        results['saved_images'] = saved_images
    
    return results


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Extract mathematical expressions from PDF images using OCR")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output directory for extracted images")
    parser.add_argument("-s", "--save", action="store_true", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Extract math expressions
    print(f"Processing PDF: {args.pdf_path}")
    results = extract_math_from_images(args.pdf_path, args.output)
    
    # Display results
    print(f"\nResults:")
    print(f"  Images processed: {results['total_images_processed']}")
    print(f"  Math expressions found: {results['math_expressions_found']}")
    
    if results['expressions']:
        print(f"\nExtracted expressions:")
        for i, expr in enumerate(results['expressions'], 1):
            print(f"  {i}. '{expr['text']}' (confidence: {expr['confidence']:.1f}%, method: {expr['extraction_method']}, page: {expr['page_number']})")
    
    # Save results if requested
    if args.save:
        output_file = os.path.splitext(args.pdf_path)[0] + "_ocr_math_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")