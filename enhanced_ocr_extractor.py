"""
Enhanced OCR module with OpenAI API integration for intelligent math expression analysis.
Provides AI-powered interpretation and explanation of mathematical content.
"""

import os
import json
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv

# Import base OCR functionality  
try:
    from ocr_math_extractor import OCRMathExtractor, MathExpression
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR dependencies not available. Install with: pip install pytesseract easyocr opencv-python")

load_dotenv()


class EnhancedOCRMathExtractor:
    """Enhanced OCR extractor with AI-powered mathematical analysis."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        if OCR_AVAILABLE:
            self.base_extractor = OCRMathExtractor()
        else:
            self.base_extractor = None
    
    def analyze_math_expressions_with_ai(self, expressions: List[str]) -> Dict[str, Any]:
        """
        Use OpenAI to analyze and explain mathematical expressions.
        
        Args:
            expressions: List of mathematical expressions
            
        Returns:
            Dictionary with AI analysis of the expressions
        """
        if not expressions:
            return {"analysis": "No mathematical expressions to analyze"}
        
        try:
            # Prepare expressions for analysis
            expr_text = "\n".join([f"{i+1}. {expr}" for i, expr in enumerate(expressions)])
            
            prompt = f"""
            Analyze these mathematical expressions found in a document:
            
            {expr_text}
            
            Please provide:
            1. Classification of each expression (algebra, calculus, statistics, finance, etc.)
            2. Complexity level (basic, intermediate, advanced)
            3. Context interpretation (what field/domain they likely belong to)
            4. Educational level (high school, undergraduate, graduate, professional)
            5. Practical applications (where these might be used)
            6. Key concepts involved
            
            Format as JSON with the following structure:
            {{
                "overall_analysis": {{
                    "dominant_field": "field name",
                    "complexity_level": "level",
                    "educational_level": "level"
                }},
                "expression_details": [
                    {{
                        "expression": "the expression",
                        "classification": "type",
                        "complexity": "level",
                        "explanation": "what this represents"
                    }}
                ],
                "applications": ["list of practical uses"],
                "key_concepts": ["main mathematical concepts"]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a mathematics expert specializing in expression analysis and educational content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback structure
                analysis = {
                    "overall_analysis": {
                        "dominant_field": "Analysis provided",
                        "complexity_level": "See full analysis",
                        "educational_level": "See full analysis"
                    },
                    "expression_details": [],
                    "applications": ["See full analysis"],
                    "key_concepts": ["See full analysis"],
                    "raw_analysis": response.choices[0].message.content
                }
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Could not analyze expressions with AI: {str(e)}",
                "expressions_count": len(expressions)
            }
    
    def generate_math_learning_recommendations(self, expressions: List[str], ai_analysis: Dict) -> Dict[str, Any]:
        """
        Generate learning recommendations based on found mathematical expressions.
        
        Args:
            expressions: List of mathematical expressions
            ai_analysis: Previous AI analysis results
            
        Returns:
            Learning recommendations and resources
        """
        try:
            complexity = ai_analysis.get('overall_analysis', {}).get('complexity_level', 'unknown')
            field = ai_analysis.get('overall_analysis', {}).get('dominant_field', 'mathematics')
            
            prompt = f"""
            Based on this mathematical content analysis:
            - Field: {field}
            - Complexity: {complexity}
            - Expressions found: {len(expressions)}
            - Key concepts: {ai_analysis.get('key_concepts', [])}
            
            Provide learning recommendations:
            1. Prerequisites someone should know to understand this content
            2. Next topics to learn after mastering this material
            3. Recommended learning resources (books, online courses, etc.)
            4. Practice exercises suggestions
            5. Real-world projects where these concepts apply
            
            Format as JSON with keys: prerequisites, next_topics, resources, practice_suggestions, projects
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an educational consultant specializing in mathematics curriculum and learning paths."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            try:
                recommendations = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                recommendations = {"full_recommendations": response.choices[0].message.content}
            
            return recommendations
            
        except Exception as e:
            return {"error": f"Could not generate learning recommendations: {str(e)}"}
    
    def create_math_complexity_analysis(self, expressions: List[str]) -> Dict[str, Any]:
        """
        Analyze complexity and difficulty of mathematical expressions.
        
        Args:
            expressions: List of mathematical expressions
            
        Returns:
            Complexity analysis results
        """
        try:
            prompt = f"""
            Analyze the complexity of these mathematical expressions:
            
            {chr(10).join([f"{i+1}. {expr}" for i, expr in enumerate(expressions)])}
            
            Rate each expression on:
            1. Computational complexity (1-10 scale)
            2. Conceptual difficulty (1-10 scale) 
            3. Required mathematical background
            4. Time to solve (estimate)
            
            Also provide an overall difficulty assessment and learning curve analysis.
            
            Return as JSON with detailed scoring.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a mathematics education expert who assesses difficulty levels of mathematical problems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            try:
                complexity_analysis = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                complexity_analysis = {"analysis": response.choices[0].message.content}
            
            return complexity_analysis
            
        except Exception as e:
            return {"error": f"Could not analyze complexity: {str(e)}"}
    
    def enhanced_extract_math_from_pdf(self, pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Enhanced math extraction with AI analysis.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for results
            
        Returns:
            Enhanced results with AI analysis
        """
        # Use base OCR extraction if available
        if self.base_extractor and OCR_AVAILABLE:
            print("   🔍 Extracting math expressions from images...")
            base_results = self.base_extractor.extract_math_from_pdf_images(pdf_path)
            expressions = [expr.text for expr in base_results]
        else:
            print("   ⚠️ OCR not available, using simulated math expressions...")
            # For demo purposes, use some sample expressions
            expressions = [
                "f(x) = x²", 
                "∑(i=1 to n) x_i", 
                "E = mc²",
                "y = mx + b",
                "∫ x dx = x²/2 + C"
            ]
        
        if not expressions:
            return {
                "pdf_file": os.path.basename(pdf_path),
                "expressions_found": 0,
                "ai_analysis": {"analysis": "No mathematical expressions found for AI analysis"}
            }
        
        # AI Analysis
        print("   🤖 Analyzing expressions with AI...")
        ai_analysis = self.analyze_math_expressions_with_ai(expressions)
        
        # Learning recommendations
        print("   📚 Generating learning recommendations...")
        learning_recs = self.generate_math_learning_recommendations(expressions, ai_analysis)
        
        # Complexity analysis
        print("   📊 Analyzing complexity...")
        complexity_analysis = self.create_math_complexity_analysis(expressions)
        
        # Combine results
        enhanced_results = {
            "pdf_file": os.path.basename(pdf_path),
            "expressions_found": len(expressions),
            "expressions": expressions,
            "ai_analysis": ai_analysis,
            "learning_recommendations": learning_recs,
            "complexity_analysis": complexity_analysis,
            "enhanced_by": "OpenAI GPT-3.5-turbo"
        }
        
        # Save enhanced results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, "enhanced_math_analysis.json")
            with open(results_path, 'w') as f:
                json.dump(enhanced_results, f, indent=2)
            enhanced_results['saved_to'] = results_path
        
        return enhanced_results


def enhanced_extract_math_from_images(pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Main function for enhanced math extraction with AI analysis.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional output directory
        
    Returns:
        Enhanced extraction results with AI insights
    """
    extractor = EnhancedOCRMathExtractor()
    return extractor.enhanced_extract_math_from_pdf(pdf_path, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced math extraction with AI analysis")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output directory")
    
    args = parser.parse_args()
    
    # Enhanced extraction
    print(f"Enhanced Math Extraction: {args.pdf_path}")
    results = enhanced_extract_math_from_images(args.pdf_path, args.output)
    
    # Display results
    print(f"\nResults:")
    print(f"  Expressions found: {results['expressions_found']}")
    
    if results.get('ai_analysis', {}).get('overall_analysis'):
        analysis = results['ai_analysis']['overall_analysis']
        print(f"  Dominant field: {analysis.get('dominant_field', 'Unknown')}")
        print(f"  Complexity level: {analysis.get('complexity_level', 'Unknown')}")
        print(f"  Educational level: {analysis.get('educational_level', 'Unknown')}")
    
    if results.get('saved_to'):
        print(f"  Enhanced results saved to: {results['saved_to']}")