"""
FastAPI Server for AI-Enhanced PDF Document Analyzer
==================================================

Provides RESTful API endpoints for PDF analysis with bonus features:
- Standard analysis endpoint
- Advanced analysis with AI features
- File upload handling
- Comprehensive error handling

Endpoints:
- GET  /health - Health check
- POST /analyze - Standard PDF analysis
- POST /analyze-advanced - AI-enhanced analysis with bonus features
- GET  /docs - API documentation
"""

import os
import tempfile
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pdf_analyzer import PDFAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="AI-Enhanced PDF Document Analyzer",
    description="Advanced PDF analysis with AI-powered insights, visual representations, and OCR capabilities",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize analyzer
analyzer = PDFAnalyzer()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI-Enhanced PDF Document Analyzer API",
        "version": "2.0",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze",
            "analyze_advanced": "POST /analyze-advanced",
            "docs": "GET /docs"
        },
        "features": [
            "PDF text extraction",
            "Mathematical expression detection",
            "Named Entity Recognition",
            "Document summarization",
            "Visual representations",
            "OCR math extraction",
            "AI-powered insights"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "analyzer_ready": True,
        "version": "2.0"
    }


@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Standard PDF analysis endpoint.
    
    Args:
        file: PDF file to analyze
        
    Returns:
        - summary: Document summary (100-150 words)
        - entities: Named entities by category
        - math_expressions: List of mathematical expressions
        - metadata: Page count and other info
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Analyze the PDF
        results = analyzer.analyze_pdf(tmp_path)
        
        # Clean up results for API response
        response = {
            "filename": file.filename,
            "total_pages": results["total_pages"],
            "summary": results["summary"],
            "entities": {
                "types": results["entities"],
                "total_count": results["total_entities"]
            },
            "math_expressions": {
                "expressions": results["math_expressions"][:10],  # First 10
                "total_count": results["total_math_expressions"]
            },
            "analysis_type": "standard"
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/analyze-advanced")
async def analyze_pdf_advanced(
    file: UploadFile = File(...),
    enable_visualization: bool = False,
    enable_ocr_math: bool = False,
    ai_enhanced: bool = True
) -> Dict[str, Any]:
    """
    Advanced PDF analysis with AI features and bonus capabilities.
    
    Parameters:
        - file: PDF file to analyze
        - enable_visualization: Create visual representations
        - enable_ocr_math: Extract math from images using OCR
        - ai_enhanced: Use AI for enhanced analysis and insights
        
    Returns:
        - Complete analysis results including bonus features
        - AI-powered insights and recommendations
        - Visual files information (if enabled)
        - OCR results (if enabled)
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Analyze the PDF
        results = analyzer.analyze_pdf(tmp_path)
        
        # Bonus Feature: OCR Math Extraction
        if enable_ocr_math:
            try:
                if ai_enhanced:
                    from enhanced_ocr_extractor import enhanced_extract_math_from_images
                    output_dir = os.path.dirname(tmp_path)
                    ocr_results = enhanced_extract_math_from_images(tmp_path, output_dir)
                    
                    if ocr_results.get('expressions'):
                        results['math_expressions'].extend(ocr_results['expressions'])
                        results['total_math_expressions'] += len(ocr_results['expressions'])
                        results['enhanced_ocr_results'] = ocr_results
                else:
                    from ocr_math_extractor import extract_math_from_images
                    output_dir = os.path.dirname(tmp_path)
                    ocr_results = extract_math_from_images(tmp_path, output_dir)
                    
                    if ocr_results['expressions']:
                        ocr_expressions = [expr['text'] for expr in ocr_results['expressions']]
                        results['math_expressions'].extend(ocr_expressions)
                        results['total_math_expressions'] += len(ocr_expressions)
                        results['ocr_math_results'] = ocr_results
                        
            except ImportError:
                results['ocr_error'] = "OCR dependencies not available"
            except Exception as e:
                results['ocr_error'] = f"OCR extraction failed: {str(e)}"
        
        # Bonus Feature: Visual Representations
        visual_files = {}
        if enable_visualization:
            try:
                if ai_enhanced:
                    from enhanced_visualizer import enhanced_visualize_pdf_analysis
                    output_dir = os.path.dirname(tmp_path)
                    visual_files = enhanced_visualize_pdf_analysis(tmp_path, results, output_dir)
                else:
                    from visualizer import visualize_pdf_analysis
                    output_dir = os.path.dirname(tmp_path)
                    visual_files = visualize_pdf_analysis(tmp_path, results, output_dir)
                    
                results['visualization_files'] = visual_files
                
            except ImportError:
                results['visualization_error'] = "Visualization dependencies not available"
            except Exception as e:
                results['visualization_error'] = f"Visualization failed: {str(e)}"
        
        # Clean up results for API response
        response = {
            "filename": file.filename,
            "total_pages": results["total_pages"],
            "summary": results["summary"],
            "entities": {
                "types": results["entities"],
                "total_count": results["total_entities"]
            },
            "math_expressions": {
                "expressions": results["math_expressions"],
                "total_count": results["total_math_expressions"]
            },
            "bonus_features": {
                "visualization_enabled": enable_visualization,
                "ocr_math_enabled": enable_ocr_math,
                "ai_enhanced": ai_enhanced,
                "visual_files": visual_files,
                "ocr_results": results.get('enhanced_ocr_results', results.get('ocr_math_results', {}))
            },
            "analysis_type": "advanced"
        }
        
        # Include AI insights if available
        if ai_enhanced and results.get('enhanced_ocr_results'):
            response['ai_insights'] = {
                "math_analysis": results['enhanced_ocr_results'].get('ai_analysis', {}),
                "learning_recommendations": results['enhanced_ocr_results'].get('learning_recommendations', {})
            }
        
        # Include any error messages
        if 'ocr_error' in results:
            response['bonus_features']['ocr_error'] = results['ocr_error']
        if 'visualization_error' in results:
            response['bonus_features']['visualization_error'] = results['visualization_error']
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/features")
async def get_features():
    """Get information about available features."""
    return {
        "core_features": [
            "PDF text extraction",
            "Mathematical expression detection",
            "Named Entity Recognition (15+ types)",
            "Document summarization (100-150 words)",
            "CLI interface",
            "Clean, modular code",
            "Comprehensive documentation"
        ],
        "bonus_features": [
            "Visual representations with entity highlighting",
            "OCR math extraction from embedded images",
            "Docker containerization"
        ],
        "ai_enhancements": [
            "Intelligent document classification",
            "Educational insights and learning recommendations",
            "Mathematical expression analysis and complexity assessment",
            "Contextual recommendations and practical applications",
            "AI-powered visualizations and insights"
        ],
        "supported_formats": ["PDF"],
        "api_endpoints": [
            "/analyze - Standard analysis",
            "/analyze-advanced - AI-enhanced analysis with bonus features"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI-Enhanced PDF Analyzer API Server...")
    print("📚 Features: PDF analysis, AI insights, visual representations, OCR")
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("📊 Standard analysis: POST /analyze")
    print("🤖 Advanced AI analysis: POST /analyze-advanced")
    print("❤️ Health check: GET /health")
    uvicorn.run(app, host="0.0.0.0", port=8000)