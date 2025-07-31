# Setup Instructions

## Quick Start (Docker - Recommended)

1. **Navigate to project directory:**
```bash
cd ai-enhanced-pdf-analyzer
```

2. **Build and start:**
```bash
chmod +x docker-run.sh
./docker-run.sh build
./docker-run.sh api
```

3. **Access the application:**
- Web API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Local Installation (Alternative)

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env to add your OpenAI API key
```

3. **Run analysis:**
```bash
python pdf_analyzer.py sample_pdfs/1706.03762v7.pdf
```

## Testing the Application

### Docker Method
```bash
# Copy a PDF to input folder
cp your-document.pdf input/

# Run analysis
./docker-run.sh advanced your-document.pdf
```

### Local Method
```bash
python pdf_analyzer.py sample_pdfs/1706.03762v7.pdf --ai-enhanced --visualize
```

## Features Demonstrated

✅ PDF Text Extraction  
✅ Mathematical Expression Detection  
✅ Named Entity Recognition (NER)  
✅ Document Summarization  
✅ CLI Interface  
✅ FastAPI Web Server  
✅ Visual Representations  
✅ OCR Math Extraction  
✅ Docker Support  

## Output Files

Results are saved in the `output/` directory:
- JSON analysis files
- Highlighted PDFs
- Visualization charts
- AI insights

## Troubleshooting

If you encounter any issues:
1. Make sure Docker is running
2. Check that all files are in the correct directories
3. Verify Python dependencies are installed
4. Review logs with `./docker-run.sh logs`