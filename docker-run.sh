#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Create necessary directories
mkdir -p input output

# Function to display usage
show_usage() {
    echo -e "${BLUE}🤖 AI-Enhanced PDF Analyzer - Docker Helper Script${NC}"
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                   Build Docker images"
    echo "  analyze [PDF_FILE]      Basic analysis"
    echo "  advanced [PDF_FILE]     Complete analysis with all features"
    echo "  api                     Start API server"
    echo "  demo                    Run demo analysis"
    echo "  clean                   Clean up Docker resources"
    echo "  logs [SERVICE]          Show container logs"
    echo "  status                  Show container status"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 analyze document.pdf"
    echo "  $0 advanced document.pdf"
    echo "  $0 api"
    echo "  $0 demo"
}

# Function to build Docker image
build_image() {
    check_docker
    print_status "Building Docker images..."
    docker-compose build --no-cache
    print_status "Build completed successfully!"
}

# Function to start API server
start_api() {
    check_docker
    print_status "Starting API server..."
    docker-compose up -d api-server
    print_status "API server started on http://localhost:8000"
    print_info "📚 Documentation: http://localhost:8000/docs"
    print_info "🤖 Advanced endpoint: POST /analyze-advanced"
}

# Function to analyze PDF
analyze_pdf() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ]; then
        print_error "Please provide a PDF file name"
        echo "Usage: $0 analyze <pdf_file>"
        exit 1
    fi
    
    if [ ! -f "input/$pdf_file" ]; then
        print_error "PDF file 'input/$pdf_file' not found"
        print_warning "Please place your PDF in the 'input' directory"
        exit 1
    fi
    
    check_docker
    print_status "Analyzing $pdf_file..."
    docker run --rm \
        -v "$(pwd)/input:/app/input:ro" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/.env:/app/.env:ro" \
        ai-enhanced-pdf-analyzer-pdf-analyzer \
        python pdf_analyzer.py "/app/input/$pdf_file"
}

# Function for complete advanced analysis
advanced_analysis() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ]; then
        print_error "Please provide a PDF file name"
        echo "Usage: $0 advanced <pdf_file>"
        exit 1
    fi
    
    if [ ! -f "input/$pdf_file" ]; then
        print_error "PDF file 'input/$pdf_file' not found"
        exit 1
    fi
    
    check_docker
    print_status "Running advanced analysis on $pdf_file..."
    docker run --rm \
        -v "$(pwd)/input:/app/input:ro" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/.env:/app/.env:ro" \
        ai-enhanced-pdf-analyzer-pdf-analyzer \
        python pdf_analyzer.py "/app/input/$pdf_file" --ai-enhanced --visualize --ocr-math --format detailed
}

# Function to analyze with visualizations
analyze_with_visualizations() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ]; then
        echo "❌ Error: Please specify a PDF file"
        echo "Usage: $0 visualize <pdf_file>"
        return 1
    fi
    
    if [ ! -f "./input/$pdf_file" ]; then
        echo "❌ Error: PDF file './input/$pdf_file' not found"
        echo "Please place your PDF in the ./input/ directory"
        return 1
    fi
    
    echo "🎨 Analyzing PDF with visualizations: $pdf_file"
    docker run --rm \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/.env:/app/.env" \
        ai-pdf-analyzer \
        python pdf_analyzer.py "/app/input/$pdf_file" \
        --visualize \
        --output-dir "/app/output" \
        -o "/app/output/analysis_results.json"
}

# Function for OCR analysis
analyze_with_ocr() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ]; then
        echo "❌ Error: Please specify a PDF file"
        echo "Usage: $0 ocr <pdf_file>"
        return 1
    fi
    
    if [ ! -f "./input/$pdf_file" ]; then
        echo "❌ Error: PDF file './input/$pdf_file' not found"
        echo "Please place your PDF in the ./input/ directory"
        return 1
    fi
    
    echo "👁️ Analyzing PDF with OCR math extraction: $pdf_file"
    docker run --rm \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/.env:/app/.env" \
        ai-pdf-analyzer \
        python pdf_analyzer.py "/app/input/$pdf_file" \
        --ocr-math \
        --output-dir "/app/output" \
        -o "/app/output/analysis_results.json"
}

# Function for AI-enhanced analysis
ai_enhanced_analysis() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ]; then
        echo "❌ Error: Please specify a PDF file"
        echo "Usage: $0 ai-enhanced <pdf_file>"
        return 1
    fi
    
    if [ ! -f "./input/$pdf_file" ]; then
        echo "❌ Error: PDF file './input/$pdf_file' not found"
        echo "Please place your PDF in the ./input/ directory"
        return 1
    fi
    
    echo "🤖 AI-enhanced analysis: $pdf_file"
    docker run --rm \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/.env:/app/.env" \
        ai-pdf-analyzer \
        python pdf_analyzer.py "/app/input/$pdf_file" \
        --ai-enhanced \
        --output-dir "/app/output" \
        -o "/app/output/analysis_results.json"
}

# Function for complete advanced analysis
advanced_analysis() {
    local pdf_file="$1"
    if [ -z "$pdf_file" ]; then
        echo "❌ Error: Please specify a PDF file"
        echo "Usage: $0 advanced <pdf_file>"
        return 1
    fi
    
    if [ ! -f "./input/$pdf_file" ]; then
        echo "❌ Error: PDF file './input/$pdf_file' not found"
        echo "Please place your PDF in the ./input/ directory"
        return 1
    fi
    
    echo "🚀 Complete AI-enhanced analysis with all features: $pdf_file"
    docker run --rm \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/.env:/app/.env" \
        ai-pdf-analyzer \
        python pdf_analyzer.py "/app/input/$pdf_file" \
        --visualize \
        --ocr-math \
        --ai-enhanced \
        --format detailed \
        --output-dir "/app/output" \
        -o "/app/output/complete_analysis.json"
}

# Function to clean up containers and images
cleanup() {
    echo "🧹 Cleaning up containers and images..."
    docker stop ai-pdf-analyzer-api ai-pdf-analyzer-cli 2>/dev/null || true
    docker rm ai-pdf-analyzer-api ai-pdf-analyzer-cli 2>/dev/null || true
    docker rmi ai-pdf-analyzer 2>/dev/null || true
    echo "✅ Cleanup complete!"
}

# Function to show logs
show_logs() {
    local container="$1"
    if [ -z "$container" ]; then
        container="ai-pdf-analyzer-api"
    fi
    
    echo "📋 Showing logs for container: $container"
    docker logs -f "$container"
}

# Function to run demo
run_demo() {
    check_docker
    print_status "Running demo analysis..."
    # Copy sample PDF if it exists
    if [ -f "2507.21964v1.pdf" ]; then
        cp "2507.21964v1.pdf" "input/"
        docker run --rm \
            -v "$(pwd)/input:/app/input:ro" \
            -v "$(pwd)/output:/app/output" \
            -v "$(pwd)/.env:/app/.env:ro" \
            ai-enhanced-pdf-analyzer-pdf-analyzer \
            python pdf_analyzer.py "/app/input/2507.21964v1.pdf" --ai-enhanced --visualize
    else
        print_error "Demo PDF not found. Please place a PDF in the input directory first."
    fi
}

# Function to clean up
cleanup() {
    check_docker
    print_status "Cleaning up Docker resources..."
    docker-compose down -v
    docker system prune -f
    print_status "Cleanup completed!"
}

# Function to show logs
show_logs() {
    docker-compose logs -f "${1:-api-server}"
}

# Function to show status
show_status() {
    docker-compose ps
}

# Main command handling
case "${1:-}" in
    "build")
        build_image
        ;;
    "analyze")
        if [ -z "$2" ]; then
            print_error "Please provide a PDF file name"
            echo "Usage: $0 analyze <pdf_file>"
            exit 1
        fi
        analyze_pdf "$2"
        ;;
    "advanced")
        if [ -z "$2" ]; then
            print_error "Please provide a PDF file name"
            echo "Usage: $0 advanced <pdf_file>"
            exit 1
        fi
        advanced_analysis "$2"
        ;;
    "api")
        start_api
        ;;
    "demo")
        run_demo
        ;;
    "clean")
        cleanup
        ;;
    "logs")
        show_logs "$2"
        ;;
    "status")
        show_status
        ;;
    "")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac