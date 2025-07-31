"""
Generate PDF report from markdown
"""

import subprocess
import os
from datetime import datetime

def generate_pdf():
    """Generate PDF from markdown report"""
    
    print("Generating PDF report...")
    
    # Input and output files
    md_file = "README.md"
    pdf_file = f"Koushik_Cruz_AI_PDF_Analyzer_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    # CSS for better formatting
    css_content = """
    body {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #34495e;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 8px;
        margin-top: 30px;
    }
    
    h3 {
        color: #7f8c8d;
        margin-top: 20px;
    }
    
    code {
        background-color: #f4f4f4;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    
    pre {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        overflow-x: auto;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        padding-left: 15px;
        color: #555;
        margin: 15px 0;
    }
    
    ul, ol {
        padding-left: 30px;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    
    th {
        background-color: #3498db;
        color: white;
    }
    
    .page-break {
        page-break-after: always;
    }
    """
    
    # Write CSS file
    with open("report_style.css", "w") as f:
        f.write(css_content)
    
    try:
        # Check if pandoc is installed
        result = subprocess.run(["pandoc", "--version"], capture_output=True)
        if result.returncode != 0:
            print("Error: pandoc is not installed!")
            print("Install with: brew install pandoc")
            return
        
        # Generate PDF using pandoc
        cmd = [
            "pandoc",
            md_file,
            "-o", pdf_file,
            "--pdf-engine=pdflatex",
            "--css=report_style.css",
            "--toc",
            "--toc-depth=2",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "documentclass=article",
            "-V", "colorlinks=true",
            "-V", "linkcolor=blue",
            "-V", "urlcolor=blue"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ PDF report generated successfully: {pdf_file}")
            print(f"📄 File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
        else:
            print(f"Error generating PDF: {result.stderr}")
            
            # Fallback: Try with different engine
            print("\nTrying alternative method...")
            cmd[3] = "--pdf-engine=xelatex"
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"\n✅ PDF report generated successfully: {pdf_file}")
            else:
                print("\nFallback: Trying wkhtmltopdf...")
                # Try wkhtmltopdf as last resort
                subprocess.run([
                    "pandoc", md_file, "-t", "html5", "-o", "temp_report.html",
                    "--css=report_style.css", "--self-contained"
                ])
                
                if os.path.exists("temp_report.html"):
                    print("✅ HTML report generated: temp_report.html")
                    print("You can open this in a browser and save as PDF")
    
    except FileNotFoundError:
        print("Error: pandoc is not installed!")
        print("\nInstallation instructions:")
        print("- macOS: brew install pandoc")
        print("- Ubuntu: sudo apt-get install pandoc")
        print("- Windows: Download from https://pandoc.org/installing.html")
        
        # Alternative: Generate HTML
        print("\nGenerating HTML report instead...")
        try:
            import markdown
            with open(md_file, 'r') as f:
                md_content = f.read()
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI-Enhanced PDF Analyzer Report - Koushik Cruz</title>
    <style>{css_content}</style>
</head>
<body>
{markdown.markdown(md_content, extensions=['tables', 'fenced_code'])}
</body>
</html>
"""
            
            html_file = pdf_file.replace('.pdf', '.html')
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            print(f"✅ HTML report generated: {html_file}")
            print("Open this file in a browser and use 'Print to PDF' to create PDF")
            
        except ImportError:
            print("Markdown library not available. Install with: pip install markdown")
    
    finally:
        # Clean up CSS file
        if os.path.exists("report_style.css"):
            os.remove("report_style.css")

if __name__ == "__main__":
    generate_pdf()