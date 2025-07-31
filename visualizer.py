"""
Visual representation module for PDF analysis results.
Highlights named entities and mathematical expressions in PDFs.
"""

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import numpy as np
from typing import Dict, List, Tuple
import json
import os


class PDFVisualizer:
    """Visualizes analysis results by highlighting entities and math expressions in PDFs."""
    
    def __init__(self):
        # Color scheme for different entity types
        self.entity_colors = {
            'PERSON': '#FF6B6B',      # Red
            'ORG': '#4ECDC4',         # Teal
            'GPE': '#45B7D1',         # Blue
            'MONEY': '#96CEB4',       # Green
            'DATE': '#FFEAA7',        # Yellow
            'CARDINAL': '#DDA0DD',    # Plum
            'ORDINAL': '#F4A460',     # Sandy Brown
            'TIME': '#FFB6C1',        # Light Pink
            'PERCENT': '#98FB98',     # Pale Green
            'PRODUCT': '#F0E68C',     # Khaki
            'EVENT': '#DEB887',       # Burlywood
            'FAC': '#B0C4DE',         # Light Steel Blue
            'LAW': '#F5DEB3',         # Wheat
            'LANGUAGE': '#E6E6FA',    # Lavender
            'NORP': '#FFA07A',        # Light Salmon
            'LOC': '#87CEEB',         # Sky Blue
            'WORK_OF_ART': '#DDA0DD', # Plum
            'QUANTITY': '#F0F8FF',    # Alice Blue
            'MATH': '#FF4500'         # Orange Red for math expressions
        }
    
    def create_highlighted_pdf(self, pdf_path: str, analysis_results: Dict, output_path: str) -> str:
        """
        Creates a new PDF with highlighted entities and math expressions.
        
        Args:
            pdf_path: Path to original PDF
            analysis_results: Results from PDF analysis
            output_path: Path for output PDF
            
        Returns:
            Path to the highlighted PDF
        """
        try:
            # Open the original PDF
            doc = fitz.open(pdf_path)
            
            # Extract text with position information
            text_with_positions = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_dict = page.get_text("dict")
                text_with_positions.append(text_dict)
            
            # Highlight entities
            self._highlight_entities(doc, text_with_positions, analysis_results.get('entities', {}))
            
            # Highlight math expressions
            if analysis_results.get('math_expressions'):
                self._highlight_math_expressions(doc, text_with_positions, analysis_results['math_expressions'])
            
            # Save the highlighted PDF
            doc.save(output_path)
            doc.close()
            
            return output_path
            
        except Exception as e:
            print(f"Error creating highlighted PDF: {e}")
            return None
    
    def _highlight_entities(self, doc: fitz.Document, text_positions: List, entities: Dict[str, List[str]]):
        """Highlights named entities in the PDF."""
        for entity_type, entity_list in entities.items():
            color = self.entity_colors.get(entity_type, '#CCCCCC')
            
            for entity in entity_list:
                if len(entity.strip()) < 2:  # Skip very short entities
                    continue
                    
                # Search for entity in each page
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Find all instances of the entity
                    text_instances = page.search_for(entity)
                    
                    for inst in text_instances:
                        # Add highlight annotation
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=to_rgb(color))
                        highlight.update()
    
    def _highlight_math_expressions(self, doc: fitz.Document, text_positions: List, math_expressions: List[str]):
        """Highlights mathematical expressions in the PDF."""
        color = self.entity_colors['MATH']
        
        for expression in math_expressions:
            if len(expression.strip()) < 2:
                continue
                
            # Search for expression in each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Find all instances of the expression
                text_instances = page.search_for(expression)
                
                for inst in text_instances:
                    # Add highlight annotation
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=to_rgb(color))
                    highlight.update()
    
    def create_entity_summary_chart(self, entities: Dict[str, List[str]], output_path: str) -> str:
        """
        Creates a bar chart showing entity type distribution.
        
        Args:
            entities: Dictionary of entity types and their instances
            output_path: Path for the output chart image
            
        Returns:
            Path to the created chart
        """
        try:
            # Prepare data
            entity_types = list(entities.keys())
            entity_counts = [len(entities[entity_type]) for entity_type in entity_types]
            
            # Create the chart
            plt.figure(figsize=(12, 8))
            bars = plt.bar(entity_types, entity_counts, 
                          color=[self.entity_colors.get(et, '#CCCCCC') for et in entity_types])
            
            # Customize the chart
            plt.title('Named Entity Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Entity Types', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, entity_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"Error creating entity chart: {e}")
            return None
    
    def create_analysis_dashboard(self, analysis_results: Dict, output_dir: str) -> Dict[str, str]:
        """
        Creates a comprehensive visual dashboard of analysis results.
        
        Args:
            analysis_results: Complete analysis results
            output_dir: Directory to save dashboard files
            
        Returns:
            Dictionary of created file paths
        """
        dashboard_files = {}
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Entity distribution chart
            if analysis_results.get('entities'):
                entity_chart_path = os.path.join(output_dir, 'entity_distribution.png')
                chart_path = self.create_entity_summary_chart(analysis_results['entities'], entity_chart_path)
                if chart_path:
                    dashboard_files['entity_chart'] = chart_path
            
            # 2. Summary statistics visualization
            stats_chart_path = os.path.join(output_dir, 'analysis_stats.png')
            stats_path = self._create_stats_chart(analysis_results, stats_chart_path)
            if stats_path:
                dashboard_files['stats_chart'] = stats_path
            
            # 3. Math expressions visualization (if any)
            if analysis_results.get('math_expressions'):
                math_chart_path = os.path.join(output_dir, 'math_expressions.png')
                math_path = self._create_math_expressions_chart(analysis_results['math_expressions'], math_chart_path)
                if math_path:
                    dashboard_files['math_chart'] = math_path
            
            return dashboard_files
            
        except Exception as e:
            print(f"Error creating analysis dashboard: {e}")
            return {}
    
    def _create_stats_chart(self, analysis_results: Dict, output_path: str) -> str:
        """Creates a summary statistics chart."""
        try:
            # Prepare statistics
            stats = {
                'Total Pages': analysis_results.get('total_pages', 0),
                'Total Entities': analysis_results.get('total_entities', 0),
                'Math Expressions': analysis_results.get('total_math_expressions', 0),
                'Entity Types': len(analysis_results.get('entities', {})),
                'Summary Words': len(analysis_results.get('summary', '').split())
            }
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = list(stats.keys())
            values = list(stats.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars = ax.barh(categories, values, color=colors)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                       str(value), va='center', fontweight='bold')
            
            ax.set_title('Document Analysis Statistics', fontsize=16, fontweight='bold')
            ax.set_xlabel('Count', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"Error creating stats chart: {e}")
            return None
    
    def _create_math_expressions_chart(self, math_expressions: List[str], output_path: str) -> str:
        """Creates a visualization of mathematical expressions."""
        try:
            # Create a simple list visualization
            fig, ax = plt.subplots(figsize=(12, max(6, len(math_expressions) * 0.5)))
            
            y_positions = range(len(math_expressions))
            
            # Create horizontal bars
            bars = ax.barh(y_positions, [1] * len(math_expressions), 
                          color=self.entity_colors['MATH'], alpha=0.7)
            
            # Add expression text
            for i, expr in enumerate(math_expressions):
                # Truncate long expressions
                display_expr = expr if len(expr) <= 50 else expr[:47] + "..."
                ax.text(0.5, i, display_expr, va='center', ha='center', 
                       fontweight='bold', fontsize=10)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"Expr {i+1}" for i in y_positions])
            ax.set_xlabel('Mathematical Expressions Found')
            ax.set_title('Detected Mathematical Expressions', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 1)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"Error creating math expressions chart: {e}")
            return None


def visualize_pdf_analysis(pdf_path: str, analysis_results: Dict, output_dir: str = None) -> Dict[str, str]:
    """
    Main function to create visual representations of PDF analysis.
    
    Args:
        pdf_path: Path to the original PDF
        analysis_results: Analysis results dictionary
        output_dir: Output directory (defaults to same dir as PDF)
        
    Returns:
        Dictionary of created visualization files
    """
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    
    visualizer = PDFVisualizer()
    created_files = {}
    
    # Create highlighted PDF
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    highlighted_pdf_path = os.path.join(output_dir, f"{pdf_name}_highlighted.pdf")
    
    highlighted_pdf = visualizer.create_highlighted_pdf(pdf_path, analysis_results, highlighted_pdf_path)
    if highlighted_pdf:
        created_files['highlighted_pdf'] = highlighted_pdf
    
    # Create dashboard
    dashboard_dir = os.path.join(output_dir, f"{pdf_name}_visuals")
    dashboard_files = visualizer.create_analysis_dashboard(analysis_results, dashboard_dir)
    created_files.update(dashboard_files)
    
    return created_files


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Create visual representations of PDF analysis")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("results_path", help="Path to the analysis results JSON file")
    parser.add_argument("-o", "--output", help="Output directory")
    
    args = parser.parse_args()
    
    # Load analysis results
    with open(args.results_path, 'r') as f:
        results = json.load(f)
    
    # Create visualizations
    created_files = visualize_pdf_analysis(args.pdf_path, results, args.output)
    
    print("Created visualization files:")
    for file_type, file_path in created_files.items():
        print(f"  {file_type}: {file_path}")