"""
Enhanced visualization module with OpenAI API integration for intelligent insights.
Provides AI-powered analysis and recommendations based on extracted data.
"""

import os
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Import base visualizer
from visualizer import PDFVisualizer

load_dotenv()


class EnhancedPDFVisualizer(PDFVisualizer):
    """Enhanced PDF visualizer with OpenAI-powered insights."""
    
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def generate_ai_insights(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate AI-powered insights about the document analysis.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            Dictionary of AI-generated insights
        """
        try:
            # Prepare data for AI analysis
            entity_summary = {}
            for entity_type, entities in analysis_results.get('entities', {}).items():
                entity_summary[entity_type] = {
                    'count': len(entities),
                    'examples': entities[:3]  # First 3 examples
                }
            
            prompt = f"""
            Analyze this PDF document data and provide intelligent insights:
            
            Document: {analysis_results.get('pdf_file', 'Unknown')}
            Pages: {analysis_results.get('total_pages', 0)}
            Summary: {analysis_results.get('summary', '')[:500]}
            
            Entities Found: {json.dumps(entity_summary, indent=2)}
            Math Expressions: {len(analysis_results.get('math_expressions', []))}
            
            Please provide:
            1. Document Type Classification (academic paper, business report, manual, etc.)
            2. Key Topics and Themes (3-5 main topics)
            3. Content Quality Assessment (complexity, technical level)
            4. Potential Use Cases (who would find this document useful)
            5. Data Insights (what the entities tell us about the document)
            6. Recommendations (how to better utilize this document)
            
            Format as JSON with keys: document_type, key_topics, quality_assessment, use_cases, data_insights, recommendations
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert document analyst. Provide detailed, professional insights about documents based on their extracted data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Try to parse as JSON, fallback to structured text
            try:
                insights = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # If not valid JSON, create structured response
                content = response.choices[0].message.content
                insights = {
                    "document_type": "Analysis provided",
                    "key_topics": "See full analysis",
                    "quality_assessment": "See full analysis", 
                    "use_cases": "See full analysis",
                    "data_insights": "See full analysis",
                    "recommendations": "See full analysis",
                    "full_analysis": content
                }
            
            return insights
            
        except Exception as e:
            return {
                "error": f"Could not generate AI insights: {str(e)}",
                "document_type": "Unknown",
                "key_topics": "Analysis unavailable",
                "quality_assessment": "Analysis unavailable",
                "use_cases": "Analysis unavailable", 
                "data_insights": "Analysis unavailable",
                "recommendations": "Analysis unavailable"
            }
    
    def create_ai_insights_chart(self, insights: Dict[str, str], output_path: str) -> str:
        """
        Create a visual representation of AI insights.
        
        Args:
            insights: AI-generated insights
            output_path: Path for output image
            
        Returns:
            Path to created chart
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AI-Powered Document Insights', fontsize=20, fontweight='bold')
            
            # Document Type (top-left)
            ax1.text(0.5, 0.5, f"Document Type:\n{insights.get('document_type', 'Unknown')}", 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title('Document Classification', fontweight='bold')
            
            # Key Topics (top-right)
            topics = insights.get('key_topics', 'Topics unavailable')
            if isinstance(topics, list):
                topics_text = '\n'.join([f"• {topic}" for topic in topics[:5]])
            else:
                topics_text = str(topics)[:200] + "..." if len(str(topics)) > 200 else str(topics)
            
            ax2.text(0.1, 0.9, "Key Topics:", fontsize=12, fontweight='bold')
            ax2.text(0.1, 0.1, topics_text, fontsize=10, va='bottom', ha='left', wrap=True)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Main Themes', fontweight='bold')
            
            # Quality Assessment (bottom-left)
            quality = str(insights.get('quality_assessment', 'Assessment unavailable'))
            quality_text = quality[:300] + "..." if len(quality) > 300 else quality
            
            ax3.text(0.1, 0.9, "Quality Assessment:", fontsize=12, fontweight='bold')
            ax3.text(0.1, 0.1, quality_text, fontsize=10, va='bottom', ha='left', wrap=True)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_title('Content Analysis', fontweight='bold')
            
            # Recommendations (bottom-right)
            recommendations = str(insights.get('recommendations', 'No recommendations available'))
            rec_text = recommendations[:300] + "..." if len(recommendations) > 300 else recommendations
            
            ax4.text(0.1, 0.9, "Recommendations:", fontsize=12, fontweight='bold')
            ax4.text(0.1, 0.1, rec_text, fontsize=10, va='bottom', ha='left', wrap=True)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('AI Recommendations', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"Error creating AI insights chart: {e}")
            return None
    
    def generate_entity_recommendations(self, entities: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Generate AI recommendations for each entity type found.
        
        Args:
            entities: Dictionary of entity types and their instances
            
        Returns:
            Dictionary of recommendations for each entity type
        """
        try:
            prompt = f"""
            Based on these named entities found in a document, provide actionable insights:
            
            {json.dumps({k: {'count': len(v), 'examples': v[:5]} for k, v in entities.items()}, indent=2)}
            
            For each entity type present, suggest:
            1. What this tells us about the document
            2. How these entities could be useful
            3. What actions someone could take with this information
            
            Provide concise, practical recommendations in JSON format with entity types as keys.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing named entities and providing actionable business insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            try:
                recommendations = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback to simple structure
                recommendations = {"general": response.choices[0].message.content}
            
            return recommendations
            
        except Exception as e:
            return {"error": f"Could not generate entity recommendations: {str(e)}"}
    
    def create_enhanced_dashboard(self, analysis_results: Dict, output_dir: str) -> Dict[str, str]:
        """
        Create an enhanced dashboard with AI insights.
        
        Args:
            analysis_results: Complete analysis results
            output_dir: Directory to save dashboard files
            
        Returns:
            Dictionary of created file paths
        """
        dashboard_files = {}
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate AI insights
            print("   🤖 Generating AI insights...")
            ai_insights = self.generate_ai_insights(analysis_results)
            
            # Create standard dashboard
            standard_files = super().create_analysis_dashboard(analysis_results, output_dir)
            dashboard_files.update(standard_files)
            
            # Create AI insights chart
            if not ai_insights.get('error'):
                insights_chart_path = os.path.join(output_dir, 'ai_insights.png')
                chart_path = self.create_ai_insights_chart(ai_insights, insights_chart_path)
                if chart_path:
                    dashboard_files['ai_insights_chart'] = chart_path
            
            # Generate entity recommendations
            if analysis_results.get('entities'):
                print("   📊 Generating entity recommendations...")
                entity_recs = self.generate_entity_recommendations(analysis_results['entities'])
                
                # Save recommendations as JSON
                rec_path = os.path.join(output_dir, 'entity_recommendations.json')
                with open(rec_path, 'w') as f:
                    json.dump(entity_recs, f, indent=2)
                dashboard_files['entity_recommendations'] = rec_path
            
            # Save AI insights as JSON
            insights_path = os.path.join(output_dir, 'ai_insights.json')
            with open(insights_path, 'w') as f:
                json.dump(ai_insights, f, indent=2)
            dashboard_files['ai_insights_data'] = insights_path
            
            return dashboard_files
            
        except Exception as e:
            print(f"Error creating enhanced dashboard: {e}")
            return dashboard_files


def enhanced_visualize_pdf_analysis(pdf_path: str, analysis_results: Dict, output_dir: str = None) -> Dict[str, str]:
    """
    Main function to create enhanced visual representations with AI insights.
    
    Args:
        pdf_path: Path to the original PDF
        analysis_results: Analysis results dictionary
        output_dir: Output directory (defaults to same dir as PDF)
        
    Returns:
        Dictionary of created visualization files
    """
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    
    visualizer = EnhancedPDFVisualizer()
    created_files = {}
    
    # Create standard visualizations
    print("   🎨 Creating standard visualizations...")
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    dashboard_dir = os.path.join(output_dir, f"{pdf_name}_enhanced_visuals")
    
    # Create enhanced dashboard with AI insights
    dashboard_files = visualizer.create_enhanced_dashboard(analysis_results, dashboard_dir)
    created_files.update(dashboard_files)
    
    return created_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create enhanced visual representations with AI insights")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("results_path", help="Path to the analysis results JSON file")
    parser.add_argument("-o", "--output", help="Output directory")
    
    args = parser.parse_args()
    
    # Load analysis results
    with open(args.results_path, 'r') as f:
        results = json.load(f)
    
    # Create enhanced visualizations
    created_files = enhanced_visualize_pdf_analysis(args.pdf_path, results, args.output)
    
    print("Created enhanced visualization files:")
    for file_type, file_path in created_files.items():
        print(f"  {file_type}: {file_path}")