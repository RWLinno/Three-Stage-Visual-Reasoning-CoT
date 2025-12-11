#!/usr/bin/env python3
"""
HTML Report Generator - Generate visual reports for washing machine knob state recognition results
"""
import os
import json
import argparse
import logging
import datetime
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Washing Machine Knob State Recognition HTML Report')
    
    parser.add_argument(
        '--results_jsonl',
        type=str,
        required=True,
        help='Results JSONL file path'
    )
    
    parser.add_argument(
        '--output_html',
        type=str,
        default='./output/report.html',
        help='Output HTML file path'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Original image directory (for displaying original images)'
    )
    
    parser.add_argument(
        '--intermediate_dir',
        type=str,
        default='',
        help='Intermediate image directory (for displaying reasoning process)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()

def load_results(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load JSONL results file"""
    results = []
    try:
        if not os.path.exists(jsonl_path):
            logger.warning(f"Results file does not exist: {jsonl_path}")
            return results
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing error: {e}, skipping line: {line.strip()}")
        logger.info(f"Successfully loaded {len(results)} results")
    except Exception as e:
        logger.error(f"Failed to load results file: {e}")
        logger.error(f"Stack trace:\n{e.__traceback__}")
    return results

def generate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistics"""
    stats = {
        'total_count': len(results),
        'success_count': sum(1 for r in results if r.get('success', False)),
        'failure_count': sum(1 for r in results if not r.get('success', False)),
        'avg_confidence': 0.0,
        'answer_distribution': Counter(),
        'processing_time_stats': {},
        'error_types': Counter()
    }
    
    # Calculate average confidence
    confidences = [r.get('confidence', 0.0) for r in results if r.get('success', False)]
    if confidences:
        stats['avg_confidence'] = sum(confidences) / len(confidences)
    
    # Answer distribution
    for result in results:
        if result.get('success', False):
            final_answer = result.get('final_answer', 'Unknown').strip()
            stats['answer_distribution'][final_answer] += 1
    
    # Processing time statistics
    processing_times = [r.get('processing_time', 0) for r in results if r.get('success', False)]
    if processing_times:
        stats['processing_time_stats'] = {
            'min': min(processing_times),
            'max': max(processing_times),
            'avg': sum(processing_times) / len(processing_times),
            'median': sorted(processing_times)[len(processing_times) // 2]
        }
    
    # Error type statistics
    for result in results:
        if not result.get('success', False) and result.get('error'):
            error_type = result['error'].split(':')[0].strip()
            stats['error_types'][error_type] += 1
    
    return stats

def create_confidence_chart(confidences: List[float], output_path: str):
    """Create confidence distribution chart"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Draw histogram
        plt.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add average line
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        plt.axvline(x=avg_conf, color='red', linestyle='--', label=f'Average Confidence: {avg_conf:.2f}')
        
        plt.title('Confidence Distribution', fontsize=14)
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Sample Count', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save chart
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return img_base64
    except Exception as e:
        logger.error(f"Failed to create confidence chart: {e}")
        return None

def create_answer_distribution_chart(distribution: Counter, output_path: str):
    """Create answer distribution chart"""
    try:
        if not distribution:
            return None
        
        # Show only top 10 most common answers
        top_answers = dict(distribution.most_common(10))
        
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        answers = list(top_answers.keys())
        counts = list(top_answers.values())
        
        y_pos = range(len(answers))
        plt.barh(y_pos, counts, color='lightgreen', edgecolor='black', alpha=0.8)
        
        # Use index numbers for answers to avoid Chinese font issues
        plt.yticks(y_pos, [f"Mode {i+1}" for i in range(len(answers))], fontsize=10)
        plt.xlabel('Frequency', fontsize=12)
        plt.title('Answer Distribution (Top 10)', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, count in enumerate(counts):
            plt.text(count + 0.5, i, str(count), va='center')
        
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return img_base64
    except Exception as e:
        logger.error(f"Failed to create answer distribution chart: {e}")
        return None

def create_processing_time_chart(times: List[float], output_path: str):
    """Create processing time chart"""
    try:
        if not times:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Create line chart
        plt.plot(range(1, len(times) + 1), times, 'b-', marker='o', markersize=4, alpha=0.7)
        
        # Add average line
        avg_time = sum(times) / len(times)
        plt.axhline(y=avg_time, color='red', linestyle='--', label=f'Average Time: {avg_time:.2f}s')
        
        plt.title('Processing Time Trend', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save chart
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return img_base64
    except Exception as e:
        logger.error(f"Failed to create processing time chart: {e}")
        return None

def generate_html_report(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_path: str,
    image_dir: str = '',
    intermediate_dir: str = ''
):
    """Generate HTML report"""
    try:
        # Create temporary directory for charts
        temp_dir = Path(output_path).parent / 'temp_charts'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate charts
        confidences = [r.get('confidence', 0.0) for r in results if r.get('success', False)]
        confidence_chart = create_confidence_chart(confidences, str(temp_dir / 'confidence_chart.png')) if confidences else None
        
        answer_chart = create_answer_distribution_chart(stats['answer_distribution'], str(temp_dir / 'answer_chart.png')) if stats['answer_distribution'] else None
        
        processing_times = [r.get('processing_time', 0) for r in results if r.get('success', False)]
        time_chart = create_processing_time_chart(processing_times, str(temp_dir / 'time_chart.png')) if processing_times else None
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Washing Machine Knob State Recognition Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f7fa;
                }}
                .header {{
                    text-align: center;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .stats-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                .chart-title {{
                    font-size: 1.2em;
                    margin-bottom: 15px;
                    color: #444;
                }}
                .results-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 25px;
                }}
                .result-card {{
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    overflow: hidden;
                    transition: transform 0.3s ease;
                }}
                .result-card:hover {{
                    transform: translateY(-5px);
                }}
                .result-header {{
                    padding: 15px;
                    background: #667eea;
                    color: white;
                    font-weight: bold;
                }}
                .result-content {{
                    padding: 15px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 10px 0;
                }}
                .result-image {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .answer-highlight {{
                    font-size: 1.3em;
                    font-weight: bold;
                    color: #28a745;
                    margin: 10px 0;
                    padding: 8px;
                    background: #e8f5e9;
                    border-radius: 4px;
                }}
                .confidence-badge {{
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 0.9em;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .high-confidence {{
                    background: #28a745;
                    color: white;
                }}
                .medium-confidence {{
                    background: #ffc107;
                    color: #212529;
                }}
                .low-confidence {{
                    background: #dc3545;
                    color: white;
                }}
                .stages-container {{
                    margin-top: 15px;
                }}
                .stage {{
                    margin-bottom: 10px;
                    padding: 10px;
                    border-left: 3px solid #667eea;
                    background: #f8f9fa;
                }}
                .stage-title {{
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #495057;
                }}
                .error-card {{
                    background: #fff0f0;
                    border-left: 3px solid #dc3545;
                }}
                .error-message {{
                    color: #dc3545;
                    font-family: monospace;
                    white-space: pre-wrap;
                    max-height: 200px;
                    overflow: auto;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
                .filter-container {{
                    margin: 20px 0;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .filter-controls {{
                    display: flex;
                    gap: 15px;
                    flex-wrap: wrap;
                }}
                select, input {{
                    padding: 8px 12px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                }}
                button {{
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background 0.3s;
                }}
                button:hover {{
                    background: #5a6fd8;
                }}
                @media (max-width: 768px) {{
                    .results-container {{
                        grid-template-columns: 1fr;
                    }}
                    .stats-container {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌀 Washing Machine Knob State Recognition Report</h1>
                <p>Three-Stage Chain-of-Thought Visual Reasoning System</p>
                <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">{stats['total_count']}</div>
                    <div class="stat-label">Total Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['success_count']}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['failure_count']}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['avg_confidence']:.2f}</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>

            {f'''
            <div class="chart-container">
                <div class="chart-title">📊 Confidence Distribution</div>
                <img src="data:image/png;base64,{confidence_chart}" alt="Confidence Distribution" style="max-width: 100%;">
            </div>
            ''' if confidence_chart else ''}

            {f'''
            <div class="chart-container">
                <div class="chart-title">📈 Answer Distribution (Top 10)</div>
                <img src="data:image/png;base64,{answer_chart}" alt="Answer Distribution" style="max-width: 100%;">
            </div>
            ''' if answer_chart else ''}

            {f'''
            <div class="chart-container">
                <div class="chart-title">⏱️ Processing Time Trend</div>
                <img src="data:image/png;base64,{time_chart}" alt="Processing Time Trend" style="max-width: 100%;">
            </div>
            ''' if time_chart else ''}

            <div class="filter-container">
                <div class="filter-controls">
                    <div>
                        <label for="confidence-filter">Min Confidence:</label>
                        <select id="confidence-filter" onchange="filterResults()">
                            <option value="0">0.0</option>
                            <option value="0.3">0.3</option>
                            <option value="0.5" selected>0.5</option>
                            <option value="0.7">0.7</option>
                            <option value="0.9">0.9</option>
                        </select>
                    </div>
                    <div>
                        <label for="success-filter">Status:</label>
                        <select id="success-filter" onchange="filterResults()">
                            <option value="all">All</option>
                            <option value="success" selected>Success</option>
                            <option value="failure">Failure</option>
                        </select>
                    </div>
                    <div>
                        <label for="search-input">Search Answer:</label>
                        <input type="text" id="search-input" placeholder="Enter keywords..." oninput="filterResults()">
                    </div>
                    <button onclick="resetFilters()">Reset Filters</button>
                </div>
            </div>

            <div class="results-container" id="results-container">
                {generate_result_cards(results, image_dir, intermediate_dir)}
            </div>

            <div class="footer">
                <p>Washing Machine Knob State Recognition System v1.0 | Three-Stage CoT Reasoning</p>
                <p>© 2025 Visual Analysis Team | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <script>
                function filterResults() {{
                    const minConfidence = parseFloat(document.getElementById('confidence-filter').value);
                    const successFilter = document.getElementById('success-filter').value;
                    const searchText = document.getElementById('search-input').value.toLowerCase();
                    
                    const cards = document.querySelectorAll('.result-card');
                    
                    cards.forEach(card => {{
                        const confidence = parseFloat(card.dataset.confidence || '0');
                        const isSuccess = card.dataset.success === 'true';
                        const answer = card.dataset.answer?.toLowerCase() || '';
                        
                        let show = true;
                        
                        if (confidence < minConfidence) show = false;
                        if (successFilter === 'success' && !isSuccess) show = false;
                        if (successFilter === 'failure' && isSuccess) show = false;
                        if (searchText && !answer.includes(searchText)) show = false;
                        
                        card.style.display = show ? 'block' : 'none';
                    }});
                }}
                
                function resetFilters() {{
                    document.getElementById('confidence-filter').value = '0.5';
                    document.getElementById('success-filter').value = 'success';
                    document.getElementById('search-input').value = '';
                    filterResults();
                }}
                
                // Initial filter
                document.addEventListener('DOMContentLoaded', filterResults);
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        
        # Clean up temporary charts
        for file in temp_dir.glob('*.png'):
            file.unlink()
        temp_dir.rmdir()
        
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        logger.error(f"Stack trace:\n{e.__traceback__}")

def generate_result_cards(results: List[Dict[str, Any]], image_dir: str, intermediate_dir: str) -> str:
    """Generate result cards HTML"""
    cards_html = []
    
    for i, result in enumerate(results):
        image_name = result.get('image_name', f'image_{i}')
        image_path = result.get('image_path', '')
        success = result.get('success', False)
        final_answer = result.get('final_answer', 'Unknown')
        confidence = result.get('confidence', 0.0)
        
        # Determine confidence style
        if confidence >= 0.8:
            confidence_class = 'high-confidence'
        elif confidence >= 0.5:
            confidence_class = 'medium-confidence'
        else:
            confidence_class = 'low-confidence'
        
        # Get image paths
        original_image_path = ''
        stage1_image_path = ''
        stage2_image_path = ''
        stage3_image_path = ''
        
        if image_dir and image_path:
            # Original image
            original_image_name = Path(image_path).name if image_path else image_name
            original_image_path = os.path.join(image_dir, original_image_name)
            
            if os.path.exists(original_image_path):
                try:
                    with open(original_image_path, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    original_image_path = f"data:image/jpeg;base64,{img_base64}"
                except Exception as e:
                    logger.warning(f"Failed to read original image {original_image_path}: {e}")
                    original_image_path = ''
        
        if intermediate_dir and success:
            # Intermediate stage images
            base_name = Path(image_name).stem
            
            stage1_path = os.path.join(intermediate_dir, f"{base_name}_stage1_rules.jpg")
            stage2_path = os.path.join(intermediate_dir, f"{base_name}_stage2_answer.jpg")
            stage3_path = os.path.join(intermediate_dir, f"{base_name}_stage3_validation.jpg")
            
            if os.path.exists(stage1_path):
                try:
                    with open(stage1_path, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    stage1_image_path = f"data:image/jpeg;base64,{img_base64}"
                except:
                    stage1_image_path = ''
            
            if os.path.exists(stage2_path):
                try:
                    with open(stage2_path, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    stage2_image_path = f"data:image/jpeg;base64,{img_base64}"
                except:
                    stage2_image_path = ''
            
            if os.path.exists(stage3_path):
                try:
                    with open(stage3_path, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    stage3_image_path = f"data:image/jpeg;base64,{img_base64}"
                except:
                    stage3_image_path = ''
        
        if success:
            # Success result card
            card_html = f'''
            <div class="result-card" data-confidence="{confidence}" data-success="true" data-answer="{final_answer}">
                <div class="result-header">
                    {image_name} | Confidence: {confidence:.2f}
                </div>
                <div class="result-content">
                    <div class="image-container">
                        {f'<img src="{original_image_path}" class="result-image" alt="Original Image">' if original_image_path else '<p>Original image unavailable</p>'}
                    </div>
                    
                    <div class="answer-highlight">
                        Final Answer: {final_answer}
                        <span class="confidence-badge {confidence_class}">{confidence:.2f}</span>
                    </div>
                    
                    <div class="stages-container">
                        <div class="stage">
                            <div class="stage-title">Stage 1: Rule Extraction</div>
                            <div>{result.get('stage1_rules', 'No rule information')[:200]}...</div>
                            {f'<div class="image-container"><img src="{stage1_image_path}" class="result-image" alt="Rule Extraction Visualization"></div>' if stage1_image_path else ''}
                        </div>
                        
                        <div class="stage">
                            <div class="stage-title">Stage 2: Application Reasoning</div>
                            <div>Initial Answer: {result.get('stage2_answer', 'None')}</div>
                            {f'<div class="image-container"><img src="{stage2_image_path}" class="result-image" alt="Application Reasoning Visualization"></div>' if stage2_image_path else ''}
                        </div>
                        
                        <div class="stage">
                            <div class="stage-title">Stage 3: Validation</div>
                            <div>{result.get('stage3_validation', 'No validation information')}</div>
                            {f'<div class="image-container"><img src="{stage3_image_path}" class="result-image" alt="Validation Visualization"></div>' if stage3_image_path else ''}
                        </div>
                    </div>
                    
                    <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        Processing Time: {result.get('processing_time', 0):.2f}s
                    </div>
                </div>
            </div>
            '''
        else:
            # Failure result card
            error_msg = result.get('error', 'Unknown error')
            card_html = f'''
            <div class="result-card error-card" data-success="false">
                <div class="result-header">
                    {image_name} | Processing Failed
                </div>
                <div class="result-content">
                    <div class="image-container">
                        {f'<img src="{original_image_path}" class="result-image" alt="Original Image">' if original_image_path else '<p>Original image unavailable</p>'}
                    </div>
                    
                    <div style="color: #dc3545; font-weight: bold; margin: 10px 0;">
                        ❌ Processing Failed
                    </div>
                    
                    <div class="error-message">
                        {error_msg[:200]}...
                    </div>
                    
                    <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        Processing Time: {result.get('processing_time', 0):.2f}s
                    </div>
                </div>
            </div>
            '''
        
        cards_html.append(card_html)
    
    return '\n'.join(cards_html)

def main():
    """Main function"""
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load results
        results = load_results(args.results_jsonl)
        
        if not results:
            logger.warning("No valid results found, generating empty report")
            # Create empty report
            empty_stats = {
                'total_count': 0,
                'success_count': 0,
                'failure_count': 0,
                'avg_confidence': 0.0,
                'answer_distribution': Counter(),
                'processing_time_stats': {}
            }
            generate_html_report(
                results=[],
                stats=empty_stats,
                output_path=args.output_html,
                image_dir=args.image_dir,
                intermediate_dir=args.intermediate_dir
            )
            return
        
        # Generate statistics
        stats = generate_statistics(results)
        
        # Generate HTML report
        generate_html_report(
            results=results,
            stats=stats,
            output_path=args.output_html,
            image_dir=args.image_dir,
            intermediate_dir=args.intermediate_dir
        )
        
        logger.info("HTML report generation complete!")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        logger.error(f"Stack trace:\n{e.__traceback__}")
        raise

if __name__ == "__main__":
    main()
