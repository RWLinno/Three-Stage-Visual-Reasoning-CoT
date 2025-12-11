#!/usr/bin/env python3
"""
生成评估HTML报告 - 增强版
包括详细的统计信息、可视化结果和错误分析
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import base64

def load_results(jsonl_path: str) -> List[Dict[str, Any]]:
    """加载JSONL结果"""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def load_metrics(json_path: str) -> Dict[str, Any]:
    """加载评估指标"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def image_to_base64(image_path: str) -> str:
    """将图像转换为base64编码"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return ""

def generate_html_report(
    results: List[Dict[str, Any]], 
    metrics: Dict[str, Any],
    output_path: str,
    include_images: bool = False
) -> None:
    """生成HTML报告"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>洗衣机旋钮识别评估报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        
        .metric-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-card.success .value {{ color: #28a745; }}
        .metric-card.accuracy .value {{ color: #17a2b8; }}
        .metric-card.time .value {{ color: #ffc107; }}
        
        .section {{
            padding: 40px;
        }}
        
        .section-title {{
            font-size: 2em;
            margin-bottom: 30px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.95em;
        }}
        
        .results-table thead {{
            background: #667eea;
            color: white;
        }}
        
        .results-table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .results-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .results-table tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .status-icon {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .status-correct {{ color: #28a745; }}
        .status-incorrect {{ color: #dc3545; }}
        .status-unknown {{ color: #ffc107; }}
        .status-failed {{ color: #6c757d; }}
        
        .answer-cell {{
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .confidence-bar {{
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
            transition: width 0.3s ease;
        }}
        
        .image-preview {{
            max-width: 150px;
            max-height: 150px;
            border-radius: 5px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }}
        
        .image-preview:hover {{
            transform: scale(1.05);
        }}
        
        .filter-buttons {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            padding: 10px 20px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }}
        
        .filter-btn:hover, .filter-btn.active {{
            background: #667eea;
            color: white;
        }}
        
        .error-section {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .error-section h3 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        
        footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
        }}
        
        .config-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        .config-box pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
    <script>
        function filterResults(filterType) {{
            const rows = document.querySelectorAll('.results-table tbody tr');
            const buttons = document.querySelectorAll('.filter-btn');
            
            // 更新按钮状态
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // 过滤行
            rows.forEach(row => {{
                const status = row.dataset.status;
                if (filterType === 'all' || status === filterType) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}
        
        function sortTable(columnIndex) {{
            const table = document.querySelector('.results-table tbody');
            const rows = Array.from(table.querySelectorAll('tr'));
            
            rows.sort((a, b) => {{
                const aText = a.children[columnIndex].textContent.trim();
                const bText = b.children[columnIndex].textContent.trim();
                return aText.localeCompare(bText, 'zh-CN');
            }});
            
            rows.forEach(row => table.appendChild(row));
        }}
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 洗衣机旋钮识别评估报告</h1>
            <div class="subtitle">Bbox增强 - 三阶段CoT推理</div>
            <div class="subtitle" style="margin-top: 10px; opacity: 0.8;">
                生成时间: {metrics.get('timestamp', 'N/A')}
            </div>
        </header>
        
        <!-- 关键指标 -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">总样本数</div>
                <div class="value">{metrics.get('total_samples', 0)}</div>
            </div>
            
            <div class="metric-card success">
                <div class="label">成功处理</div>
                <div class="value">{metrics.get('successful', 0)}</div>
                <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                    {metrics.get('success_rate', 0):.1f}%
                </div>
            </div>
            
            <div class="metric-card accuracy">
                <div class="label">识别准确率</div>
                <div class="value">
                    {metrics.get('accuracy', 'N/A') if metrics.get('accuracy') is not None else 'N/A'}{'%' if metrics.get('accuracy') is not None else ''}
                </div>
                <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                    {metrics.get('correct_predictions', 0)}/{metrics.get('samples_with_gt', 0)} 有GT
                </div>
            </div>
            
            <div class="metric-card time">
                <div class="label">平均耗时</div>
                <div class="value">{metrics.get('average_processing_time', 0):.1f}s</div>
            </div>
            
            <div class="metric-card">
                <div class="label">平均置信度</div>
                <div class="value">{metrics.get('average_confidence', 0):.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="label">总耗时</div>
                <div class="value">{metrics.get('total_time', 0):.0f}s</div>
            </div>
        </div>
        
        <!-- 详细结果 -->
        <div class="section">
            <h2 class="section-title">详细结果</h2>
            
            <div class="filter-buttons">
                <button class="filter-btn active" onclick="filterResults('all')">全部 ({len(results)})</button>
                <button class="filter-btn" onclick="filterResults('correct')">
                    ✓ 正确 ({sum(1 for r in results if r.get('correct') is True)})
                </button>
                <button class="filter-btn" onclick="filterResults('incorrect')">
                    ✗ 错误 ({sum(1 for r in results if r.get('correct') is False)})
                </button>
                <button class="filter-btn" onclick="filterResults('unknown')">
                    ? 未知 ({sum(1 for r in results if r.get('correct') is None and r.get('success'))})
                </button>
                <button class="filter-btn" onclick="filterResults('failed')">
                    ⚠ 失败 ({sum(1 for r in results if not r.get('success'))})
                </button>
            </div>
            
            <table class="results-table">
                <thead>
                    <tr>
                        <th style="width: 5%;">状态</th>
                        <th style="width: 20%;" onclick="sortTable(1)">图像名称 ↕</th>
                        <th style="width: 15%;">Ground Truth</th>
                        <th style="width: 15%;">预测结果</th>
                        <th style="width: 10%;">置信度</th>
                        <th style="width: 10%;">耗时(s)</th>
                        <th style="width: 10%;">重试次数</th>
                        {'<th style="width: 15%;">预览</th>' if include_images else ''}
                    </tr>
                </thead>
                <tbody>
"""
    
    # 添加结果行
    for result in results:
        success = result.get('success', False)
        correct = result.get('correct')
        
        # 确定状态
        if not success:
            status = 'failed'
            status_icon = '<span class="status-icon status-failed">⚠</span>'
        elif correct is True:
            status = 'correct'
            status_icon = '<span class="status-icon status-correct">✓</span>'
        elif correct is False:
            status = 'incorrect'
            status_icon = '<span class="status-icon status-incorrect">✗</span>'
        else:
            status = 'unknown'
            status_icon = '<span class="status-icon status-unknown">?</span>'
        
        image_name = result.get('image_name', 'N/A')
        ground_truth = result.get('ground_truth', 'N/A') or 'N/A'
        predicted = result.get('predicted_answer', 'N/A') or result.get('error', 'Error')
        confidence = result.get('confidence', 0)
        processing_time = result.get('processing_time', 0)
        retry_count = result.get('retry_count', 0)
        
        # 图像预览
        image_preview = ''
        if include_images and result.get('image_path'):
            img_base64 = image_to_base64(result['image_path'])
            if img_base64:
                image_preview = f'<img src="data:image/png;base64,{img_base64}" class="image-preview" alt="{image_name}">'
        
        html_content += f"""
                    <tr data-status="{status}">
                        <td>{status_icon}</td>
                        <td title="{image_name}">{image_name[:30]}...</td>
                        <td>{ground_truth}</td>
                        <td class="answer-cell" title="{predicted}">{predicted}</td>
                        <td>
                            {confidence:.2f}
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence*100}%;"></div>
                            </div>
                        </td>
                        <td>{processing_time:.1f}</td>
                        <td>{retry_count}</td>
                        {'<td>' + image_preview + '</td>' if include_images else ''}
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
"""
    
    # 错误分析
    failed_results = [r for r in results if not r.get('success')]
    incorrect_results = [r for r in results if r.get('correct') is False]
    
    if failed_results or incorrect_results:
        html_content += """
        <div class="section">
            <h2 class="section-title">错误分析</h2>
"""
        
        if failed_results:
            html_content += f"""
            <div class="error-section">
                <h3>⚠ 处理失败 ({len(failed_results)} 个)</h3>
                <ul>
"""
            for r in failed_results[:10]:  # 只显示前10个
                html_content += f"                    <li><strong>{r.get('image_name', 'N/A')}</strong>: {r.get('error', 'Unknown error')}</li>\n"
            
            if len(failed_results) > 10:
                html_content += f"                    <li><em>... 还有 {len(failed_results) - 10} 个失败样本</em></li>\n"
            
            html_content += """
                </ul>
            </div>
"""
        
        if incorrect_results:
            html_content += f"""
            <div class="error-section">
                <h3>✗ 识别错误 ({len(incorrect_results)} 个)</h3>
                <ul>
"""
            for r in incorrect_results[:10]:  # 只显示前10个
                html_content += f"""                    <li>
                        <strong>{r.get('image_name', 'N/A')}</strong>: 
                        预测={r.get('predicted_answer', 'N/A')}, 
                        GT={r.get('ground_truth', 'N/A')}, 
                        置信度={r.get('confidence', 0):.2f}
                    </li>\n"""
            
            if len(incorrect_results) > 10:
                html_content += f"                    <li><em>... 还有 {len(incorrect_results) - 10} 个错误样本</em></li>\n"
            
            html_content += """
                </ul>
            </div>
"""
        
        html_content += """
        </div>
"""
    
    # 配置信息
    config = metrics.get('config', {})
    config_json = json.dumps(config, indent=2, ensure_ascii=False)
    
    html_content += f"""
        <div class="section">
            <h2 class="section-title">配置信息</h2>
            <div class="config-box">
                <pre>{config_json}</pre>
            </div>
        </div>
        
        <footer>
            <p><strong>TSVR-CoT</strong> - Three-Stage Visual Reasoning with Chain-of-Thought</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                生成于 {metrics.get('timestamp', 'N/A')}
            </p>
        </footer>
    </div>
</body>
</html>
"""
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='生成评估HTML报告')
    parser.add_argument('--results', type=str, required=True, help='results.jsonl文件路径')
    parser.add_argument('--metrics', type=str, required=True, help='eval_report.json文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出HTML文件路径')
    parser.add_argument('--include-images', action='store_true', help='是否在报告中包含图像预览')
    
    args = parser.parse_args()
    
    # 加载数据
    results = load_results(args.results)
    metrics = load_metrics(args.metrics)
    
    # 生成报告
    generate_html_report(results, metrics, args.output, args.include_images)

if __name__ == '__main__':
    main()

