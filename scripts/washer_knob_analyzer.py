#!/usr/bin/env python3
"""
洗衣机旋钮分析器 - 改进版
支持批量处理、并发执行、结果保存等功能
"""
import sys
import os
import argparse
import json
import logging
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import traceback
import datetime

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from src.base.vlm_agent import VLMAgentEAS
from src.base.cot_engine import CoTEngine

# 设置日志
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='洗衣机旋钮状态分析器 - 批量处理版')
    
    # 输入配置
    parser.add_argument('--image_dir', type=str, required=True,
                       help='输入图像目录路径')
    parser.add_argument('--question', type=str, required=True,
                       help='输入问题文本')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--output_jsonl', type=str, required=True,
                       help='结果JSONL文件路径')
    parser.add_argument('--save_intermediate_images', type=lambda x: x.lower() == 'true', default=True,
                       help='是否保存中间推理图像')
    
    # API配置
    parser.add_argument('--eas_url', type=str, 
                       default=os.getenv('EAS_URL', 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation'),
                       help='EAS服务URL')
    parser.add_argument('--eas_token', type=str, 
                       default=os.getenv('EAS_TOKEN', ''),
                       help='EAS认证token')
    parser.add_argument('--model_name', type=str, 
                       default=os.getenv('MODEL_NAME', 'Qwen3-VL-235B-A22B-Instruct-FP8'),
                       help='模型名称')
    
    # 性能配置
    parser.add_argument('--num_processors', type=int, default=int(os.getenv('NUM_PROCESSORS', '4')),
                       help='并发处理器数量')
    parser.add_argument('--batch_size', type=int, default=int(os.getenv('BATCH_SIZE', '2')),
                       help='批处理大小')
    parser.add_argument('--max_tokens', type=int, default=int(os.getenv('MAX_TOKENS', '512')),
                       help='最大生成token数')
    parser.add_argument('--timeout', type=int, default=int(os.getenv('TIMEOUT', '120')),
                       help='请求超时时间（秒）')
    
    # 其他配置
    parser.add_argument('--log_level', type=str, default=os.getenv('LOG_LEVEL', 'INFO'),
                       help='日志级别')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    return parser.parse_args()

def get_image_files(image_dir: str) -> List[str]:
    """获取目录中的所有图像文件"""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                image_files.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    return image_files

def process_single_image(args: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """处理单个图像"""
    image_path, config = args
    
    image_name = Path(image_path).name
    result = {
        'image_path': image_path,
        'image_name': image_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'success': False,
        'error': None
    }
    
    try:
        # 初始化VLM Agent (每次处理都重新初始化，避免进程间共享问题)
        vlm_agent = VLMAgentEAS(
            base_url=config['eas_url'],
            token=config['eas_token'],
            model_name=config['model_name'],
            max_tokens=config['max_tokens'],
            timeout=config['timeout']
        )
        
        # 初始化CoT引擎
        cot_engine = CoTEngine(
            vlm_agent, 
            task_type="washer_knob",
            question=config.get('question', 'Determine the current knob position')
        )
        
        # 执行CoT推理
        start_time = time.time()
        cot_results = cot_engine.reason(
            image_path=image_path,
            depth_path=None  # 目前不处理深度图
        )
        processing_time = time.time() - start_time
        
        # 保存中间图像 (如果启用)
        if config['save_intermediate_images']:
            from src.utils.visualization import save_intermediate_images
            intermediate_dir = Path(config['output_dir']) / 'intermediate_images'
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            
            save_intermediate_images(
                image_path=image_path,
                results=cot_results,
                output_dir=str(intermediate_dir),
                image_name=image_name
            )
        
        # 整合结果
        result.update({
            'success': True,
            'processing_time': round(processing_time, 2),
            'stage1_rules': cot_results.get('stage1_rules', ''),
            'stage2_answer': cot_results.get('stage2_answer', ''),
            'stage3_validation': cot_results.get('stage3_validation', ''),
            'final_answer': cot_results.get('final_answer', ''),
            'confidence': round(cot_results.get('confidence', 0.0), 2),
            'raw_responses': cot_results.get('raw_responses', {})
        })
        
        logger.info(f"成功处理 {image_name}: {result['final_answer']} (置信度: {result['confidence']})")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"处理 {image_name} 失败: {error_msg}")
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        result.update({
            'error': error_msg,
            'stack_trace': traceback.format_exc(),
            'processing_time': round(time.time() - start_time, 2) if 'start_time' in locals() else 0
        })
    
    return result

def process_images_batch(image_paths: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """批量处理图像"""
    results = []
    num_processors = min(config['num_processors'], len(image_paths))
    
    logger.info(f"开始批量处理 {len(image_paths)} 个图像，使用 {num_processors} 个处理器")
    
    # 使用进程池进行并发处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors) as executor:
        # 提交所有任务
        future_to_image = {
            executor.submit(process_single_image, (image_path, config)): image_path
            for image_path in image_paths
        }
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result = future.result()
                results.append(result)
                
                # 实时保存结果
                if config['output_jsonl']:
                    with open(config['output_jsonl'], 'a', encoding='utf-8') as f:
                        # Extract short answer from final_answer
                        short_answer = result.get('final_answer', '')
                        if '<answer>' in short_answer and '</answer>' in short_answer:
                            import re
                            match = re.search(r'<answer>(.*?)</answer>', short_answer, re.DOTALL)
                            if match:
                                short_answer = match.group(1).strip()
                        else:
                            # Extract from last line or after common patterns
                            lines = short_answer.split('\n')
                            for line in reversed(lines):
                                line = line.strip()
                                if line and not line.startswith('#') and not line.startswith('**'):
                                    short_answer = line
                                    break
                        
                        result['answer'] = short_answer
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                processed = len(results)
                total = len(image_paths)
                if processed % max(1, total // 10) == 0 or processed == total:
                    logger.info(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"处理 {image_path} 时发生异常: {e}")
                logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e),
                    'stack_trace': traceback.format_exc()
                })
    
    logger.info(f"批量处理完成，成功: {sum(1 for r in results if r.get('success'))}/{len(results)}")
    return results

def main() -> None:
    """主函数"""
    args = parse_args()
    logger = setup_logging(args.log_level, f"{args.output_dir}/logs/processing.log")
    
    # 验证输入目录
    if not os.path.exists(args.image_dir):
        logger.error(f"输入目录不存在: {args.image_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    if args.save_intermediate_images:
        os.makedirs(os.path.join(args.output_dir, 'intermediate_images'), exist_ok=True)
    
    # 获取图像文件
    image_files = get_image_files(args.image_dir)
    if not image_files:
        logger.error("没有找到图像文件")
        return
    
    # 配置参数
    config = {
        'eas_url': args.eas_url,
        'eas_token': args.eas_token,
        'model_name': args.model_name,
        'max_tokens': args.max_tokens,
        'timeout': args.timeout,
        'output_dir': args.output_dir,
        'output_jsonl': args.output_jsonl,
        'save_intermediate_images': args.save_intermediate_images,
        'num_processors': args.num_processors,
        'batch_size': args.batch_size,
        'question': args.question
    }
    
    # 清空或创建输出文件
    if os.path.exists(args.output_jsonl):
        logger.info(f"清空现有结果文件: {args.output_jsonl}")
        open(args.output_jsonl, 'w').close()
    
    # 处理图像
    start_time = time.time()
    results = process_images_batch(image_files, config)
    total_time = time.time() - start_time
    
    # 保存最终统计
    success_count = sum(1 for r in results if r.get('success'))
    avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('success')) / max(success_count, 1)
    avg_time = total_time / len(image_files) if image_files else 0
    
    # 创建安全的配置副本，移除敏感信息
    safe_config = {
        'eas_url': config['eas_url'],
        'model_name': config['model_name'],
        'max_tokens': config['max_tokens'],
        'timeout': config['timeout'],
        'output_dir': config['output_dir'],
        'save_intermediate_images': config['save_intermediate_images'],
        'num_processors': config['num_processors'],
        'batch_size': config['batch_size'],
        'question': config['question']
        # 注意：不包含eas_token
    }
    
    summary = {
        'total_images': len(image_files),
        'success_count': success_count,
        'success_rate': round(success_count / len(image_files) * 100, 2) if image_files else 0,
        'average_confidence': round(avg_confidence, 2),
        'average_processing_time': round(avg_time, 2),
        'total_time': round(total_time, 2),
        'timestamp': datetime.datetime.now().isoformat(),
        'config': safe_config
    }
    
    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("处理完成!")
    logger.info(f"成功处理: {success_count}/{len(image_files)}")
    logger.info(f"平均置信度: {avg_confidence:.2f}")
    logger.info(f"总耗时: {total_time:.2f}秒，平均: {avg_time:.2f}秒/图像")
    logger.info(f"结果保存到: {args.output_jsonl}")
    logger.info(f"摘要保存到: {summary_file}")

if __name__ == "__main__":
    main()