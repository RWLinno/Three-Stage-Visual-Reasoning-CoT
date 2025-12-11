#!/usr/bin/env python3
"""
Rotary Control Recognition Evaluator - Bbox Enhanced Version
Generic geometric reasoning without task-specific priors
Single-process sequential processing with graceful interruption support
"""
import sys
import os
import argparse
import json
import logging
import time
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import traceback
import datetime

# Add src path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.base.vlm_agent import VLMAgentEAS
from src.base.cot_engine import CoTEngine
from src.utils.prompt_templates import TaskTemplates
from src.utils.prompt_templates_bbox import BboxEnhancedTemplates

# Global flag for graceful shutdown
interrupted = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global interrupted
    interrupted = True
    print("\n\n" + "="*80)
    print("⚠️  Interrupt signal received. Finishing current sample and saving results...")
    print("="*80 + "\n")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Rotary Control Recognition Evaluator - Bbox Enhanced')
    
    # Input configuration
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Dataset root directory (containing with_status and without_status subdirs)')
    parser.add_argument('--question', type=str, default='What is the current position of the control?',
                       help='Question text to ask')
    parser.add_argument('--use_bbox', action='store_true', default=True,
                       help='Use bbox information for enhanced prompts')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory path')
    parser.add_argument('--save_intermediate_images', action='store_true', default=True,
                       help='Save intermediate reasoning images')
    
    # API configuration
    parser.add_argument('--eas_url', type=str, 
                       default=os.getenv('EAS_URL', ''),
                       help='EAS service URL')
    parser.add_argument('--eas_token', type=str, 
                       default=os.getenv('EAS_TOKEN', ''),
                       help='EAS authentication token')
    parser.add_argument('--model_name', type=str, 
                       default=os.getenv('MODEL_NAME', 'Qwen3-VL-235B-A22B-Instruct-FP8'),
                       help='Model name')
    
    # Performance configuration
    parser.add_argument('--max_tokens', type=int, default=int(os.getenv('MAX_TOKENS', '4096')),
                       help='Maximum generation tokens (4096 recommended for 3-stage reasoning)')
    parser.add_argument('--timeout', type=int, default=int(os.getenv('TIMEOUT', '300')),
                       help='Request timeout in seconds')
    
    # Evaluation configuration
    parser.add_argument('--subset', type=str, choices=['with_status', 'without_status', 'both'], 
                       default='both',
                       help='Which subset to evaluate')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples per subset (for quick testing)')
    
    # Other configuration
    parser.add_argument('--log_level', type=str, default=os.getenv('LOG_LEVEL', 'INFO'),
                       help='Logging level')
    
    return parser.parse_args()

def load_dataset_samples(dataset_dir: str, subset: str, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Load dataset samples
    
    Returns:
        List of (image_path, json_path) tuples
    """
    samples = []
    subset_dir = Path(dataset_dir) / subset
    
    if not subset_dir.exists():
        logger.warning(f"Subset directory does not exist: {subset_dir}")
        return samples
    
    # Find all PNG files
    png_files = sorted(subset_dir.glob("*.png"))
    
    for png_file in png_files:
        # Corresponding JSON file
        json_file = png_file.with_suffix('.json')
        
        if json_file.exists():
            samples.append((str(png_file), str(json_file)))
        else:
            logger.warning(f"Corresponding JSON file not found: {json_file}")
    
    logger.info(f"Loaded {len(samples)} samples from {subset}")
    
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples for testing")
    
    return samples

def process_single_sample(
    image_path: str, 
    json_path: str, 
    config: Dict[str, Any],
    vlm_agent: VLMAgentEAS
) -> Dict[str, Any]:
    """Process a single sample"""
    image_name = Path(image_path).name
    result = {
        'image_path': image_path,
        'image_name': image_name,
        'json_path': json_path,
        'timestamp': datetime.datetime.now().isoformat(),
        'success': False,
        'error': None,
        'ground_truth': None,
        'predicted_answer': None,
        'correct': None
    }
    
    start_time = time.time()
    
    try:
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            knob_data = json.load(f)
        
        # Extract ground truth (if available)
        ground_truth = BboxEnhancedTemplates.extract_ground_truth(knob_data)
        result['ground_truth'] = ground_truth
        
        # Choose reasoning strategy based on bbox usage
        if config['use_bbox']:
            # Bbox-enhanced CoT reasoning
            cot_results = reason_with_bbox_enhancement(
                vlm_agent=vlm_agent,
                image_path=image_path,
                knob_data=knob_data,
                question=config['question']
            )
        else:
            # Standard CoT reasoning
            cot_engine = CoTEngine(
                vlm_agent, 
                task_type="washer_knob",
                question=config['question']
            )
            cot_results = cot_engine.reason(image_path=image_path)
        
        processing_time = time.time() - start_time
        
        # Extract final answer
        final_answer = cot_results.get('final_answer', '')
        predicted_answer = extract_clean_answer(final_answer)
        result['predicted_answer'] = predicted_answer
        
        # Check correctness (if ground truth available)
        if ground_truth:
            result['correct'] = compare_answers(predicted_answer, ground_truth)
        
        # Save intermediate images (if enabled)
        if config['save_intermediate_images']:
            try:
                from src.utils.visualization import save_intermediate_images
                intermediate_dir = Path(config['output_dir']) / 'intermediate_images'
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                
                save_intermediate_images(
                    image_path=image_path,
                    results=cot_results,
                    output_dir=str(intermediate_dir),
                    image_name=image_name
                )
            except Exception as e:
                logger.warning(f"Failed to save intermediate images: {e}")
        
        # Consolidate results
        result.update({
            'success': True,
            'processing_time': round(processing_time, 2),
            'stage1_rules': cot_results.get('stage1_rules', ''),  # Keep full reasoning
            'stage2_answer': cot_results.get('stage2_answer', ''),
            'stage3_validation': cot_results.get('stage3_validation', ''),  # Keep full validation
            'confidence': round(cot_results.get('confidence', 0.0), 2),
            'retry_count': len(cot_results.get('retry_history', []))
        })
        
        status = "✓" if result.get('correct') else "✗" if result.get('correct') is False else "?"
        logger.info(f"{status} {image_name}: pred={predicted_answer}, gt={ground_truth}, time={processing_time:.1f}s")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to process {image_name}: {error_msg}")
        if config.get('log_level') == 'DEBUG':
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
        result.update({
            'error': error_msg,
            'processing_time': round(time.time() - start_time, 2)
        })
    
    return result

def reason_with_bbox_enhancement(
    vlm_agent: VLMAgentEAS,
    image_path: str,
    knob_data: Dict[str, Any],
    question: str
) -> Dict[str, Any]:
    """Perform CoT reasoning with bbox enhancement"""
    
    results = {
        'stage1_rules': '',
        'stage2_answer': '',
        'stage3_validation': '',
        'final_answer': '',
        'confidence': 0.0,
        'raw_responses': {},
        'retry_history': []
    }
    
    # Stage 1: Bbox-enhanced rule extraction
    stage1_prompt = BboxEnhancedTemplates.create_stage1_prompt_with_bbox(question, knob_data)
    
    stage1_response = vlm_agent.inference_single(
        prompt_text=stage1_prompt,
        image_path=image_path,
        max_retries=3
    )
    results['stage1_rules'] = stage1_response
    results['raw_responses']['stage1'] = stage1_response
    
    # Log stage1 response length to detect truncation
    if len(stage1_response) > 0:
        logger.info(f"Stage1 response length: {len(stage1_response)} chars")
        if len(stage1_response) < 500:
            logger.warning(f"Stage1 response seems too short, may be truncated or incomplete")
    
    # Stage 2: Apply rules
    template = BboxEnhancedTemplates.get_generic_rotary_template_with_bbox()
    stage2_prompt = template['stage2'].format(rules=stage1_response)
    
    stage2_response = vlm_agent.inference_single(
        prompt_text=stage2_prompt,
        image_path=image_path,
        max_retries=3
    )
    results['stage2_answer'] = stage2_response
    results['raw_responses']['stage2'] = stage2_response
    
    # Log stage2 response length
    logger.info(f"Stage2 response length: {len(stage2_response)} chars")
    
    # Extract answer
    extracted_answer = TaskTemplates.extract_answer_tag(stage2_response)
    if extracted_answer:
        results['stage2_answer'] = extracted_answer
    
    # Stage 3: Validation
    modes = knob_data.get('modes', [])
    adjacent_modes = ', '.join(modes[:5]) if modes else ""
    
    stage3_prompt = template['stage3'].format(
        answer=results['stage2_answer'],
        adjacent_modes=adjacent_modes
    )
    
    stage3_response = vlm_agent.inference_single(
        prompt_text=stage3_prompt,
        image_path=image_path,
        max_retries=3
    )
    results['stage3_validation'] = stage3_response
    results['raw_responses']['stage3'] = stage3_response
    
    # Log stage3 response length
    logger.info(f"Stage3 response length: {len(stage3_response)} chars")
    
    # Synthesize final answer
    results['final_answer'] = results['stage2_answer']
    results['confidence'] = 0.8  # Base confidence with bbox enhancement
    
    # Adjust confidence based on validation
    if 'valid' in stage3_response.lower():
        results['confidence'] = min(0.95, results['confidence'] + 0.1)
    elif 'invalid' in stage3_response.lower():
        results['confidence'] = max(0.5, results['confidence'] - 0.2)
    
    return results

def extract_clean_answer(answer_text: str) -> str:
    """Extract clean answer from response text"""
    import re
    
    # Try extracting from <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', answer_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Remove quotes
    answer_text = answer_text.strip()
    if answer_text.startswith('"') and answer_text.endswith('"'):
        answer_text = answer_text[1:-1]
    if answer_text.startswith("'") and answer_text.endswith("'"):
        answer_text = answer_text[1:-1]
    
    # If answer is too long, try extracting last meaningful line
    if len(answer_text) > 100:
        lines = answer_text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('**') and not line.startswith('-'):
                return line
    
    return answer_text.strip()

def compare_answers(predicted: str, ground_truth: str) -> bool:
    """Compare predicted answer with ground truth"""
    # Normalize
    pred_clean = predicted.strip().lower().replace(' ', '').replace("'", '').replace('"', '')
    gt_clean = ground_truth.strip().lower().replace(' ', '').replace("'", '').replace('"', '')
    
    # Exact match
    if pred_clean == gt_clean:
        return True
    
    # Containment
    if pred_clean in gt_clean or gt_clean in pred_clean:
        return True
    
    return False

def process_samples_sequential(
    samples: List[Tuple[str, str]], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Process samples sequentially (single process) with interrupt support"""
    global interrupted
    
    results = []
    total = len(samples)
    
    logger.info(f"Starting sequential processing of {total} samples")
    logger.info("Press Ctrl+C to interrupt at any time\n")
    
    # Initialize VLM agent once (reuse for all samples)
    try:
        vlm_agent = VLMAgentEAS(
            base_url=config['eas_url'],
            token=config['eas_token'],
            model_name=config['model_name'],
            max_tokens=config['max_tokens'],
            timeout=config['timeout']
        )
    except Exception as e:
        logger.error(f"Failed to initialize VLM agent: {e}")
        return results
    
    # Result files for chunked saving (10 items per file)
    results_dir = Path(config['output_dir']) / 'results_chunks'
    results_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 10
    
    # Process each sample
    for idx, (img_path, json_path) in enumerate(samples, 1):
        # Check if interrupted
        if interrupted:
            logger.warning(f"\nProcessing interrupted after {idx-1}/{total} samples")
            break
        
        # Progress indicator
        print(f"\n[{idx}/{total}] Processing: {Path(img_path).name}")
        print("-" * 80)
        
        # Process sample
        try:
            result = process_single_sample(img_path, json_path, config, vlm_agent)
            results.append(result)
            
            # Save result to chunked file (10 items per file)
            chunk_idx = (idx - 1) // chunk_size
            chunk_file = results_dir / f'results_chunk_{chunk_idx:04d}.jsonl'
            
            # Open in append mode for current chunk
            with open(chunk_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # Log chunk transition
            if idx % chunk_size == 1:
                logger.info(f"Starting new chunk file: {chunk_file.name}")
            elif idx % chunk_size == 0:
                logger.info(f"Completed chunk file: {chunk_file.name} ({chunk_size} items)")
            
            # Progress summary
            success_count = sum(1 for r in results if r.get('success'))
            if result.get('ground_truth'):
                correct_count = sum(1 for r in results if r.get('correct') is True)
                with_gt = sum(1 for r in results if r.get('ground_truth') is not None)
                accuracy = correct_count / with_gt * 100 if with_gt > 0 else 0
                print(f"Progress: {idx}/{total} | Success: {success_count}/{idx} | Accuracy: {correct_count}/{with_gt} ({accuracy:.1f}%)")
            else:
                print(f"Progress: {idx}/{total} | Success: {success_count}/{idx}")
            
        except KeyboardInterrupt:
            # Double Ctrl+C for immediate exit
            logger.warning("\n⚠️  Force interrupted! Saving results...")
            interrupted = True
            break
        except Exception as e:
            logger.error(f"Unexpected error processing {Path(img_path).name}: {e}")
            results.append({
                'image_path': img_path,
                'image_name': Path(img_path).name,
                'success': False,
                'error': str(e)
            })
    
    logger.info(f"\nSequential processing {'interrupted' if interrupted else 'completed'}")
    logger.info(f"Processed: {len(results)}/{total} samples")
    
    return results

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate evaluation metrics"""
    total = len(results)
    success = sum(1 for r in results if r.get('success'))
    
    # Only count accuracy for samples with ground truth
    with_gt = [r for r in results if r.get('ground_truth') is not None]
    correct = sum(1 for r in with_gt if r.get('correct') is True)
    
    avg_time = sum(r.get('processing_time', 0) for r in results if r.get('success')) / max(success, 1)
    avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('success')) / max(success, 1)
    
    metrics = {
        'total_samples': total,
        'successful': success,
        'failed': total - success,
        'success_rate': round(success / total * 100, 2) if total > 0 else 0,
        'samples_with_gt': len(with_gt),
        'correct_predictions': correct,
        'accuracy': round(correct / len(with_gt) * 100, 2) if with_gt else None,
        'average_processing_time': round(avg_time, 2),
        'average_confidence': round(avg_confidence, 2)
    }
    
    return metrics

def main() -> None:
    """Main function"""
    global interrupted
    
    args = parse_args()
    
    # Setup logging
    log_file = Path(args.output_dir) / 'logs' / 'eval.log'
    global logger
    logger = setup_logging(args.log_level, str(log_file))
    
    logger.info("="*80)
    logger.info("Rotary Control Recognition Evaluation - Bbox Enhanced")
    logger.info("Single-process sequential mode with interrupt support")
    logger.info("="*80)
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Use bbox enhancement: {args.use_bbox}")
    logger.info(f"Evaluation subset: {args.subset}")
    
    # Validate input directory
    if not os.path.exists(args.dataset_dir):
        logger.error(f"Dataset directory does not exist: {args.dataset_dir}")
        return
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    if args.save_intermediate_images:
        os.makedirs(os.path.join(args.output_dir, 'intermediate_images'), exist_ok=True)
    
    # Load dataset
    samples = []
    if args.subset in ['with_status', 'both']:
        samples.extend(load_dataset_samples(args.dataset_dir, 'with_status', args.max_samples))
    if args.subset in ['without_status', 'both']:
        samples.extend(load_dataset_samples(args.dataset_dir, 'without_status', args.max_samples))
    
    if not samples:
        logger.error("No samples found")
        return
    
    logger.info(f"Total loaded samples: {len(samples)}\n")
    
    # Configuration parameters
    config = {
        'eas_url': args.eas_url,
        'eas_token': args.eas_token,
        'model_name': args.model_name,
        'max_tokens': args.max_tokens,
        'timeout': args.timeout,
        'output_dir': args.output_dir,
        'save_intermediate_images': args.save_intermediate_images,
        'question': args.question,
        'use_bbox': args.use_bbox,
        'log_level': args.log_level
    }
    
    # Clear old result files and create results_chunks directory
    results_chunks_dir = Path(args.output_dir) / 'results_chunks'
    if results_chunks_dir.exists():
        logger.info(f"Clearing existing chunk files in: {results_chunks_dir}")
        for chunk_file in results_chunks_dir.glob('results_chunk_*.jsonl'):
            chunk_file.unlink()
    else:
        results_chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples sequentially
    start_time = time.time()
    results = process_samples_sequential(samples, config)
    total_time = time.time() - start_time
    
    # === Merge chunk files into single results.jsonl for convenience ===
    logger.info("\nMerging chunk files into single results.jsonl...")
    merged_result_file = Path(args.output_dir) / 'results.jsonl'
    chunk_files = sorted(results_chunks_dir.glob('results_chunk_*.jsonl'))
    
    with open(merged_result_file, 'w', encoding='utf-8') as merged_f:
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as chunk_f:
                merged_f.write(chunk_f.read())
    
    logger.info(f"Merged {len(chunk_files)} chunk files into {merged_result_file.name}")
    logger.info(f"Chunk files preserved in: {results_chunks_dir}/")
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(results)
    metrics['total_time'] = round(total_time, 2)
    metrics['timestamp'] = datetime.datetime.now().isoformat()
    metrics['interrupted'] = interrupted
    metrics['total_chunks'] = len(chunk_files)
    metrics['chunk_size'] = 10
    
    # Create safe config copy (remove sensitive info)
    safe_config = {k: v for k, v in config.items() if k != 'eas_token'}
    metrics['config'] = safe_config
    
    # Save evaluation report
    report_file = Path(args.output_dir) / 'eval_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # Print summary
    logger.info("")
    logger.info("="*80)
    if interrupted:
        logger.info("⚠️  Evaluation Interrupted (results saved)")
    else:
        logger.info("✓ Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Processed: {len(results)} samples")
    logger.info(f"Successfully processed: {metrics['successful']} ({metrics['success_rate']}%)")
    
    if metrics['accuracy'] is not None:
        logger.info(f"Recognition accuracy: {metrics['correct_predictions']}/{metrics['samples_with_gt']} ({metrics['accuracy']}%)")
    
    logger.info(f"Average time: {metrics['average_processing_time']:.2f}s/sample")
    logger.info(f"Average confidence: {metrics['average_confidence']:.2f}")
    logger.info(f"Total time: {metrics['total_time']:.2f}s")
    logger.info(f"")
    logger.info(f"📊 Results Storage:")
    logger.info(f"  • Merged file: {merged_result_file} ({len(results)} items)")
    logger.info(f"  • Chunk files: {results_chunks_dir}/ ({metrics.get('total_chunks', 0)} files × 10 items)")
    logger.info(f"  • Evaluation report: {report_file}")
    logger.info("="*80)
    
    # Exit with appropriate code
    sys.exit(1 if interrupted else 0)

if __name__ == "__main__":
    main()
