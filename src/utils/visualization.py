"""
Visualization tools - Save intermediate reasoning images
VLM-guided visualization - saves VLM-generated auxiliary images
Falls back to original image if VLM doesn't return annotated image

All image drawing code has been removed.
Visualization is now done by VLM during reasoning.
See VLM_IMAGE_GENERATION_GUIDE.md for configuration details.
"""
import os
import json
import logging
import re
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

def save_geometric_info_to_json(
    geometric_info: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save geometric information to JSON file
    
    Args:
        geometric_info: Dictionary containing geometric analysis results
        output_path: JSON file path
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geometric_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved geometric info to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save geometric info: {e}")

def parse_geometric_info_from_rules(rules_text: str) -> Dict[str, Any]:
    """
    Parse geometric information from Stage1 VLM output
    
    Args:
        rules_text: Rules text from Stage1
        
    Returns:
        Dictionary containing knob_center, knob_radius, red_pointer_angle, green_scale_lines
    """
    geo_info = {
        'knob_center': None,
        'knob_radius': None,
        'red_pointer_angle': None,
        'red_pointer_endpoint': None,
        'green_scale_lines': []
    }
    
    # Log input text length for debugging
    logger.info(f"Parsing geometric info from rules_text of length {len(rules_text)} chars")
    
    try:
        # First, try to extract the structured output section (after </think> tag)
        # This avoids matching intermediate calculation values in the thinking process
        structured_output = rules_text
        if '</think>' in rules_text:
            # Extract content after the thinking section
            structured_output = rules_text.split('</think>')[-1]
            logger.debug("Extracted structured output section after </think> tag")
        
        # Parse knob center: look for "- Center: (x, y)" in structured output
        # Pattern matches "- Center: (300, 440.5)" or "Center: (300, 440.5)"
        center_match = re.search(r'[-*]?\s*[Cc]enter[:\s]+\(?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)?', structured_output)
        if center_match:
            geo_info['knob_center'] = (float(center_match.group(1)), float(center_match.group(2)))
            logger.info(f"Parsed knob center: {geo_info['knob_center']}")
        else:
            logger.warning("Could not parse knob center from structured output")
            # Fallback: try the full text but be more specific
            center_match = re.search(r'CIRCULAR ELEMENT GEOMETRY:[\s\S]{0,200}[-*]?\s*[Cc]enter[:\s]+\(?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)?', rules_text)
            if center_match:
                geo_info['knob_center'] = (float(center_match.group(1)), float(center_match.group(2)))
                logger.info(f"Parsed knob center (fallback): {geo_info['knob_center']}")
        
        # Parse knob radius: look for "- Radius: r pixels" in structured output
        radius_match = re.search(r'[-*]?\s*[Rr]adius[:\s]+(\d+\.?\d*)\s*(?:pixels?)?', structured_output)
        if radius_match:
            geo_info['knob_radius'] = float(radius_match.group(1))
            logger.info(f"Parsed knob radius: {geo_info['knob_radius']}")
        else:
            logger.warning("Could not parse knob radius from structured output")
            # Fallback
            radius_match = re.search(r'CIRCULAR ELEMENT GEOMETRY:[\s\S]{0,200}[-*]?\s*[Rr]adius[:\s]+(\d+\.?\d*)\s*(?:pixels?)?', rules_text)
            if radius_match:
                geo_info['knob_radius'] = float(radius_match.group(1))
                logger.info(f"Parsed knob radius (fallback): {geo_info['knob_radius']}")
        
        # Parse pointer angle: look for "- Angle: X°" in POINTER/INDICATOR section
        angle_match = re.search(r'POINTER.*?[-*]?\s*[Aa]ngle[:\s]+(\d+\.?\d*)\s*[°degrees]*', structured_output, re.DOTALL)
        if angle_match:
            geo_info['red_pointer_angle'] = float(angle_match.group(1))
            logger.info(f"Parsed pointer angle: {geo_info['red_pointer_angle']}")
        else:
            logger.warning("Could not parse pointer angle from structured output")
            # Simple fallback
            angle_match = re.search(r'[-*]?\s*[Aa]ngle[:\s]+(\d+\.?\d*)\s*[°degrees]*', structured_output)
            if angle_match:
                geo_info['red_pointer_angle'] = float(angle_match.group(1))
                logger.info(f"Parsed pointer angle (fallback): {geo_info['red_pointer_angle']}")
        
        # Parse green scale lines: look for "POSITION LABEL ANGLES:" section
        # Match lines like "- 大件: 45 degrees" or "- 大件: 186.1°"
        # First try to extract from POSITION LABEL ANGLES section
        label_section = structured_output
        if 'POSITION LABEL ANGLES:' in structured_output:
            # Extract only the label angles section
            label_section_match = re.search(r'POSITION LABEL ANGLES:(.*?)(?=\n\*\*|$)', structured_output, re.DOTALL)
            if label_section_match:
                label_section = label_section_match.group(1)
                logger.debug("Extracted POSITION LABEL ANGLES section")
        
        # Match patterns: "- 混合40°C: 192.6°" or "- 混合40°C: 192.6 degrees"
        scale_lines = re.findall(r'[-•]\s*([^:]+?):\s*(\d+\.?\d*)\s*[°degrees]*', label_section)
        for label, angle in scale_lines:
            label_clean = label.strip()
            # Filter out non-label lines (like "Angular difference")
            if label_clean and not label_clean.lower().startswith(('angular', 'tolerance', 'visual')):
                geo_info['green_scale_lines'].append({
                    'label': label_clean,
                    'angle': float(angle)
                })
        
        if geo_info['green_scale_lines']:
            logger.info(f"Parsed {len(geo_info['green_scale_lines'])} scale lines")
        else:
            logger.warning("Could not parse any scale lines from structured output")
        
        # Check if parsing was completely unsuccessful
        if not any([geo_info['knob_center'], geo_info['knob_radius'], geo_info['red_pointer_angle']]):
            logger.error("Failed to parse any geometric information from rules_text!")
            logger.debug(f"Rules text preview (first 500 chars): {rules_text[:500]}")
        
    except Exception as e:
        logger.error(f"Exception while parsing geometric info from rules: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return geo_info

def save_intermediate_images(
    image_path: str,
    results: Dict[str, Any],
    output_dir: str,
    image_name: str
) -> None:
    """
    Save intermediate visualizations
    - Parses VLM's geometric analysis from stage1
    - Draws auxiliary lines on original image
    - Saves annotated image
    - Saves all reasoning results in JSON files
    
    Args:
        image_path: Original image path
        results: CoT reasoning results with stage1_rules containing geometric info
        output_dir: Output directory
        image_name: Image name
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Parse geometric info from VLM's stage1 output
        stage1_rules = results.get('stage1_rules', '')
        geo_info = parse_geometric_info_from_rules(stage1_rules)
        
        # Check if we have valid geometric info
        if geo_info['knob_center'] and geo_info['knob_radius'] and geo_info['red_pointer_angle']:
            # Draw auxiliary lines based on parsed info
            logger.info(f"Drawing auxiliary lines for {image_name} based on VLM geometry")
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Import drawing function
            from .visualization_draw import draw_auxiliary_lines_on_image
            
            img_with_lines = draw_auxiliary_lines_on_image(
                img,
                knob_center=tuple(geo_info['knob_center']),
                knob_radius=geo_info['knob_radius'],
                pointer_angle=geo_info['red_pointer_angle'],
                label_angles=geo_info['green_scale_lines']
            )
            
            # Save annotated image
            output_path = Path(output_dir) / f"{Path(image_name).stem}_auxiliary_lines.jpg"
            img_with_lines.save(output_path, quality=95)
            logger.info(f"Saved auxiliary lines image: {output_path}")
        else:
            # Geometric info not available - save original image
            logger.warning(f"Could not parse geometric info for {image_name}, using original image")
            save_original_as_auxiliary(image_path, output_dir, image_name)
        
        # Save all reasoning results in JSON
        save_complete_results_json(results, output_dir, image_name)
        
        logger.info(f"Saved visualizations for: {image_name}")
        
    except Exception as e:
        logger.error(f"Failed to save intermediate images {image_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        # Fallback: save original image
        try:
            save_original_as_auxiliary(image_path, output_dir, image_name)
        except:
            pass

def save_complete_results_json(
    results: Dict[str, Any],
    output_dir: str,
    image_name: str
) -> None:
    """
    Save complete reasoning results to a single JSON file
    """
    try:
        # Extract short answer - must be clean answer text only
        final_answer = results.get('final_answer', '')
        stage2_answer = results.get('stage2_answer', '')
        
        # Priority 1: Extract from <answer> tag
        short_answer = None
        if '<answer>' in final_answer and '</answer>' in final_answer:
            import re
            match = re.search(r'<answer>(.*?)</answer>', final_answer, re.DOTALL)
            if match:
                short_answer = match.group(1).strip()
        
        # Priority 2: Try stage2_answer
        if not short_answer and stage2_answer:
            if '<answer>' in stage2_answer and '</answer>' in stage2_answer:
                match = re.search(r'<answer>(.*?)</answer>', stage2_answer, re.DOTALL)
                if match:
                    short_answer = match.group(1).strip()
            # If stage2_answer is short and clean, use it
            elif len(stage2_answer) < 50 and '\n' not in stage2_answer:
                short_answer = stage2_answer.strip()
        
        # Priority 3: Extract last clean line from final_answer
        if not short_answer:
            lines = final_answer.split('\n')
            for line in reversed(lines):
                line = line.strip()
                # Skip thinking tags, headers, and long explanatory lines
                if (line and 
                    not line.startswith(('#', '**', '</', '-', '•')) and
                    not line.lower().startswith(('test', 'step', 'note', 'wait', 'let')) and
                    len(line) < 100 and  # Answers should be short
                    not line.endswith(':')):
                    short_answer = line
                    break
        
        # Fallback: use stage2_answer or truncate
        if not short_answer:
            short_answer = stage2_answer[:50] if stage2_answer else final_answer[:50]
            logger.warning(f"Could not extract clean answer, using truncated: {short_answer}")
        
        # Clean up quotes and extra whitespace
        short_answer = short_answer.strip('"\'')
        
        logger.info(f"Extracted answer: {short_answer}")
        
        complete_results = {
            'image_name': image_name,
            'answer': short_answer,
            'final_answer': final_answer,
            'confidence': results.get('confidence', 0.0),
            'stage1_rules': results.get('stage1_rules', ''),  # Keep full reasoning
            'stage2_answer': results.get('stage2_answer', ''),
            'stage3_validation': results.get('stage3_validation', ''),  # Keep full validation
            'retry_history': results.get('retry_history', [])
        }
        
        output_path = Path(output_dir) / f"{Path(image_name).stem}_complete_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved complete results JSON: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save complete results JSON: {e}")

def save_vlm_auxiliary_image(
    vlm_image_data: Any,
    output_dir: str,
    image_name: str
) -> None:
    """
    Save VLM-generated auxiliary image with annotations
    
    Args:
        vlm_image_data: Image data from VLM (could be base64, PIL Image, or file path)
        output_dir: Output directory
        image_name: Original image name
    """
    try:
        output_path = Path(output_dir) / f"{Path(image_name).stem}_auxiliary_lines.jpg"
        
        # Handle different VLM image data formats
        if isinstance(vlm_image_data, str):
            # Base64 encoded image
            if vlm_image_data.startswith('data:image'):
                # Remove data URI prefix
                import base64
                image_data = vlm_image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
            else:
                # File path
                img = Image.open(vlm_image_data)
        elif isinstance(vlm_image_data, Image.Image):
            # PIL Image object
            img = vlm_image_data
        else:
            # Numpy array or other format
            img = Image.fromarray(vlm_image_data)
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save the VLM-generated annotated image
        img.save(output_path, quality=95)
        logger.info(f"Saved VLM-generated auxiliary image: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save VLM auxiliary image: {e}")
        raise

def save_original_as_auxiliary(
    image_path: str,
    output_dir: str,
    image_name: str
) -> None:
    """
    Save original image as auxiliary image (fallback when VLM doesn't return annotated image)
    
    Args:
        image_path: Original image path
        output_dir: Output directory
        image_name: Image name
    """
    try:
        output_path = Path(output_dir) / f"{Path(image_name).stem}_auxiliary_lines.jpg"
        
        # Read and save original image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(output_path, quality=95)
        logger.info(f"Saved original image as auxiliary (VLM fallback): {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save original image as auxiliary: {e}")
        raise

