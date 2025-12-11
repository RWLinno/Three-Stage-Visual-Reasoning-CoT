"""
Auxiliary line drawing functions
Draws geometric annotations based on VLM's geometric analysis
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def draw_auxiliary_lines_on_image(
    img: Image.Image,
    knob_center: Tuple[float, float],
    knob_radius: float,
    pointer_angle: float,
    label_angles: List[Dict[str, Any]]
) -> Image.Image:
    """
    Draw auxiliary lines on image based on VLM's geometric analysis
    
    Args:
        img: Original image
        knob_center: (x, y) coordinates of knob center
        knob_radius: Knob radius in pixels
        pointer_angle: Pointer angle in degrees (clockwise from right)
        label_angles: List of {'label': str, 'angle': float} dicts
        
    Returns:
        Image with auxiliary lines drawn
    """
    img_with_lines = img.copy()
    draw = ImageDraw.Draw(img_with_lines)
    
    # Try to load font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 1. Draw knob circle (blue)
    circle_bbox = [
        knob_center[0] - knob_radius,
        knob_center[1] - knob_radius,
        knob_center[0] + knob_radius,
        knob_center[1] + knob_radius
    ]
    draw.ellipse(circle_bbox, outline='blue', width=4)
    
    # 2. Draw center point (blue)
    center_size = 10
    draw.ellipse([
        knob_center[0] - center_size,
        knob_center[1] - center_size,
        knob_center[0] + center_size,
        knob_center[1] + center_size
    ], fill='blue', outline='darkblue', width=2)
    
    # Add center annotation
    center_text = f"Center: ({int(knob_center[0])}, {int(knob_center[1])})"
    text_pos = (int(knob_center[0] - 60), int(knob_center[1] + knob_radius + 10))
    try:
        bbox = draw.textbbox(text_pos, center_text, font=font_small)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='white', outline='blue')
        draw.text(text_pos, center_text, fill='blue', font=font_small)
    except:
        draw.text(text_pos, center_text, fill='blue')
    
    # 3. Draw pointer line (red)
    angle_rad = np.deg2rad(pointer_angle)
    pointer_end = (
        int(knob_center[0] + knob_radius * 1.1 * np.cos(angle_rad)),
        int(knob_center[1] + knob_radius * 1.1 * np.sin(angle_rad))
    )
    draw.line([knob_center, pointer_end], fill='red', width=6)
    
    # Draw pointer endpoint
    end_size = 8
    draw.ellipse([
        pointer_end[0] - end_size,
        pointer_end[1] - end_size,
        pointer_end[0] + end_size,
        pointer_end[1] + end_size
    ], fill='red', outline='darkred', width=2)
    
    # Add pointer angle annotation
    angle_text = f"Pointer: {pointer_angle:.1f}°"
    text_offset_x = 15 if np.cos(angle_rad) > 0 else -80
    text_offset_y = 15 if np.sin(angle_rad) > 0 else -20
    angle_pos = (pointer_end[0] + text_offset_x, pointer_end[1] + text_offset_y)
    try:
        bbox = draw.textbbox(angle_pos, angle_text, font=font_large)
        draw.rectangle([bbox[0]-3, bbox[1]-3, bbox[2]+3, bbox[3]+3], fill='white', outline='red', width=2)
        draw.text(angle_pos, angle_text, fill='red', font=font_large)
    except:
        draw.text(angle_pos, angle_text, fill='red')
    
    # 4. Draw label lines (green)
    for idx, label_info in enumerate(label_angles[:5]):  # Limit to 5 to avoid clutter
        angle_rad = np.deg2rad(label_info['angle'])
        
        # Draw line from knob edge to outside
        start_radius = knob_radius
        start_point = (
            int(knob_center[0] + start_radius * np.cos(angle_rad)),
            int(knob_center[1] + start_radius * np.sin(angle_rad))
        )
        end_radius = knob_radius * 1.3
        end_point = (
            int(knob_center[0] + end_radius * np.cos(angle_rad)),
            int(knob_center[1] + end_radius * np.sin(angle_rad))
        )
        
        draw.line([start_point, end_point], fill='green', width=3)
        
        # Draw endpoint
        end_size = 5
        draw.ellipse([
            end_point[0] - end_size,
            end_point[1] - end_size,
            end_point[0] + end_size,
            end_point[1] + end_size
        ], fill='green', outline='darkgreen', width=1)
    
    # 5. Add legend
    legend_x = 10
    legend_y = 10
    legend_items = [
        ("● Blue:", "blue", f"Knob (R={knob_radius:.0f}px)"),
        ("● Red:", "red", f"Pointer ({pointer_angle:.1f}°)"),
        ("● Green:", "green", f"Labels ({len(label_angles)} total)")
    ]
    
    for idx, (symbol, color, desc) in enumerate(legend_items):
        y_pos = legend_y + idx * 20
        # Background
        draw.rectangle([legend_x-2, y_pos-2, legend_x+220, y_pos+15], 
                     fill='white', outline='black')
        # Text
        draw.text((legend_x, y_pos), symbol, fill=color, font=font_small)
        draw.text((legend_x+50, y_pos), desc, fill='black', font=font_small)
    
    return img_with_lines

