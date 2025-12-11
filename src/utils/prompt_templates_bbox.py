"""
Bbox-enhanced prompt templates for generic rotary control recognition
Generic geometric reasoning framework without task-specific priors
"""
from typing import Dict, Any, Optional, List

class BboxEnhancedTemplates:
    """Bbox-enhanced prompt templates using generic geometric reasoning"""
    
    @staticmethod
    def get_generic_rotary_template_with_bbox() -> Dict[str, str]:
        """Get bbox-enhanced template for generic rotary control recognition"""
        return {
            "stage1": (
                "# Stage 1: Geometric Rule Extraction\n\n"
                "**User Question:** {question}\n\n"
                "You are a visual reasoning expert. Analyze this image and derive geometric rules to answer the question.\n\n"
                "## Available Information (Bounding Boxes):\n"
                "{bbox_info}\n\n"
                "## Generic Geometric Analysis Framework:\n\n"
                "### Step 1: Identify the Circular Control Element\n"
                "- Locate the circular element in the image (labeled as 'knob' in bounding boxes)\n"
                "- Using the provided bounding box `{knob_bbox}`, calculate:\n"
                "  * Center point: center_x = (bbox_x1 + bbox_x2) / 2, center_y = (bbox_y1 + bbox_y2) / 2\n"
                "  * Radius: r = min(bbox_width, bbox_height) / 2\n"
                "  * This establishes your coordinate system origin\n\n"
                "### Step 2: Identify the Pointer/Indicator\n"
                "- Look for a visual indicator on the circular element (could be a line, arrow, mark, or asymmetric feature)\n"
                "- Determine the direction this indicator points from the center\n"
                "- Measure the angle θ from horizontal right (0°) going clockwise (0° to 360°)\n"
                "- Important: DO NOT assume what the indicator looks like - observe it from the image\n\n"
                "### Step 3: Map Position Labels to Angles\n"
                "- The image contains multiple labeled positions around the circular element\n"
                "- For each label with bounding box provided:\n"
                "{mode_bboxes}\n"
                "- Calculate the angle from center to each label's center point\n"
                "- Formula: angle = atan2(label_center_y - knob_center_y, label_center_x - knob_center_x)\n"
                "- Convert to 0-360° range\n"
                "- Create a mapping: Label → Angle\n\n"
                "### Step 4: Derive Alignment Rules\n"
                "- Define what constitutes \"alignment\" between pointer and a position label\n"
                "- Suggest using angular tolerance (e.g., ±5 degrees)\n"
                "- The aligned position is the one with minimum angular distance to the pointer\n"
                "- Consider edge cases (e.g., 0°/360° boundary)\n\n"
                "## Output Format:\n"
                "Structure your response as follows:\n\n"
                "**CIRCULAR ELEMENT GEOMETRY:**\n"
                "- Center: (x, y)\n"
                "- Radius: r pixels\n\n"
                "**POINTER/INDICATOR:**\n"
                "- Angle: θ degrees (from horizontal right, clockwise)\n"
                "- Description: [describe what the indicator looks like]\n\n"
                "**POSITION LABEL ANGLES:**\n"
                "- [Label 1]: X degrees\n"
                "- [Label 2]: Y degrees\n"
                "- (list all labels)\n\n"
                "**ALIGNMENT RULES:**\n"
                "1. Calculate angular difference between pointer and each label\n"
                "2. The label with minimum angular difference (< tolerance) is selected\n"
                "3. Visual verification: extend pointer ray and check which label region it intersects\n\n"
                "**DECISION CRITERION:**\n"
                "Find argmin_label(|pointer_angle - label_angle|) where difference < threshold\n\n"
                "**CRITICAL:** Provide actual numeric values based on image observation and provided bounding boxes. Do NOT use placeholders."
            ),
            "stage2": (
                "# Stage 2: Apply Geometric Rules\n\n"
                "Based on the geometric rules derived in Stage 1, determine the answer to the question.\n\n"
                "## Rules from Stage 1:\n"
                "{rules}\n\n"
                "## Task:\n"
                "Apply the geometric rules to determine which position label the indicator is currently pointing to.\n\n"
                "## Reasoning Process:\n"
                "1. **Confirm pointer angle:** Re-identify the pointer's current angle from the image\n"
                "2. **Calculate distances:** Compute angular distance from pointer to each position label\n"
                "3. **Find minimum:** Identify which label has the smallest angular distance\n"
                "4. **Verify alignment:** Check if the minimum distance satisfies the alignment criterion\n"
                "5. **State answer:** Report the position label name\n\n"
                "## Geometric Verification:\n"
                "- Mentally draw a ray from center through the pointer direction\n"
                "- Check which position label region this ray passes through or is closest to\n"
                "- Confirm the label text associated with that region\n\n"
                "## Output Format:\n"
                "First, show your step-by-step reasoning.\n"
                "Then, output the final answer:\n\n"
                "<answer>[Position Label]</answer>\n\n"
                "Note: Output the exact text of the position label as shown in the image."
            ),
            "stage3": (
                "# Stage 3: Geometric Validation\n\n"
                "The answer from Stage 2 is: **{answer}**\n\n"
                "Perform STRICT geometric validation to verify the pointer-label alignment.\n\n"
                "## Geometric Validation Tests:\n\n"
                "### Test 1: Radial Collinearity (MANDATORY)\n"
                "**Objective:** Verify that the pointer and the position label '{answer}' lie on the same radial line from the center.\n\n"
                "**Method:**\n"
                "- Measure the exact angle of the pointer from the center point\n"
                "- Measure the exact angle of the position label '{answer}' from the center point\n"
                "- Calculate angular difference: |pointer_angle - label_angle|\n"
                "- Check if difference < tolerance threshold (typically 5°)\n\n"
                "**Report Format:**\n"
                "- Pointer angle: [X]°\n"
                "- '{answer}' label angle: [Y]°\n"
                "- Angular difference: [Z]°\n"
                "- **Collinearity Status: PASS / FAIL**\n\n"
                "### Test 2: Minimum Distance Check (MANDATORY)\n"
                "**Objective:** Verify that '{answer}' is actually the closest position label to the pointer.\n\n"
                "**Method:**\n"
                "- List all position labels and their angles from center\n"
                "- Calculate angular distance from pointer to each label\n"
                "- Identify which label has the MINIMUM angular distance\n\n"
                "**Report Format:**\n"
                "- Closest label: [Label Name]\n"
                "- Angular distance: [X]°\n"
                "- **Match Status: MATCH (same as Stage 2) / MISMATCH (different from Stage 2)**\n\n"
                "### Test 3: Alternative Labels Check\n"
                "**Objective:** Check if any neighboring labels are closer than '{answer}'.\n\n"
                "**Neighboring labels to check:** {adjacent_modes}\n\n"
                "For each neighboring label, report angular distance:\n"
                "- [Label 1]: [X]°\n"
                "- [Label 2]: [Y]°\n"
                "- ...\n\n"
                "**Conclusion:** Is any neighboring label closer? YES [label name] / NO\n\n"
                "## Validation Decision:\n\n"
                "**Decision Rules:**\n"
                "- If Collinearity Status = FAIL → **INVALID**\n"
                "- If Match Status = MISMATCH → **INVALID: Pointer actually points to [closest label], not '{answer}'**\n"
                "- If any neighbor is closer → **INVALID: Should be [neighbor label]**\n"
                "- Only if ALL tests pass → **VALID**\n\n"
                "**YOUR FINAL DECISION:**\n"
                "[Write VALID or INVALID: [reason] here]\n\n"
                "**IMPORTANT:** Be EXTREMELY strict. Even 6° deviation should trigger INVALID. The goal is geometric precision."
            )
        }
    
    @staticmethod
    def format_bbox_info(knob_data: Dict[str, Any]) -> tuple[str, str, str, str]:
        """
        Format bbox information for prompt injection
        
        Args:
            knob_data: Dictionary containing knob_close, modes, status(optional)
            
        Returns:
            (bbox_info, knob_bbox, mode_bboxes, current_status) as strings
        """
        bbox_info_lines = []
        knob_bbox_str = ""
        mode_bboxes_lines = []
        current_status = ""
        
        # Extract bboxes
        knob_close = knob_data.get('knob_close', [])
        for item in knob_close:
            label = item.get('label', '')
            bbox = item.get('bbox', [])
            
            if label.lower() == 'knob':
                knob_bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                bbox_info_lines.append(f"- Circular element region: {label}, bbox: {bbox}")
            else:
                bbox_info_lines.append(f"- Position label: {label}, bbox: {bbox}")
                mode_bboxes_lines.append(f"  * {label}: bbox {bbox}")
        
        # Extract current status (ground truth if available)
        if 'status' in knob_data and knob_data['status']:
            status_data = knob_data['status']
            status_label = status_data.get('label', '')
            status_bbox = status_data.get('bbox', [])
            current_status = f"Current state annotation: {status_label} (bbox: {status_bbox})"
            bbox_info_lines.append(f"- Note: {current_status}")
        
        bbox_info = "\n".join(bbox_info_lines)
        mode_bboxes = "\n".join(mode_bboxes_lines)
        
        return bbox_info, knob_bbox_str, mode_bboxes, current_status
    
    @staticmethod
    def create_stage1_prompt_with_bbox(question: str, knob_data: Dict[str, Any]) -> str:
        """
        Create Stage1 prompt with bbox information
        
        Args:
            question: User's question
            knob_data: Dictionary with bbox information
            
        Returns:
            Formatted prompt text
        """
        template = BboxEnhancedTemplates.get_generic_rotary_template_with_bbox()
        stage1_template = template['stage1']
        
        bbox_info, knob_bbox, mode_bboxes, _ = BboxEnhancedTemplates.format_bbox_info(knob_data)
        
        return stage1_template.format(
            question=question,
            bbox_info=bbox_info,
            knob_bbox=knob_bbox,
            mode_bboxes=mode_bboxes
        )
    
    @staticmethod
    def extract_ground_truth(knob_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract ground truth label from knob_data (if exists)
        
        Args:
            knob_data: Dictionary with bbox information
            
        Returns:
            Ground truth label, or None if not available
        """
        if 'status' in knob_data and knob_data['status']:
            return knob_data['status'].get('label', None)
        return None
