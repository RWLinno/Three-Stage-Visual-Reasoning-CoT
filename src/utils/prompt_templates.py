"""
Task-specific prompt templates - Provide dedicated prompts for different visual reasoning tasks
Enhanced version with improved reasoning quality and visual prompts
"""
from typing import Dict, Any, Optional, List

class TaskTemplates:
    """
    Task Template Manager
    
    Predefined prompt templates for various visual reasoning tasks
    Supports dynamic extension and customization
    """
    
    @staticmethod
    def get_template(task_type: str) -> Dict[str, str]:
        """
        Get task template
        
        Args:
            task_type: Task type
            
        Returns:
            Dictionary containing prompts for each stage
        """
        templates = {
            "washer_knob": {
                "stage1": (
                    "# Stage 1: Rule Extraction\n\n"
                    "**Original User Question:** {question}\n\n"
                    "As a visual reasoning expert, your task is to analyze this washing machine knob image and derive precise geometric rules for determining the knob position.\n\n"
                    "## Visual Analysis Tasks:\n"
                    "1. **Knob Detection:** Identify the circular knob region\n"
                    "   - Locate the knob center point (geometric center of the circular knob)\n"
                    "   - Estimate the knob radius\n"
                    "   - Output format: Center coordinates (x, y) and radius r\n\n"
                    "2. **Red Pointer Detection:** Locate the red pointer/indicator\n"
                    "   - The red pointer extends from the knob center to the edge\n"
                    "   - Determine the pointer's angular position (0-360 degrees)\n"
                    "   - Output format: Angle θ from horizontal axis\n\n"
                    "3. **Green Scale Lines Detection:** Identify all green extension lines\n"
                    "   - Each text label has a corresponding green line connecting to the knob edge\n"
                    "   - Each green line marks a specific mode/setting position\n"
                    "   - Output format: List of (label_name, angle) pairs\n\n"
                    "4. **Alignment Rule Derivation:**\n"
                    "   - Define the criterion for \"alignment\" between red pointer and green scale line\n"
                    "   - Consider angular tolerance (e.g., ±5 degrees)\n"
                    "   - Explain how to verify center-pointer-scale collinearity\n\n"
                    "## Output Requirements:\n"
                    "You MUST structure your response EXACTLY as follows (use this exact format):\n\n"
                    "**KNOB GEOMETRY:**\n"
                    "- Center: (640, 480)\n"
                    "- Radius: 200 pixels\n\n"
                    "**RED POINTER:**\n"
                    "- Angle: 45 degrees\n"
                    "- Endpoint: (780, 620)\n\n"
                    "**GREEN SCALE LINES:**\n"
                    "- Off: 90 degrees\n"
                    "- Quick Wash 15: 60 degrees\n"
                    "- Speed Wash 30: 30 degrees\n"
                    "- (continue for all visible modes)\n\n"
                    "**ALIGNMENT RULES:**\n"
                    "1. The pointer must be within 5 degrees of a scale line to be considered aligned\n"
                    "2. The mode with minimum angular distance is the selected mode\n"
                    "3. Visual check: extend pointer line and see which scale line it intersects\n\n"
                    "**DECISION CRITERION:**\n"
                    "Find the scale line with minimum angular distance from the red pointer. If distance < 5 degrees, that mode is selected.\n\n"
                    "**CRITICAL:** You MUST provide numeric values for center coordinates, radius, angles. Do NOT use placeholders like [X], [Y], [θ]. Measure from the image and provide actual numbers!"
                ),
                "stage2": (
                    "# Stage 2: Application Reasoning\n\n"
                    "Based on the geometric rules and visual analysis from Stage 1, now determine the current knob position.\n\n"
                    "## Rules from Stage 1:\n"
                    "{rules}\n\n"
                    "## Task:\n"
                    "Apply the derived rules to determine which mode/setting the knob is currently pointing to.\n\n"
                    "## Reasoning Process:\n"
                    "1. **Red Pointer Position:** Confirm the pointer's current angle\n"
                    "2. **Compare with Scale Lines:** Calculate angular distance to each green scale line\n"
                    "3. **Find Closest Match:** Identify which scale line has minimum angular distance\n"
                    "4. **Verify Alignment:** Check if the alignment meets the criterion (within tolerance)\n"
                    "5. **Output Answer:** State the mode name clearly\n\n"
                    "## Visual Verification:\n"
                    "- Mentally draw a line from center through red pointer to knob edge\n"
                    "- Check which green scale line endpoint this line passes through\n"
                    "- Confirm the text label associated with that green line\n\n"
                    "## Output Format:\n"
                    "First, show your step-by-step reasoning.\n"
                    "Then, output the final answer in the following format:\n\n"
                    "<answer>[Exact Mode Name]</answer>\n\n"
                    "Note: Only output the exact text label shown in the image (e.g., \"大件\", \"Quick Wash 15\", \"Wool\", etc.)"
                ),
                "stage3": (
                    "# Stage 3: Geometric Alignment Validation\n\n"
                    "The answer from Stage 2 is: **{answer}**\n\n"
                    "Now perform STRICT geometric validation to verify pointer-scale alignment.\n\n"
                    "## Critical Geometric Checks:\n\n"
                    "### 1. Pointer-Scale Collinearity Test (MANDATORY)\n"
                    "**Task:** Verify that the red pointer and the green scale line for '{answer}' are on the SAME radial line from knob center.\n\n"
                    "**Method:**\n"
                    "- Identify the exact angle of the red pointer from knob center\n"
                    "- Identify the exact angle of the green scale line endpoint for '{answer}' from knob center\n"
                    "- Calculate angular difference (should be < 5 degrees for valid alignment)\n\n"
                    "**Result Format:**\n"
                    "- Red pointer angle: [X] degrees\n"
                    "- '{answer}' scale line angle: [Y] degrees\n"
                    "- Angular difference: [Z] degrees\n"
                    "- **Collinearity Status: PASS / FAIL**\n\n"
                    "### 2. Nearest Scale Line Test (MANDATORY)\n"
                    "**Task:** Find which scale line is ACTUALLY closest to the red pointer.\n\n"
                    "**Method:**\n"
                    "- List all visible scale lines and their angles\n"
                    "- Calculate angular distance from red pointer to each scale line\n"
                    "- Identify the scale line with MINIMUM angular distance\n\n"
                    "**Result Format:**\n"
                    "- Closest scale line: [Mode Name]\n"
                    "- Angular distance: [X] degrees\n"
                    "- **Match Status: MATCH (same as Stage2 answer) / MISMATCH (different from Stage2 answer)**\n\n"
                    "### 3. Alternative Modes Check\n"
                    "**Task:** Check if any adjacent mode is closer than '{answer}'.\n\n"
                    "**Adjacent modes to check:** {adjacent_modes}\n\n"
                    "For each adjacent mode, calculate angular distance and report:\n"
                    "- [Mode 1]: [X] degrees\n"
                    "- [Mode 2]: [Y] degrees\n"
                    "- ...\n\n"
                    "**Conclusion:** Is any adjacent mode closer than '{answer}'? YES [mode name] / NO\n\n"
                    "## STRICT VALIDATION DECISION:\n\n"
                    "**Decision Rules:**\n"
                    "- If Collinearity Status = FAIL → **INVALID**\n"
                    "- If Match Status = MISMATCH → **INVALID: Pointer points to [actual closest mode], not '{answer}'**\n"
                    "- If any adjacent mode is closer → **INVALID: Should be [adjacent mode name]**\n"
                    "- Only if ALL tests pass → **VALID**\n\n"
                    "**YOUR FINAL DECISION:**\n"
                    "[Write VALID or INVALID: [reason] here]\n\n"
                    "**IMPORTANT:** Be EXTREMELY strict. Even a 6-degree deviation should trigger INVALID. The goal is geometric precision, not approximate matching."
                )
            },
            "generic_visual": {
                "stage1": (
                    "# Stage 1: Rule Extraction\n\n"
                    "**Original User Question:** {question}\n\n"
                    "As a visual analysis expert, carefully examine this image and derive the core reasoning rules.\n\n"
                    "## Analysis Tasks:\n"
                    "1. **Visual Element Identification:** List all key visual elements\n"
                    "2. **Spatial Relationships:** Describe how elements relate to each other\n"
                    "3. **Visual Cues:** Identify colors, shapes, positions, patterns\n"
                    "4. **Interaction Patterns:** Understand element dependencies\n\n"
                    "## Output Requirements:\n"
                    "**VISUAL ELEMENTS:**\n"
                    "- [Element 1]: Description\n"
                    "- [Element 2]: Description\n"
                    "- ...\n\n"
                    "**REASONING RULES:**\n"
                    "1. [Rule 1]\n"
                    "2. [Rule 2]\n"
                    "3. ...\n\n"
                    "**DECISION CRITERION:**\n"
                    "[How to make the final determination]\n\n"
                    "Think step by step."
                ),
                "stage2": (
                    "# Stage 2: Application Reasoning\n\n"
                    "## Rules from Stage 1:\n"
                    "{rules}\n\n"
                    "## Task:\n"
                    "Apply these rules to analyze the image and provide your answer.\n\n"
                    "## Output Format:\n"
                    "Show your reasoning process, then output:\n\n"
                    "<answer>[Your Answer]</answer>"
                ),
                "stage3": (
                    "# Stage 3: Validation\n\n"
                    "The answer from Stage 2 is: **{answer}**\n\n"
                    "## Validation Questions:\n\n"
                    "1. Does this answer satisfy all rules derived in Stage 1?\n"
                    "2. Are there alternative answers that better fit the image?\n"
                    "3. Are there any contradictory visual evidence?\n\n"
                    "## Final Validation Result:\n"
                    "**VALID** / **INVALID: [reason]** / **UNCERTAIN: [issue]**"
                )
            }
        }
        
        return templates.get(task_type, templates["generic_visual"])
    
    @staticmethod
    def format_stage1_prompt(template: str, question: str) -> str:
        """Format stage 1 prompt with user question"""
        return template.format(question=question)
    
    @staticmethod
    def format_stage2_prompt(template: str, rules: str) -> str:
        """Format stage 2 prompt with rules from stage 1"""
        return template.format(rules=rules)
    
    @staticmethod
    def format_stage3_prompt(template: str, answer: str, adjacent_modes: str = "") -> str:
        """Format stage 3 prompt with answer from stage 2"""
        return template.format(answer=answer, adjacent_modes=adjacent_modes)
    
    @staticmethod
    def extract_answer_tag(text: str) -> Optional[str]:
        """Extract answer from <answer> tags"""
        import re
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def register_template(task_type: str, template: Dict[str, str]):
        """
        Register new template (static method, can be stored to config file in actual use)
        
        Args:
            task_type: Task type name
            template: Template content
        """
        # In actual implementation, can be stored to config file or database
        pass
    
    @staticmethod
    def get_available_templates() -> List[str]:
        """Get available template types"""
        return ["washer_knob", "generic_visual"]
