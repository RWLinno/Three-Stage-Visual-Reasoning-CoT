"""
CoT Reasoning Engine - Implements three-stage chain-of-thought reasoning
"""
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from src.base.vlm_agent import VLMAgentEAS
from src.utils.prompt_templates import TaskTemplates

logger = logging.getLogger(__name__)

class CoTEngine:
    """
    Chain-of-Thought Reasoning Engine
    
    Implements three-stage reasoning:
    1. Rule Extraction - Analyze visual elements and relationships
    2. Application - Apply rules to derive answer
    3. Validation - Verify the correctness of the answer
    """
    
    def __init__(
        self,
        vlm_agent: VLMAgentEAS,
        task_type: str = "washer_knob",
        templates: Optional[Dict[str, str]] = None,
        question: str = "Determine the current knob position"
    ):
        """
        Initialize CoT Engine
        
        Args:
            vlm_agent: VLM agent instance
            task_type: Task type
            templates: Custom prompt templates
            question: User's original question
        """
        self.vlm_agent = vlm_agent
        self.task_type = task_type
        self.templates = templates or TaskTemplates.get_template(task_type)
        self.question = question
        
        logger.debug(f"Initialized CoT reasoning engine - Task type: {task_type}")
    
    def reason(
        self,
        image_path: str,
        depth_path: Optional[str] = None,
        max_retries: int = 3,
        max_validation_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Execute complete CoT reasoning flow with validation-based retry
        
        Args:
            image_path: Image path
            depth_path: Depth map path
            max_retries: Maximum retry count per API call
            max_validation_retries: Maximum retries if validation fails
            
        Returns:
            Reasoning result dictionary
        """
        results = {
            'stage1_rules': '',
            'stage2_answer': '',
            'stage3_validation': '',
            'final_answer': '',
            'confidence': 0.0,
            'raw_responses': {},
            'retry_history': []
        }
        
        logger.debug(f"Starting CoT reasoning - Image: {image_path}")
        
        try:
            # Stage 1: Rule Extraction (only done once)
            stage1_template = self.templates.get('stage1', '')
            stage1_prompt = TaskTemplates.format_stage1_prompt(stage1_template, self.question)
            
            stage1_response = self.vlm_agent.inference_single(
                prompt_text=stage1_prompt,
                image_path=image_path,
                depth_path=depth_path,
                max_retries=max_retries
            )
            results['stage1_rules'] = stage1_response
            results['raw_responses']['stage1'] = stage1_response
            
            # Extract all available modes from stage1 for validation
            adjacent_modes = self._extract_modes_from_rules(stage1_response)
            
            # Stage 2-3 loop with validation-based retry
            validation_passed = False
            retry_count = 0
            previous_invalid_answers = []
            
            while not validation_passed and retry_count <= max_validation_retries:
                logger.info(f"Reasoning attempt {retry_count + 1}/{max_validation_retries + 1}")
                
                # Stage 2: Apply rules to derive answer
                stage2_template = self.templates.get('stage2', '')
                
                # Add hint about previous invalid answers and reflection
                if previous_invalid_answers:
                    # Add reflection from previous validation failure
                    last_failure = results['retry_history'][-1] if results.get('retry_history') else None
                    reflection_hint = f"\n\n## REFLECTION ON PREVIOUS FAILURE:\n"
                    reflection_hint += f"Previous attempt identified: **{', '.join(previous_invalid_answers)}**\n"
                    if last_failure:
                        reflection_hint += f"Validation failure reason: {last_failure['validation_failure']}\n"
                        if last_failure.get('suggested_correct_answer'):
                            reflection_hint += f"Suggested correct answer: {last_failure['suggested_correct_answer']}\n"
                    reflection_hint += f"\n**What went wrong?** Reflect on why the previous answer was incorrect.\n"
                    reflection_hint += f"**What to check?** Re-examine the pointer angle and scale line positions more carefully.\n"
                    reflection_hint += f"**Corrective action:** Measure angles precisely and find the truly closest scale line.\n"
                    
                    stage2_prompt = TaskTemplates.format_stage2_prompt(stage2_template, stage1_response) + reflection_hint
                else:
                    stage2_prompt = TaskTemplates.format_stage2_prompt(stage2_template, stage1_response)
                
                stage2_response = self.vlm_agent.inference_single(
                    prompt_text=stage2_prompt,
                    image_path=image_path,
                    depth_path=depth_path,
                    max_retries=max_retries
                )
                results[f'stage2_answer_attempt{retry_count}'] = stage2_response
                results['stage2_answer'] = stage2_response
                
                # Extract answer from <answer> tags if present
                extracted_answer = TaskTemplates.extract_answer_tag(stage2_response)
                if extracted_answer:
                    results['stage2_answer'] = extracted_answer
                
                # Stage 3: Validation
                stage3_template = self.templates.get('stage3', '')
                stage3_prompt = TaskTemplates.format_stage3_prompt(
                    stage3_template, 
                    results['stage2_answer'],
                    adjacent_modes=', '.join(adjacent_modes[:5]) if adjacent_modes else ""
                )
                
                stage3_response = self.vlm_agent.inference_single(
                    prompt_text=stage3_prompt,
                    image_path=image_path,
                    depth_path=depth_path,
                    max_retries=max_retries
                )
                results[f'stage3_validation_attempt{retry_count}'] = stage3_response
                results['stage3_validation'] = stage3_response
                
                # Check if validation passed
                validation_result = self._check_validation_status(stage3_response)
                
                # If validation failed, ask VLM to reflect on why
                if not validation_result['passed'] and retry_count < max_validation_retries:
                    reflection_prompt = (
                        f"# Validation Reflection\n\n"
                        f"Your previous answer '{results['stage2_answer']}' was INVALID.\n\n"
                        f"**Validation failure reason:** {validation_result['reason']}\n\n"
                        f"Please reflect:\n"
                        f"1. Why did you think '{results['stage2_answer']}' was correct?\n"
                        f"2. What geometric evidence contradicts this answer?\n"
                        f"3. What is the most likely correct answer based on geometric analysis?\n\n"
                        f"Provide a brief reflection (2-3 sentences) to guide the next attempt."
                    )
                    
                    reflection_response = self.vlm_agent.inference_single(
                        prompt_text=reflection_prompt,
                        image_path=image_path,
                        depth_path=depth_path,
                        max_retries=1
                    )
                    
                    validation_result['vlm_reflection'] = reflection_response
                    logger.info(f"VLM Reflection: {reflection_response[:200]}...")
                
                if validation_result['passed']:
                    validation_passed = True
                    logger.info(f"Validation PASSED on attempt {retry_count + 1}")
                else:
                    logger.warning(f"Validation FAILED on attempt {retry_count + 1}: {validation_result['reason']}")
                    previous_invalid_answers.append(results['stage2_answer'])
                    results['retry_history'].append({
                        'attempt': retry_count + 1,
                        'answer': results['stage2_answer'],
                        'validation_failure': validation_result['reason'],
                        'suggested_correct_answer': validation_result.get('suggested_answer', None)
                    })
                    
                    # If validation suggests a different answer, we can use it
                    if retry_count == max_validation_retries and validation_result.get('suggested_answer'):
                        logger.info(f"Using validation-suggested answer: {validation_result['suggested_answer']}")
                        results['stage2_answer'] = validation_result['suggested_answer']
                        results['final_answer'] = validation_result['suggested_answer']
                        results['confidence'] = 0.6  # Lower confidence for corrected answer
                        return results
                    
                    retry_count += 1
            
            # Store final responses
            results['raw_responses']['stage2'] = results['stage2_answer']
            results['raw_responses']['stage3'] = results['stage3_validation']
            
            # Synthesize final answer
            final_result = self._synthesize_final_answer(
                results['stage2_answer'], 
                results['stage3_validation'],
                stage1_response
            )
            results['final_answer'] = final_result['answer']
            results['confidence'] = final_result['confidence']
            
            # Reduce confidence if multiple retries were needed
            if retry_count > 0:
                results['confidence'] = max(0.5, results['confidence'] - 0.1 * retry_count)
            
            logger.info(f"Reasoning completed: {results['final_answer']} (confidence: {results['confidence']:.2f}, retries: {retry_count})")
            
        except Exception as e:
            logger.error(f"CoT reasoning exception: {e}")
            results['final_answer'] = f"Reasoning failed: {str(e)}"
            results['confidence'] = 0.0
        
        return results
    
    def batch_reason(
        self,
        image_paths: List[str],
        depth_paths: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Batch reasoning
        
        Args:
            image_paths: Image path list
            depth_paths: Depth map path list
            max_retries: Maximum retry count
            
        Returns:
            Result list
        """
        if depth_paths is None:
            depth_paths = [None] * len(image_paths)
        
        results = []
        for i, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
            result = self.reason(image_path, depth_path, max_retries)
            results.append(result)
        
        return results
    
    def _extract_modes_from_rules(self, rules_text: str) -> List[str]:
        """
        Extract mode names from Stage1 rules
        
        Args:
            rules_text: Rules text from Stage1
            
        Returns:
            List of mode names
        """
        import re
        modes = []
        
        # Look for patterns like "- [Label]: angle" or "Label: angle degrees"
        patterns = [
            r'[-•]\s*\[?([^\]:\n]+)\]?[:\s]+\d+\.?\d*\s*(?:degrees?)?',
            r'"([^"]+)"[:\s]+\d+\.?\d*\s*(?:degrees?)?',
            r'\'([^\']+)\'[:\s]+\d+\.?\d*\s*(?:degrees?)?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, rules_text)
            for match in matches:
                mode_name = match.strip()
                if mode_name and len(mode_name) < 50:  # Sanity check
                    modes.append(mode_name)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_modes = []
        for mode in modes:
            if mode not in seen:
                seen.add(mode)
                unique_modes.append(mode)
        
        logger.debug(f"Extracted {len(unique_modes)} modes from rules: {unique_modes[:10]}")
        return unique_modes
    
    def _check_validation_status(self, validation_text: str) -> Dict[str, Any]:
        """
        Check if validation passed based on Stage3 output
        
        Args:
            validation_text: Validation text from Stage3
            
        Returns:
            Dictionary with 'passed', 'reason', and optionally 'suggested_answer'
        """
        import re
        
        validation_lower = validation_text.lower()
        validation_upper = validation_text.upper()
        
        # Check for explicit INVALID markers
        if 'INVALID:' in validation_upper or 'FAIL' in validation_upper.split('\n')[-5:]:
            # Extract reason
            invalid_match = re.search(r'INVALID:?\s*(.+?)(?:\n|$)', validation_text, re.IGNORECASE)
            reason = invalid_match.group(1).strip() if invalid_match else "Validation check failed"
            
            # Try to extract suggested correct answer
            suggested_answer = None
            suggest_patterns = [
                r'should be\s*["\']?([^"\'\n]+)["\']?',
                r'actually?\s*["\']?([^"\'\n]+)["\']?',
                r'points? to\s*["\']?([^"\'\n]+)["\']?',
                r'closer[^\n]*["\']?([^"\'\n]+)["\']?'
            ]
            
            for pattern in suggest_patterns:
                match = re.search(pattern, validation_text, re.IGNORECASE)
                if match:
                    suggested_answer = match.group(1).strip()
                    break
            
            return {
                'passed': False,
                'reason': reason,
                'suggested_answer': suggested_answer
            }
        
        # Check for explicit VALID markers
        last_lines = '\n'.join(validation_upper.split('\n')[-10:])
        if 'VALID' in last_lines and 'INVALID' not in last_lines:
            return {
                'passed': True,
                'reason': 'All validation checks passed'
            }
        
        # Also check for PASS indicators
        if 'collinearity status: pass' in validation_lower and 'match status: match' in validation_lower:
            return {
                'passed': True,
                'reason': 'Geometric tests passed'
            }
        
        # Check for geometric test failures
        if 'collinearity status: fail' in validation_lower or 'match status: mismatch' in validation_lower:
            reason = "Geometric alignment test failed"
            
            # Extract closest mode
            closest_match = re.search(r'closest scale line:\s*["\']?([^"\'\n]+)["\']?', validation_text, re.IGNORECASE)
            suggested_answer = closest_match.group(1).strip() if closest_match else None
            
            return {
                'passed': False,
                'reason': reason,
                'suggested_answer': suggested_answer
            }
        
        # Default: if uncertain, treat as failed to be safe
        logger.warning("Validation result unclear, treating as failed for safety")
        return {
            'passed': False,
            'reason': 'Validation result unclear or ambiguous'
        }
    
    def _synthesize_final_answer(
        self,
        stage2_answer: str,
        stage3_validation: str,
        stage1_rules: str
    ) -> Dict[str, Any]:
        """
        Synthesize results from all stages to derive final answer
        
        Args:
            stage2_answer: Stage 2 answer
            stage3_validation: Stage 3 validation result
            stage1_rules: Stage 1 rules
            
        Returns:
            Dictionary containing final answer and confidence
        """
        confidence = 0.7  # Base confidence
        
        # Clean answer
        clean_answer = stage2_answer.strip()
        if clean_answer.startswith('"') and clean_answer.endswith('"'):
            clean_answer = clean_answer[1:-1]
        
        # Analyze validation result
        validation_lower = stage3_validation.lower()
        if 'valid' in validation_lower or 'yes' in validation_lower:
            confidence += 0.2
            final_answer = clean_answer
        elif 'invalid' in validation_lower or 'no' in validation_lower:
            confidence -= 0.3
            # Try to extract correct answer from validation result
            import re
            match = re.search(r'should be\s*"?([^"\n]+)"?', validation_lower)
            if match:
                final_answer = match.group(1).strip()
                confidence += 0.1
            else:
                final_answer = f"Uncertain: {clean_answer}"
        else:
            # Validation result unclear, keep original answer but reduce confidence
            confidence -= 0.1
            final_answer = clean_answer
        
        # Rule quality assessment
        rules_lower = stage1_rules.lower()
        if any(keyword in rules_lower for keyword in ['pointer', 'green line', 'center', 'endpoint', 'extension line', 'indicator', 'scale']):
            confidence += 0.1
        
        # Normalize confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'answer': final_answer,
            'confidence': confidence
        }
