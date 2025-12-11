"""
Validation mechanism - Implements multi-stage validation logic
"""
import logging
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ValidationEngine:
    """
    Validation Engine
    
    Implements multiple validation strategies:
    1. Counter-question validation
    2. Visual reasoning validation
    3. Consistency validation
    """
    
    def __init__(self):
        """Initialize validation engine"""
        logger.info("Initialized validation engine")
    
    def counter_question_validation(
        self,
        original_answer: str,
        context: str,
        image_path: str,
        vlm_agent: Any
    ) -> Dict[str, Any]:
        """
        Counter-question validation
        
        Verify answer correctness by asking counter-questions
        """
        validation_prompt = (
            f"Original question context: {context}\n"
            f"Original answer: {original_answer}\n\n"
            f"Now please strictly answer based on image evidence: Is the current state really '{original_answer}'? "
            "Please only answer 'yes' or 'no', and briefly explain key evidence."
        )
        
        try:
            response = vlm_agent.inference_single(
                prompt_text=validation_prompt,
                image_path=image_path
            )
            
            # Parse validation result
            is_valid = self.parse_counter_question_response(response)
            
            return {
                'validation_type': 'counter_question',
                'prompt': validation_prompt,
                'response': response,
                'is_valid': is_valid,
                'confidence': 0.8 if is_valid else 0.2
            }
            
        except Exception as e:
            logger.error(f"Counter-question validation failed: {e}")
            return {
                'validation_type': 'counter_question',
                'error': str(e),
                'is_valid': False,
                'confidence': 0.0
            }
    
    def visual_reasoning_validation(
        self,
        answer: str,
        visual_context: str,
        image_path: str,
        vlm_agent: Any
    ) -> Dict[str, Any]:
        """
        Visual reasoning validation
        
        Require model to visually reason and verify answer
        """
        validation_prompt = (
            f"Assume the current state is: {answer}\n\n"
            "Please perform the following visual reasoning:\n"
            f"1. Imagine drawing a line from the knob center to the '{answer}' label\n"
            "2. Check if this line coincides with the actual red pointer\n"
            "3. Verify if the green scale lines support this conclusion\n\n"
            "Based on this visual reasoning, is the answer correct? Answer VALID or INVALID, and briefly explain."
        )
        
        try:
            response = vlm_agent.inference_single(
                prompt_text=validation_prompt,
                image_path=image_path
            )
            
            # Parse validation result
            is_valid = self.parse_visual_reasoning_response(response)
            
            return {
                'validation_type': 'visual_reasoning',
                'prompt': validation_prompt,
                'response': response,
                'is_valid': is_valid,
                'confidence': 0.9 if is_valid else 0.1
            }
            
        except Exception as e:
            logger.error(f"Visual reasoning validation failed: {e}")
            return {
                'validation_type': 'visual_reasoning',
                'error': str(e),
                'is_valid': False,
                'confidence': 0.0
            }
    
    def parse_counter_question_response(self, response: str) -> bool:
        """Parse counter-question validation response"""
        response_lower = response.lower().strip()
        
        # Check for explicit yes/no
        if any(keyword in response_lower for keyword in ['yes', 'correct', 'indeed', 'true']):
            return True
        if any(keyword in response_lower for keyword in ['no', 'not', 'wrong', 'incorrect', 'false']):
            return False
        
        # Check response start
        if response_lower.startswith(('yes', 'y')):
            return True
        if response_lower.startswith(('no', 'n')):
            return False
        
        # Default case - conservative handling
        logger.warning(f"Unable to clearly parse counter-question validation response: {response}")
        return False
    
    def parse_visual_reasoning_response(self, response: str) -> bool:
        """Parse visual reasoning validation response"""
        response_upper = response.upper().strip()
        return 'VALID' in response_upper
    
    def multi_stage_validation(
        self,
        stage2_answer: str,
        cot_results: Dict[str, Any],
        image_path: str,
        vlm_agent: Any
    ) -> Dict[str, Any]:
        """
        Multi-stage validation
        
        Combine multiple validation strategies
        """
        results = {
            'counter_question': None,
            'visual_reasoning': None,
            'final_validation': 'UNKNOWN',
            'overall_confidence': 0.0
        }
        
        try:
            # Counter-question validation
            cq_result = self.counter_question_validation(
                stage2_answer,
                cot_results['stage1_rules'],
                image_path,
                vlm_agent
            )
            results['counter_question'] = cq_result
            
            # Visual reasoning validation
            vr_result = self.visual_reasoning_validation(
                stage2_answer,
                cot_results['stage1_rules'],
                image_path,
                vlm_agent
            )
            results['visual_reasoning'] = vr_result
            
            # Synthesize validation results
            if cq_result.get('is_valid', False) and vr_result.get('is_valid', False):
                results['final_validation'] = 'VALID'
                results['overall_confidence'] = min(
                    cq_result.get('confidence', 0.8) + 0.1,
                    vr_result.get('confidence', 0.9) + 0.1,
                    1.0
                )
            elif not cq_result.get('is_valid', True) and not vr_result.get('is_valid', True):
                results['final_validation'] = 'INVALID'
                results['overall_confidence'] = max(
                    cq_result.get('confidence', 0.2) - 0.1,
                    vr_result.get('confidence', 0.1) - 0.1,
                    0.0
                )
            else:
                # Partial validation passed - needs further analysis
                results['final_validation'] = 'PARTIAL'
                results['overall_confidence'] = 0.6
            
        except Exception as e:
            logger.error(f"Multi-stage validation process error: {e}")
            results['final_validation'] = f'ERROR: {str(e)}'
            results['overall_confidence'] = 0.0
        
        return results
