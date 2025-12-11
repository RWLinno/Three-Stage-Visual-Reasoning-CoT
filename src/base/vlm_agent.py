"""
VLM Agent EAS - Alibaba Cloud EAS Model Deployment Service Agent
Call large models through Alibaba Cloud EAS API for inference
Supports multimodal input (image + text)
"""
import os
import base64
import requests
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)


class VLMAgentEAS:
    """
    Alibaba Cloud EAS VLM Agent
    Call EAS-deployed large models through HTTP API
    """
    
    def __init__(
        self,
        base_url: str,
        token: str,
        model_name: str = "Qwen3-VL-235B-A22B-Instruct-FP8",
        max_tokens: int = 256,
        timeout: int = 60
    ):
        """
        Initialize EAS VLM Agent
        
        Args:
            base_url: EAS service base URL
            token: Authentication token
            model_name: Model name
            max_tokens: Maximum generation tokens
            timeout: Request timeout (seconds)
        """
        self.base_url = base_url.rstrip('/')
        # Check if base_url already contains /v1/chat/completions
        if '/v1/chat/completions' in self.base_url:
            self.api_url = self.base_url
        elif '/api/predict/' in self.base_url:
            # EAS predict endpoint format: add /v1/chat/completions
            self.api_url = f"{self.base_url}/v1/chat/completions"
        else:
            # Standard format: add /v1/chat/completions path if not present
            self.api_url = f"{self.base_url}/v1/chat/completions"
        self.token = token
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # EAS API uses Bearer authentication, format: "Bearer {token}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        logger.info(f"Initialized EAS VLM Agent: {self.model_name}")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Max tokens: {self.max_tokens}, Timeout: {self.timeout}s")
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert image file to base64 encoding
        
        Args:
            image_path: Image file path
            
        Returns:
            Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95)
                img_bytes = buffer.getvalue()
                
                # Convert to base64, use data URI format
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Failed to convert image to base64 {image_path}: {e}")
            raise
    
    def _build_message_content(
        self,
        prompt_text: str,
        image_path: Optional[str] = None,
        depth_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Build message content, supports multiple images
        
        Args:
            prompt_text: Prompt text
            image_path: RGB image path
            depth_path: Depth map path (optional)
            
        Returns:
            Message content list
        """
        content = []
        
        if image_path:
            try:
                img_base64 = self._image_to_base64(image_path)
                # Log image info for debugging
                img_size = len(img_base64)
                logger.info(f"Added image to content: {image_path}, base64 size: {img_size} chars")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64
                    }
                })
            except Exception as e:
                logger.error(f"Failed to process RGB image: {e}")
                raise
        
        if depth_path:
            try:
                depth_base64 = self._image_to_base64(depth_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": depth_base64
                    }
                })
                logger.debug(f"Added depth image to content: {depth_path}")
            except Exception as e:
                logger.error(f"Failed to process depth image: {e}")
        
        content.append({
            "type": "text",
            "text": prompt_text
        })
        
        return content
    
    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> str:
        """
        Call EAS API
        
        Args:
            messages: Message list
            max_retries: Maximum retry count
            
        Returns:
            Model response text
        """
        # EAS API format: include model field and messages
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Calling EAS API (attempt {attempt + 1}/{max_retries})")
                logger.debug(f"Payload model: {self.model_name}, messages count: {len(messages)}")
                # Log message content structure for debugging
                if messages and len(messages) > 0:
                    user_msg = messages[-1] if messages[-1].get('role') == 'user' else messages[0]
                    if isinstance(user_msg.get('content'), list):
                        content_types = [item.get('type') for item in user_msg.get('content', [])]
                        logger.debug(f"User message content types: {content_types}")
                        image_count = sum(1 for item in user_msg.get('content', []) if item.get('type') == 'image_url')
                        logger.info(f"Number of images in request: {image_count}")
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                logger.debug(f"API response status code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    # EAS API may return different formats, try multiple parsing methods
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        return content.strip()
                    elif 'output' in result:
                        # EAS may directly return output field
                        output = result['output']
                        if isinstance(output, str):
                            return output.strip()
                        elif isinstance(output, dict) and 'text' in output:
                            return output['text'].strip()
                    elif 'text' in result:
                        return result['text'].strip()
                    else:
                        logger.error(f"API response format abnormal: {result}")
                        return f"Error: Invalid response format"
                else:
                    error_msg = f"API request failed (status code: {response.status_code})"
                    error_text = response.text[:200] if response.text else "No response content"
                    logger.error(f"{error_msg}, response: {error_text}")
                    if attempt < max_retries - 1:
                        logger.debug(f"Waiting to retry...")
                        continue
                    return f"Error: {error_msg}"
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                return "Error: API request timeout"
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request exception (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    continue
                return f"Error: {str(e)}"
                
            except Exception as e:
                logger.error(f"API call exception (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
    def inference_single(
        self,
        prompt_text: str,
        image_path: Optional[str] = None,
        depth_path: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Single inference
        
        Args:
            prompt_text: Prompt text
            image_path: Image path (optional)
            depth_path: Depth map path (optional)
            max_retries: Maximum retry count
            
        Returns:
            Response text
        """
        try:
            content = self._build_message_content(prompt_text, image_path, depth_path)
            
            messages = [
                {"role": "system", "content": "You are a multi-modal VQA Data Quality Assessment Expert, able to accurately assess the quality of image and dialogue data."},
                {"role": "user", "content": content}
            ]
            
            logger.debug(f"Inference request - image_path: {image_path}, content items: {len(content)}")
            
            # Call API
            response = self._call_api(messages, max_retries)
            return response
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"Error: {str(e)}"
    
    def inference_batch(
        self,
        prompts: List[Tuple[str, Any]],
        max_retries: int = 3
    ) -> List[str]:
        """
        Batch inference (EAS API usually doesn't support true batch, use sequential calls here)
        
        Args:
            prompts: Prompt list, each element is (prompt_text, image_data)
                    image_data can be None, image path string, or [image_path, depth_path] list
            max_retries: Maximum retry count
            
        Returns:
            Response text list
        """
        responses = []
        
        for i, (prompt_text, image_data) in enumerate(prompts):
            logger.debug(f"Processing batch inference {i+1}/{len(prompts)}")
            
            # Parse image_data
            image_path = None
            depth_path = None
            
            if image_data is not None:
                if isinstance(image_data, str):
                    # Single image path
                    image_path = image_data
                elif isinstance(image_data, list) and len(image_data) > 0:
                    # Image list (RGB + Depth)
                    image_path = image_data[0]
                    if len(image_data) > 1:
                        depth_path = image_data[1]
            
            # Single inference
            response = self.inference_single(
                prompt_text=prompt_text,
                image_path=image_path,
                depth_path=depth_path,
                max_retries=max_retries
            )
            responses.append(response)
        
        return responses
    
    def evaluate_sample(
        self,
        question: str,
        image_path: str,
        depth_path: Optional[str] = None,
        answer: Optional[str] = None,
        system_prompt: str = "",
    ) -> Tuple[int, str]:
        """
        Evaluate a single sample
        
        Note: Only pass answer parameter when system_prompt explicitly requires answer evaluation
        Otherwise the model will only generate answer and score based on question and image
        
        Args:
            question: Question text
            image_path: Image path
            depth_path: Depth map path (optional)
            answer: Answer text (optional, only used when include_answer=True)
            system_prompt: System prompt
            
        Returns:
            (score, model_response) - Score and model-generated response
        """
        # Build prompt: only include answer when explicitly passed
        # This ensures the model generates its own answer based on image and question when it shouldn't see the answer
        if answer:
            prompt_text = f"{system_prompt}{question}\n\nExpected answer: {answer}"
        else:
            # Don't include answer, let model generate answer based on image and question
            prompt_text = f"{system_prompt}{question}"
        
        # Inference - model will return score and/or answer
        response = self.inference_single(prompt_text, image_path, depth_path)
        
        # Extract score
        try:
            # Try to extract numeric score
            score_str = response.strip().split()[0]
            score = int(score_str)
            if not (0 <= score <= 10):
                logger.warning(f"Score out of range: {score}, using original response")
                score = -1
        except (ValueError, IndexError):
            logger.warning(f"Unable to parse score: {response}")
            score = -1
        
        return score, response
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        system_prompt: str = "",
        include_answer: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch evaluate samples
        
        Args:
            samples: Sample list, each sample contains question, image_path, answer (optional)
            system_prompt: System prompt
            include_answer: Whether to include answer in evaluation
            
        Returns:
            Evaluation result list, each result contains score, raw_response
        """
        # Prepare batch inference data
        prompts = []
        valid_samples = []
        
        for sample in samples:
            question = sample.get('question', '')
            image_path = sample.get('image_path', '')
            depth_path = sample.get('depth_path', '')
            answer = sample.get('answer', '') if include_answer else None
            
            if not question or not image_path:
                logger.warning(f"Skipping invalid sample: {sample}")
                continue
            
            # Build prompt
            if answer and include_answer:
                prompt_text = f"{system_prompt}{question}\n\nExpected answer: {answer}"
            else:
                prompt_text = f"{system_prompt}{question}"
            
            # Prepare image data
            if depth_path:
                image_data = [image_path, depth_path]
            else:
                image_data = image_path
            
            prompts.append((prompt_text, image_data))
            valid_samples.append(sample)
        
        # Batch inference
        if prompts:
            responses = self.inference_batch(prompts)
            
            # Parse results
            for sample, response in zip(valid_samples, responses):
                try:
                    # Extract score
                    score_str = response.strip().split()[0]
                    score = int(score_str)
                    if not (0 <= score <= 10):
                        score = -1
                except (ValueError, IndexError):
                    score = -1
                
                sample['score'] = score
                sample['raw_response'] = response
        
        return samples
    
    def __del__(self):
        """Clean up resources"""
        logger.debug("EAS VLM Agent resources released")
