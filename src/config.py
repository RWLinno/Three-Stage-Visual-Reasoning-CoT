"""
Configuration Management - System configuration parameters
"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Config:
    """System configuration class"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # EAS API configuration
            'eas_base_url': os.getenv('EAS_BASE_URL', 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation'),
            'eas_token': os.getenv('EAS_TOKEN', ''),
            'eas_model_name': os.getenv('EAS_MODEL_NAME', 'Qwen3-VL-235B-A22B-Instruct-FP8'),
            
            # Processing configuration
            'num_processors': int(os.getenv('NUM_PROCESSORS', '4')),
            'batch_size': int(os.getenv('BATCH_SIZE', '2')),
            'max_tokens': int(os.getenv('MAX_TOKENS', '512')),
            'timeout': int(os.getenv('TIMEOUT', '120')),
            
            # Output configuration
            'save_intermediate_images': os.getenv('SAVE_INTERMEDIATE_IMAGES', 'true').lower() == 'true',
            
            # Logging configuration
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            
            # Task-specific configuration
            'washer_knob_question': (
                "How would you determine the current gear position of the washing machine, "
                "where the red line in the image corresponds to the gear indicator pointer. "
                "You should treat this as a clock hand, where the red point represents the pointer, "
                "and the surrounding text represents the scale marks. "
                "The text is connected to the knob perimeter (not the center) through green extension lines below. "
                "The mode is determined by which green scale line endpoint on the knob edge the pointer most likely connects to. "
                "The correct mode's text extension line and the red line can connect to point to the center. "
                "Remember that the green lines below the text correspond to the text. "
                "You should only focus on modes that have corresponding green lines, ignore others."
            )
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration validity"""
        required_fields = ['eas_token']
        
        for field in required_fields:
            if field in config and not config[field]:
                logger.warning(f"Configuration field '{field}' is empty")
        
        if config.get('num_processors', 0) < 1:
            logger.warning("Number of processors must be at least 1")
            config['num_processors'] = 1
        
        if config.get('batch_size', 0) < 1:
            logger.warning("Batch size must be at least 1")
            config['batch_size'] = 1
        
        return True
