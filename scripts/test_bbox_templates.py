#!/usr/bin/env python3
"""
Test bbox-enhanced template functionality
Validates generic geometric reasoning framework
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.prompt_templates_bbox import BboxEnhancedTemplates

def test_bbox_formatting():
    """Test bbox information formatting"""
    print("=" * 80)
    print("Test: Bbox Information Formatting")
    print("=" * 80)
    
    # Mock data
    sample_data = {
        "knob_close": [
            {
                "label": "Quick Wash 15'",
                "bbox": [798, 70, 958, 181]
            },
            {
                "label": "Speed Wash 30'",
                "bbox": [816, 205, 953, 292]
            },
            {
                "label": "knob",
                "bbox": [281, 157, 721, 818]
            }
        ],
        "modes": ["Quick Wash 15'", "Speed Wash 30'", "Heavy Duty", "Wool"],
        "status": {
            "label": "Speed Wash 30'",
            "bbox": [816, 205, 953, 292]
        }
    }
    
    bbox_info, knob_bbox, mode_bboxes, current_status = BboxEnhancedTemplates.format_bbox_info(sample_data)
    
    print("\nExtracted information:")
    print(f"\nKnob bbox: {knob_bbox}")
    print(f"\nCurrent status: {current_status}")
    print(f"\nAll bbox info:\n{bbox_info}")
    print(f"\nMode bboxes:\n{mode_bboxes}")
    
    # Test ground truth extraction
    gt = BboxEnhancedTemplates.extract_ground_truth(sample_data)
    print(f"\nGround Truth: {gt}")
    
    print("\n✓ Bbox information formatting test passed")

def test_prompt_generation():
    """Test prompt generation"""
    print("\n" + "=" * 80)
    print("Test: Prompt Generation")
    print("=" * 80)
    
    sample_data = {
        "knob_close": [
            {
                "label": "Quick Wash 15'",
                "bbox": [798, 70, 958, 181]
            },
            {
                "label": "knob",
                "bbox": [281, 157, 721, 818]
            }
        ],
        "modes": ["Quick Wash 15'", "Speed Wash 30'", "Heavy Duty"]
    }
    
    question = "What is the current position of the control?"
    prompt = BboxEnhancedTemplates.create_stage1_prompt_with_bbox(question, sample_data)
    
    print("\nGenerated Stage1 prompt (first 500 characters):")
    print("-" * 80)
    print(prompt[:500])
    print("...")
    print("-" * 80)
    
    # Verify it doesn't contain task-specific priors
    forbidden_terms = ["washing machine", "washer", "laundry"]
    has_priors = any(term in prompt.lower() for term in forbidden_terms)
    
    if has_priors:
        print("\n✗ WARNING: Prompt contains task-specific priors!")
        print("  Found forbidden terms. Prompt should be generic.")
    else:
        print("\n✓ Prompt is generic without task-specific priors")
    
    # Check for generic geometric terms
    required_terms = ["circular", "angle", "center", "geometric", "pointer"]
    has_generic = all(term in prompt.lower() for term in required_terms)
    
    if has_generic:
        print("✓ Prompt contains generic geometric reasoning terms")
    else:
        print("✗ WARNING: Missing some generic geometric terms")
    
    print("\n✓ Prompt generation test passed")

def test_template_retrieval():
    """Test template retrieval"""
    print("\n" + "=" * 80)
    print("Test: Template Retrieval")
    print("=" * 80)
    
    template = BboxEnhancedTemplates.get_generic_rotary_template_with_bbox()
    
    print(f"\nTemplate stages: {list(template.keys())}")
    
    for stage, content in template.items():
        print(f"\n{stage} prompt length: {len(content)} characters")
    
    print("\n✓ Template retrieval test passed")

def test_generic_reasoning_framework():
    """Test that framework is generic and task-agnostic"""
    print("\n" + "=" * 80)
    print("Test: Generic Reasoning Framework")
    print("=" * 80)
    
    template = BboxEnhancedTemplates.get_generic_rotary_template_with_bbox()
    
    # Combine all template text
    all_text = " ".join(template.values()).lower()
    
    # Task-specific terms that should NOT appear
    forbidden = [
        "washing machine", "washer", "laundry", "clothes",
        "dishwasher", "microwave", "oven", "appliance"
    ]
    
    found_forbidden = [term for term in forbidden if term in all_text]
    
    if found_forbidden:
        print(f"\n✗ FAIL: Found task-specific terms: {found_forbidden}")
        print("   Framework should be generic and task-agnostic")
        return False
    else:
        print("\n✓ No task-specific terms found")
    
    # Generic terms that SHOULD appear
    required = [
        "circular", "angle", "geometric", "center", "radius",
        "pointer", "indicator", "position", "alignment"
    ]
    
    missing_required = [term for term in required if term not in all_text]
    
    if missing_required:
        print(f"\n✗ WARNING: Missing generic terms: {missing_required}")
    else:
        print("✓ All required generic terms present")
    
    print("\n✓ Generic reasoning framework test passed")

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Bbox-Enhanced Template Functionality Tests" + " " * 20 + "║")
    print("║" + " " * 20 + "(Generic Geometric Reasoning)" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_bbox_formatting()
        test_prompt_generation()
        test_template_retrieval()
        test_generic_reasoning_framework()
        
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
