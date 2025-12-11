#!/usr/bin/env python3
"""
Test visualization pipeline - verify that geometric info is correctly parsed and drawn
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.visualization import parse_geometric_info_from_rules, save_intermediate_images
from src.utils.visualization_draw import draw_auxiliary_lines_on_image
from PIL import Image


def test_parsing():
    """Test 1: Verify geometric info parsing"""
    print("="*80)
    print("TEST 1: Geometric Info Parsing")
    print("="*80)
    
    # Load a sample result
    result_file = project_root / "output/washer_knob_eval/intermediate_images/knob_with_status_1_complete_results.json"
    
    if not result_file.exists():
        print(f"❌ Result file not found: {result_file}")
        print("Please run evaluation first: bash scripts/run_eval.sh")
        return False
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    stage1_rules = data.get('stage1_rules', '')
    
    print(f"\nStage1 rules length: {len(stage1_rules)} chars")
    print(f"Contains 'CIRCULAR ELEMENT GEOMETRY': {'CIRCULAR ELEMENT GEOMETRY' in stage1_rules}")
    print(f"Contains 'POINTER': {'POINTER' in stage1_rules}")
    print(f"Contains 'POSITION LABEL ANGLES': {'POSITION LABEL ANGLES' in stage1_rules}")
    
    # Parse geometric info
    geo_info = parse_geometric_info_from_rules(stage1_rules)
    
    print("\n" + "-"*80)
    print("Parsed Geometric Info:")
    print("-"*80)
    print(f"  Knob center: {geo_info['knob_center']}")
    print(f"  Knob radius: {geo_info['knob_radius']}")
    print(f"  Pointer angle: {geo_info['red_pointer_angle']}")
    print(f"  Label angles: {len(geo_info['green_scale_lines'])} found")
    
    if geo_info['green_scale_lines']:
        print("\n  Sample labels:")
        for label_info in geo_info['green_scale_lines'][:3]:
            print(f"    - {label_info['label']}: {label_info['angle']:.1f}°")
    
    # Verify parsing success
    success = all([
        geo_info['knob_center'] is not None,
        geo_info['knob_radius'] is not None,
        geo_info['red_pointer_angle'] is not None,
        len(geo_info['green_scale_lines']) > 0
    ])
    
    if success:
        print("\n✅ TEST 1 PASSED: All geometric info successfully parsed")
        
        # Verify values are reasonable
        center_x, center_y = geo_info['knob_center']
        if 200 < center_x < 600 and 300 < center_y < 500:
            print(f"✅ Center coordinates look reasonable: ({center_x}, {center_y})")
        else:
            print(f"⚠️  Warning: Center coordinates may be unusual: ({center_x}, {center_y})")
        
        if 30 < geo_info['knob_radius'] < 150:
            print(f"✅ Radius looks reasonable: {geo_info['knob_radius']}")
        else:
            print(f"⚠️  Warning: Radius may be unusual: {geo_info['knob_radius']}")
    else:
        print("\n❌ TEST 1 FAILED: Failed to parse complete geometric info")
    
    print("")
    return success, geo_info


def test_drawing(geo_info):
    """Test 2: Verify drawing function works"""
    print("="*80)
    print("TEST 2: Auxiliary Lines Drawing")
    print("="*80)
    
    # Load original image
    image_path = project_root / "data/test/with_status/knob_with_status_1.png"
    
    # Try multiple possible paths
    possible_paths = [
        project_root / "data/test/with_status/knob_with_status_1.png",
        Path("/mnt/data/datasets/washing_machine_eval_data/knob/with_status/knob_with_status_1.png"),
        project_root / "output/washer_knob_eval/intermediate_images/knob_with_status_1_auxiliary_lines.jpg"
    ]
    
    image_path = None
    for path in possible_paths:
        if path.exists():
            image_path = path
            break
    
    if not image_path:
        print(f"❌ Could not find test image")
        print("Searched in:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    print(f"\nUsing image: {image_path}")
    
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        print(f"Image size: {img.size}")
        
        # Draw auxiliary lines
        img_with_lines = draw_auxiliary_lines_on_image(
            img,
            knob_center=tuple(geo_info['knob_center']),
            knob_radius=geo_info['knob_radius'],
            pointer_angle=geo_info['red_pointer_angle'],
            label_angles=geo_info['green_scale_lines']
        )
        
        # Save test output
        test_output_dir = project_root / "output/visualization_test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = test_output_dir / "test_auxiliary_lines.jpg"
        img_with_lines.save(output_path, quality=95)
        
        print(f"\n✅ TEST 2 PASSED: Successfully drew auxiliary lines")
        print(f"   Output saved to: {output_path}")
        print(f"   Please open this image to verify:")
        print(f"   - Blue circle on knob")
        print(f"   - Red line for pointer")
        print(f"   - Green lines for labels")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test 3: Verify full save_intermediate_images pipeline"""
    print("="*80)
    print("TEST 3: Full Pipeline")
    print("="*80)
    
    # Load test data
    result_file = project_root / "output/washer_knob_eval/intermediate_images/knob_with_status_1_complete_results.json"
    
    if not result_file.exists():
        print(f"❌ Result file not found: {result_file}")
        return False
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Find original image
    possible_paths = [
        project_root / "data/test/with_status/knob_with_status_1.png",
        Path("/mnt/data/datasets/washing_machine_eval_data/knob/with_status/knob_with_status_1.png"),
    ]
    
    image_path = None
    for path in possible_paths:
        if path.exists():
            image_path = path
            break
    
    if not image_path:
        print(f"❌ Could not find original image")
        return False
    
    # Run full pipeline
    test_output_dir = project_root / "output/visualization_test"
    
    try:
        save_intermediate_images(
            image_path=str(image_path),
            results=results,
            output_dir=str(test_output_dir),
            image_name="knob_with_status_1.png"
        )
        
        # Verify outputs
        aux_image = test_output_dir / "knob_with_status_1_auxiliary_lines.jpg"
        complete_json = test_output_dir / "knob_with_status_1_complete_results.json"
        
        if aux_image.exists() and complete_json.exists():
            print(f"\n✅ TEST 3 PASSED: Full pipeline successful")
            print(f"   Auxiliary image: {aux_image}")
            print(f"   Complete JSON: {complete_json}")
            
            # Verify answer field
            with open(complete_json, 'r') as f:
                saved_data = json.load(f)
            
            answer = saved_data.get('answer', '')
            print(f"\n   Answer field: '{answer}'")
            print(f"   Answer length: {len(answer)} chars")
            
            if len(answer) < 100:
                print(f"   ✅ Answer is concise (<100 chars)")
            else:
                print(f"   ⚠️  Answer seems long (>100 chars)")
            
            return True
        else:
            print(f"\n❌ TEST 3 FAILED: Output files not created")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("VISUALIZATION PIPELINE TEST SUITE")
    print("="*80)
    print("")
    
    # Test 1: Parsing
    success1, geo_info = test_parsing()
    if not success1:
        print("\n⚠️  Parsing test failed, skipping drawing tests")
        return 1
    
    print("")
    
    # Test 2: Drawing
    success2 = test_drawing(geo_info)
    
    print("")
    
    # Test 3: Full pipeline
    success3 = test_full_pipeline()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"  Test 1 (Parsing):      {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Test 2 (Drawing):      {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"  Test 3 (Full Pipeline): {'✅ PASS' if success3 else '❌ FAIL'}")
    print("")
    
    if all([success1, success2, success3]):
        print("🎉 ALL TESTS PASSED!")
        print("")
        print("The visualization pipeline is working correctly:")
        print("  1. VLM outputs geometric analysis in stage1")
        print("  2. Code parses this analysis")
        print("  3. Code draws auxiliary lines on original image")
        print("  4. Saves annotated image for visual reasoning")
        print("")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Please check the errors above")
        print("")
        return 1


if __name__ == "__main__":
    sys.exit(main())

