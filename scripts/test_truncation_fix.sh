#!/bin/bash
# Test script to verify reasoning truncation fix
# This script runs evaluation on a small sample and checks if reasoning is complete

set -e

echo "=========================================="
echo "Testing Reasoning Truncation Fix"
echo "=========================================="
echo ""

# Configuration
DATASET_DIR="${1:-data/test}"
OUTPUT_DIR="output/truncation_test_$(date +%Y%m%d_%H%M%S)"

echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Run evaluation with new config
echo "Step 1: Running evaluation with max_tokens=4096..."
python scripts/washer_knob_eval_bbox.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --use_bbox \
    --save_intermediate_images \
    --max_tokens 4096 \
    --timeout 300 \
    --max_samples 2 \
    --log_level INFO

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed"
    exit 1
fi

echo ""
echo "Step 2: Checking results..."
echo ""

# Check if results exist
RESULT_FILE="$OUTPUT_DIR/results.jsonl"
if [ ! -f "$RESULT_FILE" ]; then
    echo "❌ Results file not found: $RESULT_FILE"
    exit 1
fi

# Check JSON files in intermediate_images
JSON_FILES=$(find "$OUTPUT_DIR/intermediate_images" -name "*_complete_results.json" 2>/dev/null | head -1)

if [ -z "$JSON_FILES" ]; then
    echo "⚠️  Warning: No complete_results.json files found"
else
    echo "✅ Found complete results JSON files"
    echo ""
    
    # Check reasoning lengths
    echo "Step 3: Analyzing reasoning completeness..."
    echo ""
    
    for json_file in $(find "$OUTPUT_DIR/intermediate_images" -name "*_complete_results.json" | head -2); do
        filename=$(basename "$json_file")
        echo "File: $filename"
        
        # Check stage1_rules length
        stage1_len=$(jq -r '.stage1_rules | length' "$json_file" 2>/dev/null || echo "0")
        echo "  Stage1 length: $stage1_len chars"
        
        if [ "$stage1_len" -lt 500 ]; then
            echo "  ⚠️  Stage1 seems too short (< 500 chars)"
        else
            echo "  ✅ Stage1 length looks good"
        fi
        
        # Check stage3_validation length
        stage3_len=$(jq -r '.stage3_validation | length' "$json_file" 2>/dev/null || echo "0")
        echo "  Stage3 length: $stage3_len chars"
        
        if [ "$stage3_len" -lt 500 ]; then
            echo "  ⚠️  Stage3 seems too short (< 500 chars)"
        else
            echo "  ✅ Stage3 length looks good"
        fi
        
        # Check for truncation markers
        has_dots=$(jq -r '.stage1_rules' "$json_file" 2>/dev/null | grep -c '\.\.\.$' || echo "0")
        if [ "$has_dots" -gt 0 ]; then
            echo "  ❌ Found truncation marker '...' in stage1"
        else
            echo "  ✅ No truncation markers found"
        fi
        
        echo ""
    done
fi

# Check log for response lengths
LOG_FILE="$OUTPUT_DIR/logs/eval.log"
if [ -f "$LOG_FILE" ]; then
    echo "Step 4: Checking logs..."
    echo ""
    
    # Check for response length logs
    if grep -q "response length" "$LOG_FILE"; then
        echo "✅ Response length logging is working"
        echo ""
        echo "Sample response lengths:"
        grep "response length" "$LOG_FILE" | head -6
    else
        echo "⚠️  No response length logs found"
    fi
    
    echo ""
    
    # Check for truncation warnings
    if grep -q "truncated\|too short" "$LOG_FILE"; then
        echo "⚠️  Found truncation warnings:"
        grep -i "truncated\|too short" "$LOG_FILE"
    else
        echo "✅ No truncation warnings"
    fi
    
    echo ""
    
    # Check for parsing errors
    if grep -q "Could not parse" "$LOG_FILE"; then
        echo "⚠️  Found parsing warnings:"
        grep "Could not parse" "$LOG_FILE" | head -5
    else
        echo "✅ No parsing errors"
    fi
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "To manually inspect results:"
echo "  1. Check JSON files: ls $OUTPUT_DIR/intermediate_images/*_complete_results.json"
echo "  2. View logs: cat $OUTPUT_DIR/logs/eval.log"
echo "  3. Check images: ls $OUTPUT_DIR/intermediate_images/*_auxiliary_lines.jpg"
echo ""
echo "To verify no truncation:"
echo "  jq '.stage1_rules, .stage3_validation' $OUTPUT_DIR/intermediate_images/*_complete_results.json | less"
echo ""

echo "✅ Test completed"

