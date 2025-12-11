#!/bin/bash
# Complete evaluation script for rotary control recognition

set -e

# Configuration parameters
DATASET_DIR="${DATASET_DIR:-/mnt/data/datasets/washing_machine_eval_data/knob}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/washer_knob_eval}"
EAS_URL="${EAS_BASE_URL}"
EAS_TOKEN="${EAS_TOKEN}"
MODEL_NAME="${EAS_MODEL_NAME}"

# Evaluation parameters
SUBSET="${SUBSET:-both}"  # with_status, without_status, or both
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Leave empty for all samples, set number for quick test
QUESTION="${QUESTION:-What is the current position of the control?}"

echo "=================================="
echo "Rotary Control Recognition Evaluation - Bbox Enhanced"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Dataset directory: $DATASET_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL_NAME"
echo "  Evaluation subset: $SUBSET"
echo "  Processing mode: Sequential (single process, interruptible)"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max samples: $MAX_SAMPLES"
fi
echo ""
echo "📌 You can press Ctrl+C at any time to stop processing"
echo "   (Current sample will finish, then results will be saved)"
echo ""

# Check dataset directory
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory does not exist: $DATASET_DIR"
    exit 1
fi

# Check API configuration
if [ -z "$EAS_URL" ] || [ -z "$EAS_TOKEN" ]; then
    echo "Error: Please set EAS_URL and EAS_TOKEN environment variables"
    echo "Example:"
    echo "  export EAS_URL='your-eas-url'"
    echo "  export EAS_TOKEN='your-token'"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Starting evaluation..."
echo ""

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

python3 scripts/washer_knob_eval_bbox.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --question "$QUESTION" \
    --use_bbox \
    --subset "$SUBSET" \
    --eas_url "$EAS_URL" \
    --eas_token "$EAS_TOKEN" \
    --model_name "$MODEL_NAME" \
    --max_tokens 4096 \
    --timeout 300 \
    --save_intermediate_images \
    --log_level INFO \
    $MAX_SAMPLES_ARG

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "Evaluation failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "Generating HTML report..."
echo "=================================="
echo ""

# Generate HTML report
python3 scripts/generate_eval_report.py \
    --results "$OUTPUT_DIR/results.jsonl" \
    --metrics "$OUTPUT_DIR/eval_report.json" \
    --output "$OUTPUT_DIR/report.html"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Evaluation Complete!"
    echo "=================================="
    echo ""
    echo "Result files:"
    echo "  - Detailed results: $OUTPUT_DIR/results.jsonl"
    echo "  - Evaluation report: $OUTPUT_DIR/eval_report.json"
    echo "  - HTML report: $OUTPUT_DIR/report.html"
    echo "  - Log file: $OUTPUT_DIR/logs/eval.log"
    echo ""
    echo "Open $OUTPUT_DIR/report.html in browser to view detailed report"
else
    echo ""
    echo "HTML report generation failed!"
    exit 1
fi
