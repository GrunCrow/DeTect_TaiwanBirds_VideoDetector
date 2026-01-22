#!/bin/bash

# Inference script runner for DeTect Taiwan Birds Video Detector
# Now with threshold optimization support!

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DeTect Inference Runner${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if model path is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}No model path provided. Searching for available models...${NC}\n"
    
    if [ -d "DeTect-BMMS/runs" ]; then
        echo -e "${GREEN}Available trained models:${NC}"
        find DeTect-BMMS/runs -name "best.pt" -type f | while read -r model; do
            run_dir=$(dirname $(dirname "$model"))
            echo "  - $run_dir"
        done
        echo ""
    fi
    
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 <model_path> [options]"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Standard inference"
    echo "  $0 DeTect-BMMS/runs/yolov26s-default-singlecls-bgundersampled-DeTect1000v1_"
    echo ""
    echo "  # With threshold optimization (finds best conf threshold)"
    echo "  $0 DeTect-BMMS/runs/yolov26s-default-singlecls-bgundersampled-DeTect1000v1_ --optimize"
    echo ""
    echo "  # Validation only with optimization"
    echo "  $0 DeTect-BMMS/runs/yolov26n-default-singlecls-bgundersampled-DeTect1000v1_ --optimize --val-only"
    echo ""
    echo "  # Custom thresholds"
    echo "  $0 DeTect-BMMS/runs/yolov26s-default-singlecls-bgundersampled-DeTect1000v1_ --conf 0.1 --iou 0.5"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --optimize       Use threshold optimization (finds best conf threshold from val set)"
    echo "  --val-only       Run inference only on validation set"
    echo "  --test-only      Run inference only on test set (not compatible with --optimize)"
    echo "  --conf FLOAT     Confidence threshold (default: 0.25, ignored if --optimize)"
    echo "  --iou FLOAT      IoU threshold for NMS (default: 0.45)"
    echo "  --no-wandb       Disable wandb logging"
    echo ""
    echo -e "${BLUE}What is threshold optimization?${NC}"
    echo "  1. Runs validation with very low confidence (0.01) to capture all predictions"
    echo "  2. Analyzes predictions to find optimal confidence threshold (maximizes F1)"
    echo "  3. Re-runs validation with optimal threshold"
    echo "  4. Applies optimal threshold to test set"
    echo "  5. Generates analysis plots and bias reports"
    echo ""
    exit 1
fi

MODEL_PATH=$1
shift

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path does not exist: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$MODEL_PATH/weights/best.pt" ]; then
    echo -e "${RED}Error: best.pt not found in $MODEL_PATH/weights/${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Model found: $MODEL_PATH${NC}"
echo -e "${GREEN}✓ Using weights: $MODEL_PATH/weights/best.pt${NC}\n"

# Check if --optimize flag is present
if [[ " $@ " =~ " --optimize " ]]; then
    # Remove --optimize from arguments
    ARGS="${@/--optimize/}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}   THRESHOLD OPTIMIZATION MODE${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}\n"
    
    python inference_optimized.py \
        --model-path "$MODEL_PATH" \
        --data cfg/datasets/DeTect.yaml \
        --project DeTect-BMMS \
        $ARGS
else
    # Standard inference
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}   STANDARD INFERENCE MODE${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}\n"
    
    python inference.py \
        --model-path "$MODEL_PATH" \
        --data cfg/datasets/DeTect.yaml \
        --project DeTect-BMMS \
        "$@"
fi

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Inference completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo -e "${BLUE}Results saved to:${NC}"
    echo "  Validation: DeTect-BMMS/inference/val/$MODEL_NAME"
    echo "  Test: DeTect-BMMS/inference/test/$MODEL_NAME"
    
    if [[ " $@ " =~ " --optimize " ]]; then
        echo "  Analysis: DeTect-BMMS/inference/threshold_analysis/$MODEL_NAME"
    fi
    echo ""
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Inference failed!${NC}"
    echo -e "${RED}========================================${NC}\n"
    exit 1
fi
