#!/bin/sh
MODEL="ai85cdnet"
DATASET="fish"
QUANTIZED_MODEL="../ai8x-training/logs/2025.11.10-050944/qat_best-quantized.pth.tar"

# evaluate scripts for cats vs dogs
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL  -8 --save-sample 1 --device MAX78000 --workers 0 "$@"

#evaluate scripts for kws
# python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"

