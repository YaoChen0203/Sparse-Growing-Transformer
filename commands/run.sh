#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29605 scripts/train.py configs/OLMo-300M-Dynamic_Loop-HighEntropyHead-TopKLayers_3-LoopDepth_1-NumHead_2-EmaDecay_0.5_DecayWindow_250-FirstSelectionStep_250.yaml
