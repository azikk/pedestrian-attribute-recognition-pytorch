#!/usr/bin/env python

python ./script/experiment/train_deepmar_resnet50.py \
    --sys_device_ids="(0,1)" \
    --dataset=peta \
    --partition_idx=0 \
    --test_split=test \
    --resize="(224,224)" \
    --exp_subpath=deepmar_resnet50 \
    --run=1 \
    --test_only=True \
    --load_model_weight=True \
    --model_weight_file='./model/ckpt_epoch150.pth'
