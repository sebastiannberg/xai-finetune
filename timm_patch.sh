#!/bin/bash

timm_path_lib=/cluster/projects/uasc/sebastian/xai-finetune/venv/lib/python3.9/site-packages/timm
timm_path_lib64=/cluster/projects/uasc/sebastian/xai-finetune/venv/lib64/python3.9/site-packages/timm

# Copy to lib
cp timm_patch/swin_transformer.py $timm_path_lib/models/
cp timm_patch/helpers.py $timm_path_lib/models/layers/

# Copy to lib64
cp timm_patch/swin_transformer.py $timm_path_lib64/models/
cp timm_patch/helpers.py $timm_path_lib64/models/layers/

