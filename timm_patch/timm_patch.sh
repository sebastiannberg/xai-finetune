#!/bin/bash

# Only change these two
PYTHON_VERSION="3.10"
VENV_NAME="venv"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_PATH="${PROJECT_DIR}/${VENV_NAME}"
TIMM_PATH_LIB="${VENV_PATH}/lib/python${PYTHON_VERSION}/site-packages/timm"
TIMM_PATH_LIB64="${VENV_PATH}/lib64/python${PYTHON_VERSION}/site-packages/timm"

# Copy to lib
cp swin_transformer.py $TIMM_PATH_LIB/models/
cp helpers.py $TIMM_PATH_LIB/models/layers/

# Copy to lib64
cp swin_transformer.py $TIMM_PATH_LIB64/models/
cp helpers.py $TIMM_PATH_LIB64/models/layers/
