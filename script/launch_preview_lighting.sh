#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
export ROBOTWIN_CAMERA_SHADER_DIR=${ROBOTWIN_CAMERA_SHADER_DIR:-default}

python3 script/preview_lighting.py "$@"
