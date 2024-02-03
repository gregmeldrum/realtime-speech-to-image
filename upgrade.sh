#!/bin/bash

set -eu
cd "$(dirname $0)"

source .venv/bin/activate

pip3 install --upgrade \
    accelerate \
    diffusers \
    gradio \
    optimum \
    torch \
    transformers

pip3 freeze > requirements.txt
