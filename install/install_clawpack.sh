#!/bin/bash

python3 -m pip install --src=$SOFTWARE/clawpack_src --user -e \
    git+https://github.com/clawpack/clawpack.git@v5.7.0#egg=clawpack-v5.7.0
