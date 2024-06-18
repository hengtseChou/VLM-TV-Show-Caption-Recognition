#!/bin/bash
if ! which "virtualenv" >/dev/null 2>&1; then
    echo "virtualenv not installed. exiting"
    exit 1
fi
virtualenv .tinyllava_factory
source .tinyllava_factory/bin/activate
cd TinyLLaVA_Factory
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
cd ..
deactivate