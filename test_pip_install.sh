#!/bin/bash

set -e
cur_dir=$(pwd)

# Python 2

rm -rf /tmp/clean-venv
virtualenv /tmp/clean-venv
cd /tmp/clean-venv
source bin/activate
pip install --upgrade pip
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl
pip install prettytensor
pip install nose
nosetests prettytensor

deactivate

cd "$cur_dir"
# Python 3

rm -rf /tmp/clean-venv
virtualenv -p python3 /tmp/clean-venv
cd /tmp/clean-venv
source bin/activate
pip install --upgrade pip
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl
pip install prettytensor
pip install nose
nosetests prettytensor

deactivate
rm -rf /tmp/clean-venv

cd "$cur_dir"
