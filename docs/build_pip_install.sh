#!/bin/bash
python3 -m pip uninstall -y autoapex
python3 -m pip uninstall -y autoapex.fair
python3 -m pip uninstall -y autoapex.eda
python3 -m pip uninstall -y autoapex.timeseries
python3 -m pip uninstall -y autoapex.multimodal
python3 -m pip uninstall -y autoapex.tabular
python3 -m pip uninstall -y autoapex.features
python3 -m pip uninstall -y autoapex.core
python3 -m pip uninstall -y autoapex.common

cd common/
python3 -m pip install -e .
cd ..

cd core/
python3 -m pip install -e .[all]
cd ..

cd features/
python3 -m pip install -e .
cd ..

cd tabular/
# Python 3.7 bug workaround: https://github.com/python/typing/issues/573
python3 -m pip uninstall -y typing
python3 -m pip install -e .[all,tests]
cd ..

cd multimodal/
python3 -m pip install -e .
cd ..

cd timeseries/
python3 -m pip install -e .[all,tests]
cd ..

cd eda/
python3 -m pip install -e .[tests]
# Resolve awscli and tox conflict
python3 -m pip install "colorama<0.4.5,>=0.2.5"
cd ..

cd autoapex/
python3 -m pip install -e .
cd ..
