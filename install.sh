#!/bin/bash
cd FastMinv
python setup.py install
cd ../MCGpu
python setup.py install
cd ../MCAcc/cuda
python setup.py install