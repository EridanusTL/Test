# !/bin/bash
export PROJECT_ROOT_DIR=$(pwd)
# docker 
export LD_LIBRARY_PATH=$(pwd)/install/lib:$LD_LIBRARY_PATH
export AMENT_PREFIX_PATH=$(pwd)/install:$AMENT_PREFIX_PATH

# physics
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
export AMENT_PREFIX_PATH=$(pwd):$AMENT_PREFIX_PATH

export LD_LIBRARY_PATH=$(pwd)/install_arm64/lib:$LD_LIBRARY_PATH
export AMENT_PREFIX_PATH=$(pwd)/install_arm64:$AMENT_PREFIX_PATH
export AMENT_PREFIX_PATH=$(pwd)/install_arm64/lib:$AMENT_PREFIX_PATH

export PYTHONPATH=$(pwd)/install/lib/python3.8/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/torch/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4