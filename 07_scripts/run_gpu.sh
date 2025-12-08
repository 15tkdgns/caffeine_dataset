#!/bin/bash

# Define the python path for gemini_gpu environment
PYTHON_EXEC="/root/miniconda3/envs/gemini_gpu/bin/python"

# Set up environment variables
# Preload system libstdc++ to fix ABI mismatch
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Add nvidia libraries and cuml libraries to LD_LIBRARY_PATH
# We dynamically find the paths to ensure we get the correct ones
NVIDIA_LIB_PATH=$($PYTHON_EXEC -c "import os; import nvidia; print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
CUML_LIB_PATH=$($PYTHON_EXEC -c "import os; import cuml; print(os.path.join(os.path.dirname(cuml.__file__), '..', 'libcuml', 'lib64'))" 2>/dev/null)

if [ -z "$NVIDIA_LIB_PATH" ]; then
    echo "Error: Could not find nvidia package path."
    exit 1
fi

# Construct LD_LIBRARY_PATH including all nvidia subdirectories
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(find $NVIDIA_LIB_PATH -name lib -type d | tr '\n' ':'):$CUML_LIB_PATH

echo "Environment setup complete."
echo "Python: $PYTHON_EXEC"
echo "LD_LIBRARY_PATH set."

# Run the requested command
$PYTHON_EXEC "$@"
