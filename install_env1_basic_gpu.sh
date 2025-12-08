#!/bin/bash
# Environment 1: Basic GPU Models Installation
# TensorFlow, XGBoost, CatBoost

echo "======================================================================"
echo "Environment 1: Basic GPU Models Setup"
echo "======================================================================"

# 환경 이름
ENV_NAME="gpu_basic"

# Conda 환경 생성
echo -e "\n[1/3] Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

# 환경 활성화
echo -e "\n[2/3] Activating environment"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 패키지 설치
echo -e "\n[3/3] Installing packages"
pip install -r requirements_env1_basic_gpu.txt

# 검증
echo -e "\n======================================================================"
echo "Verification"
echo "======================================================================"

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

# TensorFlow GPU
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✅ TensorFlow: {tf.__version__}, GPU: {len(gpus)}")
except Exception as e:
    print(f"❌ TensorFlow: {e}")

# XGBoost
try:
    import xgboost as xgb
    print(f"✅ XGBoost: {xgb.__version__}")
except Exception as e:
    print(f"❌ XGBoost: {e}")

# CatBoost
try:
    import catboost as cb
    print(f"✅ CatBoost: {cb.__version__}")
except Exception as e:
    print(f"❌ CatBoost: {e}")
EOF

echo -e "\n======================================================================"
echo "Environment 1 Setup Complete!"
echo "======================================================================"
echo ""
echo "Activate with: conda activate $ENV_NAME"
echo "Models available: TensorFlow, XGBoost, CatBoost"
