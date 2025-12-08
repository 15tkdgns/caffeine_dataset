#!/bin/bash
# Environment 2: RAPIDS cuML Installation
# cuML RandomForest, cuDF

echo "======================================================================"
echo "Environment 2: RAPIDS cuML Setup"
echo "======================================================================"

# 환경 이름
ENV_NAME="rapids_cuml"

# Conda 환경 생성 (RAPIDS는 Conda 필수)
echo -e "\n[1/4] Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

# 환경 활성화
echo -e "\n[2/4] Activating environment"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# RAPIDS 설치
echo -e "\n[3/4] Installing RAPIDS cuML (this may take several minutes)"
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.12 \
    cudf=24.12 \
    python=3.12 \
    cuda-version=12.6 \
    -y

# 추가 패키지
echo -e "\n[4/4] Installing additional packages"
pip install pandas scikit-learn joblib imbalanced-learn

# 검증
echo -e "\n======================================================================"
echo "Verification"
echo "======================================================================"

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

# cuML
try:
    import cuml
    print(f"✅ cuML: {cuml.__version__}")
    from cuml.ensemble import RandomForestClassifier
    print(f"✅ cuML RandomForestClassifier: Available")
except Exception as e:
    print(f"❌ cuML: {e}")

# cuDF
try:
    import cudf
    print(f"✅ cuDF: {cudf.__version__}")
except Exception as e:
    print(f"❌ cuDF: {e}")

# CuPy
try:
    import cupy as cp
    print(f"✅ CuPy: {cp.__version__}")
except Exception as e:
    print(f"❌ CuPy: {e}")
EOF

echo -e "\n======================================================================"
echo "Environment 2 Setup Complete!"
echo "======================================================================"
echo ""
echo "Activate with: conda activate $ENV_NAME"
echo "Models available: cuML RandomForest (GPU)"
echo ""
echo "Note: This environment is optimized for RAPIDS cuML only."
