#!/bin/bash
# cuML GPU 설치 (pip 방식)
# RAPIDS cuML을 pip로 설치 (Conda 없이)

echo "======================================================================"
echo "cuML GPU 설치 (pip 방식)"
echo "======================================================================"

# 1. CUDA 환경 변수 설정
echo -e "\n[1/4] CUDA 환경 변수 설정"
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 확인
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version 2>/dev/null || echo "nvcc 설치 확인 중..."

# 2. Python 가상환경 생성
echo -e "\n[2/4] Python 가상환경 생성"
VENV_DIR="/root/ibm_data2/venv_cuml"

if [ -d "$VENV_DIR" ]; then
    echo "기존 가상환경 삭제..."
    rm -rf $VENV_DIR
fi

python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 3. cuML 설치 (NVIDIA PyPI)
echo -e "\n[3/4] cuML 설치 (CUDA 12)"
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cuml-cu12 \
    cudf-cu12 \
    pylibraft-cu12 \
    rmm-cu12

# 4. 추가 패키지
echo -e "\n[4/4] 추가 패키지 설치"
pip install pandas scikit-learn joblib numpy

# 검증
echo -e "\n======================================================================"
echo "검증"
echo "======================================================================"

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import cuml
    print(f"✅ cuML: {cuml.__version__}")
    
    from cuml.ensemble import RandomForestClassifier
    print(f"✅ cuML RandomForestClassifier: 사용 가능")
    
    import cupy as cp
    print(f"✅ CuPy: {cp.__version__}")
    
    # GPU 테스트
    import numpy as np
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int32)
    
    X_gpu = cp.array(X)
    y_gpu = cp.array(y)
    
    model = RandomForestClassifier(n_estimators=10, max_depth=5)
    model.fit(X_gpu, y_gpu)
    print(f"✅ cuML RandomForest GPU 테스트: 성공!")
    
except Exception as e:
    print(f"❌ cuML 설치 실패: {e}")
EOF

echo -e "\n======================================================================"
echo "cuML 설치 완료!"
echo "======================================================================"
echo ""
echo "사용 방법:"
echo "  source /root/ibm_data2/venv_cuml/bin/activate"
echo "  python3 01_src/01_training/12_train_by_environment.py"
