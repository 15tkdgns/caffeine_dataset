#!/bin/bash
# Environment 3: LightGBM CUDA Installation
# LightGBM with CUDA support (compiled from source)

echo "======================================================================"
echo "Environment 3: LightGBM CUDA Setup"
echo "======================================================================"

# 환경 이름
ENV_NAME="lightgbm_cuda"

# 시스템 의존성 확인
echo -e "\n[1/6] Checking system dependencies"
echo "Required: cmake, libboost-dev, CUDA Toolkit"

# CMake 버전 확인
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo "✅ $CMAKE_VERSION"
else
    echo "❌ CMake not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y cmake
fi

# Boost 라이브러리 확인
if dpkg -l | grep libboost-dev &> /dev/null; then
    echo "✅ Boost libraries installed"
else
    echo "❌ Boost not found. Installing..."
    sudo apt-get install -y libboost-dev libboost-system-dev libboost-filesystem-dev
fi

# Conda 환경 생성
echo -e "\n[2/6] Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

# 환경 활성화
echo -e "\n[3/6] Activating environment"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 기본 패키지 설치
echo -e "\n[4/6] Installing base packages"
pip install numpy scipy scikit-learn pandas joblib imbalanced-learn

# LightGBM 소스 다운로드 및 컴파일
echo -e "\n[5/6] Building LightGBM from source with CUDA support"
echo "(This may take 10-15 minutes)"

# 임시 디렉토리로 이동
cd /tmp

# 기존 LightGBM 디렉토리 제거
rm -rf LightGBM

# Clone
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM

# Build
mkdir build
cd build
cmake -DUSE_CUDA=1 ..
make -j$(nproc)

# Python 패키지 설치
cd ../python-package
python setup.py install --cuda

# 검증
echo -e "\n[6/6] Verification"
echo "======================================================================"

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

# LightGBM
try:
    import lightgbm as lgb
    print(f"✅ LightGBM: {lgb.__version__}")
    
    # CUDA 지원 확인
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    dtrain = lgb.Dataset(X, label=y)
    params = {'device': 'cuda', 'gpu_platform_id': 0, 'gpu_device_id': 0}
    
    try:
        model = lgb.train(params, dtrain, num_boost_round=1)
        print(f"✅ LightGBM CUDA: Working")
    except Exception as e:
        print(f"⚠️ LightGBM CUDA test failed: {e}")
        print("   Will fall back to CPU if needed")
        
except Exception as e:
    print(f"❌ LightGBM: {e}")
EOF

# 정리
cd /root/ibm_data2
rm -rf /tmp/LightGBM

echo -e "\n======================================================================"
echo "Environment 3 Setup Complete!"
echo "======================================================================"
echo ""
echo "Activate with: conda activate $ENV_NAME"
echo "Models available: LightGBM (CUDA)"
echo ""
echo "Note: LightGBM was compiled from source with CUDA support."
