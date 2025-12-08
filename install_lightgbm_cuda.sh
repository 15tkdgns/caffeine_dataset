#!/bin/bash
# LightGBM GPU (CUDA) 안정적인 설치 v2
# 수동으로 빌드된 라이브러리 복사 방식

set -e

echo "======================================================================"
echo "LightGBM GPU (CUDA) 안정적인 설치 v2"
echo "======================================================================"

# 1. CUDA 환경 변수 설정
echo -e "\n[1/7] CUDA 환경 변수 설정"
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc --version

# 2. Python 가상환경 생성
echo -e "\n[2/7] Python 가상환경 생성"
VENV_DIR="/root/ibm_data2/venv_lightgbm"

if [ -d "$VENV_DIR" ]; then
    rm -rf $VENV_DIR
fi

python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install numpy scipy scikit-learn pandas joblib

# 3. LightGBM 소스 다운로드
echo -e "\n[3/7] LightGBM 소스 다운로드"
cd /tmp
rm -rf LightGBM

git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM

# 4. CUDA로 빌드
echo -e "\n[4/7] LightGBM CUDA 빌드"
mkdir -p build
cd build

cmake -DUSE_CUDA=1 \
      -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="89" \
      ..

make -j$(nproc)

if [ ! -f "../lib_lightgbm.so" ]; then
    echo "❌ lib_lightgbm.so 빌드 실패"
    exit 1
fi

echo "✅ lib_lightgbm.so 빌드 성공"

# 5. 일반 pip로 lightgbm 설치 (pre-built wheel)
echo -e "\n[5/7] LightGBM pip 설치"
pip install lightgbm

# 6. 빌드된 CUDA 라이브러리로 교체
echo -e "\n[6/7] CUDA 라이브러리 교체"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
LGB_LIB_DIR="$SITE_PACKAGES/lightgbm/lib"

echo "LightGBM 라이브러리 경로: $LGB_LIB_DIR"

# 기존 라이브러리 백업
if [ -f "$LGB_LIB_DIR/lib_lightgbm.so" ]; then
    mv "$LGB_LIB_DIR/lib_lightgbm.so" "$LGB_LIB_DIR/lib_lightgbm.so.backup"
fi

# CUDA 빌드 라이브러리 복사
cp /tmp/LightGBM/lib_lightgbm.so "$LGB_LIB_DIR/"

echo "✅ CUDA 라이브러리 교체 완료"

# 7. 검증
echo -e "\n[7/7] 검증"
echo "======================================================================"

cd /root/ibm_data2

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import lightgbm as lgb
    print(f"✅ LightGBM: {lgb.__version__}")
    
    import numpy as np
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    dtrain = lgb.Dataset(X, label=y)
    
    # CUDA 테스트
    params = {
        'device': 'cuda',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'objective': 'binary',
        'num_iterations': 10,
        'verbose': -1
    }
    
    try:
        model = lgb.train(params, dtrain)
        print(f"✅ LightGBM CUDA 테스트: 성공!")
    except Exception as e:
        print(f"⚠️ LightGBM CUDA: {e}")
        print("   GPU (OpenCL) 모드로 재시도...")
        
        params['device'] = 'gpu'
        try:
            model = lgb.train(params, dtrain)
            print(f"✅ LightGBM GPU (OpenCL) 테스트: 성공!")
        except Exception as e2:
            print(f"❌ GPU 모드 실패: {e2}")
            print("   CPU 모드로 사용 가능")
    
except Exception as e:
    print(f"❌ LightGBM 설치 실패: {e}")
EOF

echo -e "\n======================================================================"
echo "LightGBM GPU 설치 완료!"
echo "======================================================================"
echo ""
echo "사용 방법:"
echo "  source /root/ibm_data2/venv_lightgbm/bin/activate"
