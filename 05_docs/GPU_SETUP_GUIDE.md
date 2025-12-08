# GPU 환경 구축 가이드

## 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [환경 구성](#환경-구성)
3. [모델별 GPU 지원 현황](#모델별-gpu-지원-현황)
4. [문제 해결 과정](#문제-해결-과정)
5. [최종 작동 방법](#최종-작동-방법)

---

## 시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA GeForce RTX 4070 Ti
- **GPU 메모리**: 12GB
- **CUDA Version**: 12.6
- **Driver Version**: 560.94

### 소프트웨어
- **OS**: Linux (WSL2)
- **Conda**: Miniconda3
- **Python**: 3.11.14 (gemini_gpu 환경)

---

## 환경 구성

### 1. 기본 Python 환경 (실패)
**초기 시도**: 기본 Python 3.13 환경에서 pip로 설치

**문제점**:
- `cuml-cu12==25.10.0`: ExtraTreesClassifier 미지원 (RandomForest만 제공)
- `tensorflow`: GPU 라이브러리 로딩 실패 (`Cannot dlopen some GPU libraries`)
- `lightgbm`: GPU 컴파일 없이 설치되어 GPU 미지원
- `xgboost`: 버전 충돌 (2.1.1 vs 3.1.1)

**의존성 충돌**:
```
numpy 2.3.5 설치 -> numba 호환성 오류 (numba requires numpy < 2.3)
cuml 설치 -> libstdc++ ABI 버전 불일치
tensorflow 설치 -> nvidia 라이브러리 경로 문제
```

### 2. 전용 Conda 환경 구축 (성공)
**환경 이름**: `gemini_gpu` (Python 3.11.14)

**설치 패키지**:
```bash
# 기본 환경 생성
conda create -n gemini_gpu python=3.11

# 의존성 설치
pip install xgboost==3.1.2
pip install lightgbm==4.6.0
pip install cuml-cu12==24.04.0 --extra-index-url=https://pypi.nvidia.com
pip install tensorflow[and-cuda] pandas scikit-learn
```

**주요 의존성 버전**:
- `cuml-cu12`: 24.04.0 (24.10.0/25.10.0은 ExtraTrees 미지원)
- `xgboost`: 3.1.2
- `tensorflow`: 2.20.0
- `numpy`: 2.2.6 (numba 호환성 고려)
- `protobuf`: 6.33.1 (tensorflow 요구)

### 3. 라이브러리 경로 설정

**필수 환경 변수**:
```bash
# libstdc++ ABI 버전 문제 해결
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# NVIDIA 라이브러리 경로
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
$(find /root/miniconda3/envs/gemini_gpu/lib/python3.11/site-packages/nvidia -name lib -type d | tr '\n' ':'):\
/root/miniconda3/envs/gemini_gpu/lib/python3.11/site-packages/libcuml/lib64
```

**자동화 스크립트**: `run_gpu.sh` 생성
```bash
#!/bin/bash
PYTHON_EXEC="/root/miniconda3/envs/gemini_gpu/bin/python"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# ... (라이브러리 경로 자동 설정)
$PYTHON_EXEC "$@"
```

---

## 모델별 GPU 지원 현황

### 1. Neural Network (TensorFlow)
**상태**: ✅ 성공

**GPU 설정 방법**:
```python
import tensorflow as tf

# GPU 메모리 동적 할당
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 모델 학습
with tf.device('/GPU:0'):
    model.fit(X_train, y_train, ...)
```

**주요 문제 및 해결**:
- **문제**: `Cannot dlopen some GPU libraries`
- **원인**: LD_LIBRARY_PATH에 nvidia 라이브러리 경로 누락
- **해결**: `run_gpu.sh`로 경로 자동 설정

**검증 결과**:
```
성공: 1개의 GPU를 찾았습니다:
  - GPU 0: /physical_device:GPU:0
학습 디바이스: /GPU:0
최종 테스트 정확도: 0.4762
```

### 2. XGBoost
**상태**: ✅ 성공

**GPU 설정 방법**:
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    device="cuda",          # GPU 사용
    tree_method="hist",     # XGBoost 3.x 호환
    n_estimators=500,
    random_state=42
)
```

**주요 문제 및 해결**:
- **문제 1**: 버전 충돌 (2.1.1 vs 3.1.1)
- **해결**: 완전 삭제 후 재설치 (`pip uninstall -y xgboost && pip install xgboost`)

- **문제 2**: `'gpu_hist' is not valid` (XGBoost 3.x)
- **해결**: `tree_method='hist', device='cuda'` 파라미터 변경

**검증 결과**:
```
XGBoost를 GPU로 실행합니다.
정확도: 0.4018
```

### 3. ExtraTrees (cuML → XGBoost RF Fallback)
**상태**: ✅ 성공 (대체 구현)

**시도한 방법들**:

#### 시도 1: cuML ExtraTreesClassifier (실패)
```python
from cuml.ensemble import ExtraTreesClassifier
```
- **문제**: 모든 버전(23.10.0, 24.04.0, 24.10.0, 25.10.0)에서 ExtraTreesClassifier 미제공
- **확인**: `dir(cuml.ensemble)` → RandomForest만 존재

#### 시도 2: 구버전 cuML 설치 시도 (실패)
```bash
pip install cuml-cu12==23.12.0
pip install cuml-cu12==23.10.0
```
- **문제**: Python 3.11과 호환되는 빌드 파일 없음
- **오류**: `RuntimeError: Bad params`, `Didn't find wheel for cuml-cu12`

#### 시도 3: XGBoost Random Forest로 대체 (성공)
```python
# ExtraTrees 대신 XGBoost Random Forest 사용
try:
    from cuml.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier(...)
except ImportError:
    print("대안으로 XGBoost Random Forest (GPU)를 사용하여 ExtraTrees를 대체합니다.")
    model = xgb.XGBRFClassifier(
        device="cuda",
        tree_method="hist",
        n_estimators=500,
        subsample=0.8,
        colsample_bynode=0.8,
        random_state=42
    )
```

**검증 결과**:
```
경고: cuML에서 ExtraTreesClassifier를 찾을 수 없습니다.
대안으로 XGBoost Random Forest (GPU)를 사용하여 ExtraTrees를 대체합니다.
XGBoost Random Forest (GPU)를 ExtraTrees 대용으로 실행합니다.
정확도: 0.3975
```

### 4. LightGBM
**상태**: ❌ 실패

**시도한 방법**:
```python
model = lgb.LGBMClassifier(
    device='gpu',
    n_estimators=500,
    random_state=42
)
```

**문제**:
```
[LightGBM] [Fatal] GPU Tree Learner was not enabled in this build.
Please recompile with CMake option -DUSE_GPU=1
```

**원인**: pip로 설치한 LightGBM은 GPU 지원 없이 빌드됨

**해결 방법 (미구현)**:
```bash
# 소스에서 GPU 지원 빌드 필요
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake -DUSE_GPU=1 ..
make -j4
cd ../python-package
python setup.py install
```

---

## 문제 해결 과정

### 1. cuML 라이브러리 문제

#### 문제: `libcuml++.so` 로딩 실패
```
ImportError: libcuml++.so: cannot open shared object file
```

**해결 과정**:
1. 파일 위치 확인: `find /root/miniconda3 -name "libcuml++.so"`
2. LD_LIBRARY_PATH에 추가: `/root/miniconda3/lib/python3.13/site-packages/libcuml/lib64`

#### 문제: `undefined symbol: __nvJitLinkGetErrorLogSize_12_9`
```
ImportError: /root/miniconda3/lib/python3.13/site-packages/nvidia/cusparse/lib/libcusparse.so.12: 
undefined symbol: __nvJitLinkGetErrorLogSize_12_9, version libnvJitLink.so.12
```

**해결 과정**:
1. nvidia-nvjitlink-cu12 버전 확인 및 업그레이드
2. cuml-cu12 버전 다운그레이드 (25.10.0 → 24.04.0)

#### 문제: `ExtraTreesClassifier` 미제공
```
ImportError: cannot import name 'ExtraTreesClassifier' from 'cuml.ensemble'
```

**조사 결과**:
```bash
# 사용 가능한 클래스 확인
python -c "import cuml.ensemble; print(dir(cuml.ensemble))"
# 출력: ['RandomForestClassifier', 'RandomForestRegressor', ...]
```

**결론**: cuML은 ExtraTreesClassifier를 제공하지 않음 (모든 버전)

### 2. TensorFlow GPU 라이브러리 문제

#### 문제: GPU 라이브러리 로딩 실패
```
WARNING: Cannot dlopen some GPU libraries.
```

**해결 과정**:
1. **라이브러리 경로 확인**:
```bash
find /root/miniconda3 -name "libcudart.so*"
find /root/miniconda3 -name "libcudnn.so*"
```

2. **LD_LIBRARY_PATH 설정**:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
/root/miniconda3/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
/root/miniconda3/lib/python3.13/site-packages/nvidia/cudnn/lib:\
/root/miniconda3/lib/python3.13/site-packages/nvidia/cublas/lib
```

3. **자동화 스크립트 작성**: `run_gpu.sh`

### 3. 의존성 충돌 문제

#### 문제: NumPy 버전 충돌
```
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.
```

**해결**:
```bash
pip install "numpy<2.3"  # 2.2.6 설치
```

#### 문제: libstdc++ ABI 버전 불일치
```
ImportError: /root/miniconda3/lib/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

**해결**:
```bash
# 시스템 libstdc++를 우선 로드
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

#### 문제: protobuf 버전 충돌
```python
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

**해결**:
```bash
pip install --upgrade protobuf  # 6.33.1로 업그레이드
# 주의: cudf-cu12는 protobuf<5를 요구하지만, tensorflow는 최신 버전 필요
```

### 4. XGBoost 버전 문제

#### 문제: 파라미터 호환성
```
XGBoostError: Invalid Input: 'gpu_hist', valid values are: {'approx', 'auto', 'exact', 'hist'}
```

**원인**: XGBoost 3.x에서 API 변경

**해결**:
```python
# XGBoost 2.x (구버전)
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')

# XGBoost 3.x (신버전)
model = XGBClassifier(device='cuda', tree_method='hist')
```

---

## 최종 작동 방법

### 실행 스크립트: `run_gpu.sh`
```bash
#!/bin/bash

# Define the python path for gemini_gpu environment
PYTHON_EXEC="/root/miniconda3/envs/gemini_gpu/bin/python"

# Set up environment variables
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Add nvidia libraries and cuml libraries to LD_LIBRARY_PATH
NVIDIA_LIB_PATH=$($PYTHON_EXEC -c "import os; import nvidia; print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
CUML_LIB_PATH=$($PYTHON_EXEC -c "import os; import cuml; print(os.path.join(os.path.dirname(cuml.__file__), '..', 'libcuml', 'lib64'))" 2>/dev/null)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(find $NVIDIA_LIB_PATH -name lib -type d | tr '\n' ':'):$CUML_LIB_PATH

$PYTHON_EXEC "$@"
```

### 사용 방법
```bash
# 실행 권한 부여
chmod +x run_gpu.sh

# Neural Network 학습
./run_gpu.sh 01_src/01_training/00_train_nn.py --device gpu

# XGBoost 학습
./run_gpu.sh 01_src/01_training/01_train_tree.py --model xgboost --device gpu

# ExtraTrees (XGBRF fallback) 학습
./run_gpu.sh 01_src/01_training/01_train_tree.py --model extratrees --device gpu
```

### 코드 수정 사항

#### `00_train_nn.py`
```python
# --device 인자 추가
parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto')

# GPU 강제 모드
def check_gpu(device_arg):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus and device_arg == 'gpu':
        raise RuntimeError("GPU를 찾을 수 없습니다. --device=gpu가 지정되어 종료합니다.")
```

#### `01_train_tree.py`
```python
# XGBoost 파라미터 업데이트 (v3.x 호환)
model = xgb.XGBClassifier(
    device="cuda",         # gpu_hist 대신 device 사용
    tree_method="hist",    # 기본 hist
    # ...
)

# ExtraTrees fallback 구현
elif model_name == "extratrees":
    if use_gpu:
        try:
            from cuml.ensemble import ExtraTreesClassifier as CumlExtraTreesClassifier
            model = CumlExtraTreesClassifier(...)
        except ImportError:
            # XGBoost Random Forest로 대체
            model = xgb.XGBRFClassifier(device="cuda", ...)
```

---

## 요약

### 성공한 구성
| 구성 요소 | 버전/설정 | 비고 |
|----------|----------|------|
| Python 환경 | 3.11.14 (gemini_gpu) | Conda 환경 |
| TensorFlow | 2.20.0 | GPU 지원 |
| XGBoost | 3.1.2 | device='cuda' |
| cuML | 24.04.0 | RandomForest 제공 |
| ExtraTrees | XGBRFClassifier | cuML 대체 |
| LightGBM | 실패 | 소스 컴파일 필요 |

### 핵심 해결책
1. **전용 Conda 환경 사용** (Python 3.11)
2. **LD_LIBRARY_PATH 자동 설정** (`run_gpu.sh`)
3. **LD_PRELOAD로 libstdc++ ABI 해결**
4. **cuML 버전 다운그레이드** (25.10.0 → 24.04.0)
5. **ExtraTrees → XGBoost RF Fallback**

### 여전히 남은 문제
- LightGBM GPU 지원 (소스 컴파일 필요)
- cuML ExtraTreesClassifier 미제공 (공식적으로 지원하지 않음)
