# GPU 모델 학습 - 핵심 코드 가이드

## 🚀 빠른 시작

### 실행 명령어
```bash
# Neural Network
./run_gpu.sh 01_src/01_training/00_train_nn.py --device gpu

# Tree Models
./run_gpu.sh 01_src/01_training/01_train_tree.py --model xgboost --device gpu
./run_gpu.sh 01_src/01_training/01_train_tree.py --model randomforest --device gpu
./run_gpu.sh 01_src/01_training/01_train_tree.py --model extratrees --device gpu
```

---

## 📝 핵심 코드

### 1. Neural Network (TensorFlow)

```python
import tensorflow as tf

# GPU 설정
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    return False

# 모델 학습
use_gpu = check_gpu()
device = '/GPU:0' if use_gpu else '/CPU:0'

with tf.device(device):
    model.fit(X_train, y_train, 
             epochs=10, 
             batch_size=1024,
             validation_data=(X_test, y_test))
```

### 2. XGBoost

```python
import xgboost as xgb

# GPU 모델 생성
model = xgb.XGBClassifier(
    device="cuda",           # GPU 사용
    tree_method="hist",      # XGBoost 3.x
    n_estimators=500,
    random_state=42
)

# 학습
model.fit(X_train, y_train)
```

### 3. RandomForest (cuML)

```python
from cuml.ensemble import RandomForestClassifier

# GPU 모델 생성
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=16,
    n_streams=1,         # GPU 스트림 수
    random_state=42
)

# 학습
model.fit(X_train, y_train)
```

### 4. ExtraTrees (XGBoost RF Fallback)

```python
import xgboost as xgb

# cuML ExtraTrees가 없으므로 XGBoost RF 사용
model = xgb.XGBRFClassifier(
    device="cuda",
    tree_method="hist",
    n_estimators=500,
    subsample=0.8,
    colsample_bynode=0.8,
    random_state=42
)

# 학습
model.fit(X_train, y_train)
```

---

## 🔧 환경 설정 (run_gpu.sh)

```bash
#!/bin/bash

PYTHON_EXEC="/root/miniconda3/envs/gemini_gpu/bin/python"

# libstdc++ ABI 호환성
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# NVIDIA 라이브러리 경로
NVIDIA_LIB_PATH=$($PYTHON_EXEC -c "import os; import nvidia; print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
CUML_LIB_PATH=$($PYTHON_EXEC -c "import os; import cuml; print(os.path.join(os.path.dirname(cuml.__file__), '..', 'libcuml', 'lib64'))" 2>/dev/null)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(find $NVIDIA_LIB_PATH -name lib -type d | tr '\n' ':'):$CUML_LIB_PATH

$PYTHON_EXEC "$@"
```

---

## 📊 모델별 GPU 파라미터

### TensorFlow
| 파라미터 | 설정 | 설명 |
|---------|------|------|
| device | `/GPU:0` | GPU 디바이스 지정 |
| memory_growth | `True` | 메모리 동적 할당 |

### XGBoost
| 파라미터 | 설정 | 설명 |
|---------|------|------|
| device | `"cuda"` | GPU 사용 (v3.x) |
| tree_method | `"hist"` | 히스토그램 기반 알고리즘 |

### cuML RandomForest
| 파라미터 | 설정 | 설명 |
|---------|------|------|
| n_streams | `1` | GPU 병렬 스트림 수 |
| max_depth | `16` | 트리 최대 깊이 |

---

## ⚡ 성능 비교 (500k 샘플)

| 모델 | GPU 시간 | CPU 시간 | 배속 |
|------|---------|---------|-----|
| Neural Network | 90초 | ~300초 | 3.3x |
| XGBoost | 8초 | ~40초 | 5x |
| RandomForest | **3.7초** | ~60초 | 16x |
| ExtraTrees | 8초 | ~50초 | 6x |

---

## 🐛 트러블슈팅

### GPU 인식 실패
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# [] → GPU 미인식

# 해결: run_gpu.sh 사용
```

### cuML ImportError
```bash
# 오류: libcuml++.so not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libcuml/lib64
```

### XGBoost 파라미터 오류
```python
# ❌ XGBoost 2.x (구버전)
model = XGBClassifier(tree_method='gpu_hist')

# ✅ XGBoost 3.x (신버전)
model = XGBClassifier(device='cuda', tree_method='hist')
```

---

## 📦 필수 패키지

```bash
pip install tensorflow[and-cuda]==2.20.0
pip install xgboost==3.1.2
pip install cuml-cu12==24.4.0 --extra-index-url=https://pypi.nvidia.com
pip install pandas scikit-learn
```

---

## ✅ GPU 검증

### TensorFlow
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"사용 가능한 GPU: {len(gpus)}개")
```

### XGBoost
```python
import xgboost as xgb
model = xgb.XGBClassifier(device='cuda')
# 오류 없이 생성되면 성공
```

### cuML
```python
from cuml.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# 오류 없이 생성되면 성공
```

---

## 📌 핵심 요약

1. **환경 변수**: `run_gpu.sh` 스크립트 필수 사용
2. **TensorFlow**: `with tf.device('/GPU:0')`
3. **XGBoost**: `device='cuda', tree_method='hist'`
4. **cuML**: `RandomForestClassifier` 직접 사용
5. **검증**: 각 라이브러리별 GPU 인식 확인

---

## 🔗 관련 문서

- 상세 가이드: `05_docs/GPU_SETUP_GUIDE.md`
- 최종 보고서: `05_docs/GPU_FINAL_REPORT.md`
- Requirements: `requirements_gpu.txt`
