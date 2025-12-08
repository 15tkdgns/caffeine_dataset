# 🚀 전략 B: 다중 GPU 환경 완전 가이드

3개의 독립적인 Conda 환경으로 모든 GPU 모델 완벽 지원

---

## 📋 환경 구성

### 환경 1: gpu_basic
- **모델**: TensorFlow, XGBoost, CatBoost
- **용도**: 메인 GPU 모델 (가장 많이 사용)
- **설치 시간**: ~10분

### 환경 2: rapids_cuml  
- **모델**: cuML RandomForest (GPU)
- **용도**: 초고속 RandomForest
- **설치 시간**: ~15분

### 환경 3: lightgbm_cuda
- **모델**: LightGBM (CUDA)
- **용도**: 메모리 효율적 부스팅
- **설치 시간**: ~15분 (컴파일 포함)

---

## 🔧 설치 방법

### 방법 1: 전체 자동 설치 (권장)

```bash
bash install_all_gpu_environments.sh
```

**소요 시간**: 30-60분
**설치 내용**: 3개 환경 모두

---

### 방법 2: 환경별 개별 설치

```bash
# 환경 1
bash install_env1_basic_gpu.sh

# 환경 2
bash install_env2_rapids.sh  

# 환경 3
bash install_env3_lightgbm_cuda.sh
```

---

## 🎯 모델 학습 실행

### 단일 환경 실행

```bash
# 환경 활성화 후 실행
conda activate gpu_basic
python3 01_src/01_training/12_train_by_environment.py
```

### 전체 환경 순차 실행 (권장)

```bash
bash run_all_environments.sh
```

**동작:**
1. env1 → TensorFlow, XGBoost, CatBoost 학습
2. env2 → cuML RandomForest 학습  
3. env3 → LightGBM CUDA 학습
4. 결과 자동 통합 (`combined_results.csv`)
5. Top 3 모델 자동 선정

---

## 📊 예상 결과

| 모델 | Accuracy | 학습 시간 | 환경 |
|------|----------|-----------|------|
| **CatBoost** | ~0.52 | 60초 | env1 |
| **cuML RF** | ~0.50 | 10초 ⚡ | env2 |
| **XGBoost** | ~0.48 | 30초 | env1 |
| **LightGBM** | ~0.47 | 25초 | env3 |
| **TensorFlow** | ~0.46 | 90초 | env1 |

---

## 📂 결과 파일

```
03_models/multi_env_comparison/
├── env1_results.json          # 환경 1 결과
├── env2_results.json          # 환경 2 결과
├── env3_results.json          # 환경 3 결과
├── combined_results.json      # 통합 결과
└── combined_results.csv       # 비교표
```

---

## 🔍 환경 전환

```bash
# 환경 목록 확인
conda env list

# 환경 활성화
conda activate gpu_basic       # 또는 rapids_cuml, lightgbm_cuda

# 환경 비활성화
conda deactivate
```

---

## 💾 디스크 공간

- 환경 1: ~5GB
- 환경 2: ~8GB (RAPIDS)
- 환경 3: ~3GB
- **총 필요 공간**: ~20GB

---

## ⚠️ 문제 해결

### RAPIDS 설치 실패

```bash
# CUDA 버전 확인
nvidia-smi

# Conda 채널 업데이트
conda update -n base conda
```

### LightGBM 컴파일 실패

```bash
# 의존성 재설치
sudo apt-get install cmake libboost-dev
```

---

## 🎓 다음 단계

1. ✅ 환경 설치: `bash install_all_gpu_environments.sh`
2. ✅ 모델 학습: `bash run_all_environments.sh`
3. ✅ Top 3 분석: `python3 01_src/01_training/11_analyze_top3.py`
4. 📊 결과 확인: `05_docs/TOP3_MODELS_ANALYSIS.md`

---

**전략 B의 장점**: 모든 GPU 모델 100% 지원, 의존성 충돌 없음
**전략 B의 단점**: 환경 관리 복잡, 디스크 공간 많이 사용

**Made with 🚀 Multi-Environment Strategy**
