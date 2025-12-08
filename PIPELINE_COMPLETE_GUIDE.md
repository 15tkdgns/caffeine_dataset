# 🚀 머신러닝 파이프라인 완전 재현 가이드

전처리 → 피처 엔지니어링 → 피처 셀렉션 → 사용자 필터링 → SMOTE 증강 → GPU 모델 비교 → Top 3 분석

---

## 📋 목차

1. [환경 구성](#1-환경-구성)
2. [GPU 환경 확인](#2-gpu-환경-확인)
3. [데이터 전처리](#3-데이터-전처리)
4. [피처 엔지니어링](#4-피처-엔지니어링)
5. [사용자 필터링](#5-사용자-필터링)
6. [SMOTE 데이터 증강](#6-smote-데이터-증강)
7. [GPU 모델 비교](#7-gpu-모델-비교)
8. [Top 3 모델 분석](#8-top-3-모델-분석)
9. [전체 파이프라인 실행](#9-전체-파이프라인-실행)

---

## 1. 환경 구성

### 필수 패키지

```bash
# GPU 환경
pip install tensorflow[and-cuda]==2.20.0
pip install xgboost==3.1.2
pip install catboost
pip install lightgbm

# 기본 패키지
pip install pandas scikit-learn joblib
pip install imbalanced-learn  # SMOTE용
```

### GPU 요구사항

- **GPU**: NVIDIA GPU (CUDA 지원)
- **CUDA**: 12.6+
- **메모리**: 16GB+ RAM
- **Python**: 3.11+

---

## 2. GPU 환경 확인

### GPU 사용 가능 여부 체크

```bash
python3 01_src/utils/gpu_check.py
```

**예상 출력:**
```
✅ TensorFlow GPU: 1개 감지
✅ XGBoost GPU: 사용 가능
✅ CatBoost GPU: 사용 가능
```

---

## 3. 데이터 전처리

### 기존 전처리 상태

**파일:** `02_data/01_processed/preprocessed_enhanced.csv`
- 샘플 수: 18,142,056개
- 피처 수: 27개 (스케일링 완료)
- 타겟: Next_Category_encoded (6개 클래스)

---

## 4. 피처 엔지니어링

### 생성된 27개 피처

⭐ **핵심 피처**: User_*_Ratio (카테고리별 이용 비율 6개)

### 피처 셀렉션 결과

**27개 → 16개** 선택 (메모리 -40.7% 절감)

---

## 5. 사용자 필터링

**조건:** 월 10건 이상 거래 & 5개월 이상 활동

```bash
python3 01_src/00_preprocessing/08_filter_active_monthly.py
```

---

## 6. SMOTE 데이터 증강

```bash
python3 01_src/00_preprocessing/09_apply_smote.py
```

**효과:** 클래스 불균형 해결, Macro F1 +3~5%p

---

## 7. GPU 모델 비교

### 지원 모델 (6개)

1. **XGBoost (GPU)**
2. **TensorFlow NN (GPU)**  
3. **CatBoost (GPU)**
4. **LightGBM (GPU)**
5. ExtraTrees (CPU)
6. RandomForest (CPU)

```bash
python3 01_src/01_training/10_compare_gpu_models.py
```

---

## 8. Top 3 모델 분석

```bash
python3 01_src/01_training/11_analyze_top3.py
```

**출력:** `05_docs/TOP3_MODELS_ANALYSIS.md`

---

## 9. 전체 파이프라인 실행

### 원클릭 실행

```bash
bash run_full pipeline_gpu.sh
```

### 수동 실행 (단계별)

```bash
# 1. GPU 체크
python3 01_src/utils/gpu_check.py

# 2. 사용자 필터링
python3 01_src/00_preprocessing/08_filter_active_monthly.py

# 3. SMOTE 증강
python3 01_src/00_preprocessing/09_apply_smote.py

# 4. 모델 비교
python3 01_src/01_training/10_compare_gpu_models.py

# 5. Top 3 분석
python3 01_src/01_training/11_analyze_top3.py
```

---

## 📊 예상 결과

### Top 3 모델 (예상)

1. **XGBoost (GPU)**: ~50% Accuracy
2. **CatBoost (GPU)**: ~48% Accuracy  
3. **TensorFlow NN**: ~46% Accuracy

---

## 🔑 핵심 인사이트

- **최강 피처**: User_*_Ratio (카테고리별 이용 비율)
- **GPU 효과**: 학습 속도 10-100배 향상
- **SMOTE 효과**: 소수 클래스 성능 +10%p

---

**Made with 🚀 GPU acceleration**
