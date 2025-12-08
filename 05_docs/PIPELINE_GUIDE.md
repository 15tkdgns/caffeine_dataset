# 전체 데이터 GPU 학습 파이프라인 가이드

## 📋 개요

전체 데이터셋(2,400만 건)을 사용하여 피처 엔지니어링부터 GPU 그리드 서치까지 수행하는 완전한 머신러닝 파이프라인입니다.

---

## 🚀 빠른 시작

### 전체 파이프라인 실행 (원클릭)
```bash
# 전체 데이터 사용
./run_full_pipeline.sh

# 데이터 일부만 사용 (테스트용, 10%)
./run_full_pipeline.sh 0.1
```

### 개별 단계 실행

#### 1단계: 전처리
```bash
python3 01_src/00_preprocessing/01_preprocess_full.py
```

#### 2단계: 그리드 서치
```bash
./run_gpu.sh 01_src/01_training/02_gridsearch_gpu.py
```

---

## 📁 파일 구조

```
ibm_data2/
├── run_full_pipeline.sh              # 마스터 파이프라인 스크립트
├── run_gpu.sh                         # GPU 환경 설정 스크립트
│
├── 01_src/
│   ├── 00_preprocessing/
│   │   └── 01_preprocess_full.py     # 전체 데이터 전처리 + 피처 엔지니어링
│   └── 01_training/
│       └── 02_gridsearch_gpu.py      # GPU 그리드 서치
│
├── 02_data/
│   ├── 00_raw/
│   │   └── credit_card_transactions-ibm_v2.csv  # 원본 (24M 건)
│   └── 01_processed/
│       ├── preprocessed_full_featured.csv        # 전처리 완료 데이터
│       └── preprocessing_metadata.txt            # 전처리 메타정보
│
└── 03_models/
    └── 05_gridsearch/
        ├── best_xgboost_TIMESTAMP.joblib         # 최적 모델
        ├── metadata_xgboost_TIMESTAMP.json       # 하이퍼파라미터
        └── cv_results_xgboost_TIMESTAMP.csv      # CV 상세 결과
```

---

## 🔧 파이프라인 상세

### 1단계: 전처리 및 피처 엔지니어링

**실행**: `python3 01_src/00_preprocessing/01_preprocess_full.py`

#### 생성되는 피처 (총 20개)

**시간 관련 (6개)**:
- `Hour`: 시간 (0-23)
- `DayOfWeek`: 요일 (0=월, 6=일)
- `DayOfMonth`: 날짜 (1-31)
- `IsWeekend`: 주말 여부 (0/1)
- `IsNight`: 야간 거래 (22시-6시)
- `IsBusinessHour`: 업무 시간 (9시-18시)

**금액 관련 (3개)**:
- `Amount`: 원본 금액
- `Amount_log`: 로그 변환 금액
- `Amount_bin`: 금액 구간 (0-4)

**사용자 통계 (5개)**:
- `User_TotalTransactions`: 사용자 총 거래 수
- `User_AvgAmount`: 사용자 평균 금액
- `User_StdAmount`: 사용자 금액 표준편차
- `User_MaxAmount`: 사용자 최대 금액
- `User_MinAmount`: 사용자 최소 금액

**카드 통계 (2개)**:
- `Card_TotalTransactions`: 카드 총 거래 수
- `Card_AvgAmount`: 카드 평균 금액

**MCC 통계 (2개)**:
- `MCC_AvgAmount`: MCC 평균 금액
- `MCC_TotalCount`: MCC 거래 수

**상대적 특성 (2개)**:
- `Amount_vs_UserAvg`: 현재 금액 / 사용자 평균
- `Amount_vs_CardAvg`: 현재 금액 / 카드 평균

#### 처리 과정
1. MCC 카테고리 매핑 (6개 카테고리)
2. 무효 데이터 필터링
3. 고급 피처 엔지니어링
4. StandardScaler로 정규화
5. CSV 저장

**출력**:
- `02_data/01_processed/preprocessed_full_featured.csv`
- `02_data/01_processed/preprocessing_metadata.txt`

---

### 2단계: GPU 그리드 서치

**실행**: `./run_gpu.sh 01_src/01_training/02_gridsearch_gpu.py`

#### XGBoost 그리드 파라미터

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| `max_depth` | [6, 8, 10] | 트리 최대 깊이 |
| `learning_rate` | [0.01, 0.05, 0.1] | 학습률 |
| `n_estimators` | [100, 300, 500] | 트리 개수 |
| `subsample` | [0.8, 1.0] | 샘플 비율 |
| `colsample_bytree` | [0.8, 1.0] | 특성 샘플 비율 |

**총 조합**: 3 × 3 × 3 × 2 × 2 = **108개**

#### cuML RandomForest 그리드 파라미터

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| `n_estimators` | [100, 300, 500] | 트리 개수 |
| `max_depth` | [10, 16, 20] | 트리 최대 깊이 |
| `max_features` | [0.5, 0.8, 1.0] | 사용할 특성 비율 |

**총 조합**: 3 × 3 × 3 = **27개**

#### 평가 지표
- **CV 점수**: F1 Score (weighted)
- **테스트 평가**: Accuracy, F1 Score

#### 출력 파일

**최적 모델**:
- `03_models/05_gridsearch/best_xgboost_YYYYMMDD_HHMMSS.joblib`
- `03_models/05_gridsearch/best_randomforest_YYYYMMDD_HHMMSS.joblib`

**메타데이터** (JSON):
```json
{
  "model_name": "xgboost",
  "best_params": {
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 300,
    ...
  },
  "best_cv_score": 0.7523,
  "test_accuracy": 0.7612,
  "test_f1": 0.7589,
  "training_time": 1234.56
}
```

**CV 결과** (CSV):
- 모든 파라미터 조합별 성능
- Train/Test 점수
- 학습 시간

---

## 💡 사용 예시

### 예시 1: 전체 파이프라인 (전체 데이터)
```bash
# 2,400만 건 전체 데이터 사용
./run_full_pipeline.sh
```

**예상 소요 시간**:
- 전처리: ~30-60분
- 그리드 서치 (XGBoost): ~2-4시간
- 그리드 서치 (RandomForest): ~1-2시간

### 예시 2: 샘플 데이터로 테스트
```bash
# 10% 샘플로 빠른 테스트
./run_full_pipeline.sh 0.1
```

**예상 소요 시간**:
- 전처리: ~5분
- 그리드 서치: ~30분

### 예시 3: 전처리만 실행
```bash
python3 01_src/00_preprocessing/01_preprocess_full.py
```

### 예시 4: 그리드 서치만 실행 (전처리 완료 후)
```bash
./run_gpu.sh 01_src/01_training/02_gridsearch_gpu.py
```

---

## 🔍 결과 분석

### 1. 메타데이터 확인
```bash
cat 03_models/05_gridsearch/metadata_xgboost_*.json | jq .
```

### 2. CV 결과 분석
```python
import pandas as pd

cv_results = pd.read_csv('03_models/05_gridsearch/cv_results_xgboost_*.csv')

# 상위 10개 조합
top10 = cv_results.nlargest(10, 'mean_test_score')
print(top10[['params', 'mean_test_score', 'std_test_score']])
```

### 3. 최적 모델 로드
```python
import joblib

model = joblib.load('03_models/05_gridsearch/best_xgboost_*.joblib')

# 예측
predictions = model.predict(X_new)
```

---

## ⚙️ 설정 조정

### 데이터 샘플링 비율
`01_src/01_training/02_gridsearch_gpu.py` 파일에서:
```python
# 17번째 줄
sample_frac = 1.0  # 0.1 = 10%, 1.0 = 100%
```

### 그리드 파라미터 범위
`01_src/01_training/02_gridsearch_gpu.py` 파일에서:
```python
# XGBoost 그리드 (38-45번째 줄)
param_grid = {
    'max_depth': [6, 8, 10],          # 더 추가 가능
    'learning_rate': [0.01, 0.05, 0.1],
    ...
}
```

### CV Fold 수
```python
# 60번째 줄
cv=3,  # 5로 변경하면 더 정확하지만 느림
```

---

## 🎯 성능 최적화 팁

### GPU 메모리 부족 시
```python
# 배치 크기 줄이기
sample_frac = 0.5  # 50%만 사용
```

### 더 빠른 그리드 서치
```python
# 파라미터 범위 축소
param_grid = {
    'max_depth': [8],           # 하나만
    'learning_rate': [0.05],
    'n_estimators': [100, 300]  # 2개만
}
```

### RandomizedSearchCV 사용
```python
from sklearn.model_selection import RandomizedSearchCV

# GridSearchCV 대신 사용
random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_grid,
    n_iter=20,  # 20개 조합만 랜덤 샘플링
    cv=3,
    ...
)
```

---

## 📊 예상 성능

### 데이터 규모별 예상 시간 (GPU: RTX 4070 Ti)

| 데이터 크기 | 전처리 | XGBoost GS | RandomForest GS |
|-----------|--------|-----------|----------------|
| 100만 건 (4%) | 5분 | 20분 | 10분 |
| 500만 건 (20%) | 15분 | 1시간 | 30분 |
| 1,200만 건 (50%) | 30분 | 2시간 | 1시간 |
| 2,400만 건 (100%) | 60분 | 4시간 | 2시간 |

### 예상 성능 지표

| 모델 | Accuracy | F1 Score |
|------|---------|----------|
| XGBoost (기본) | ~0.40 | ~0.35 |
| XGBoost (튜닝 후) | ~0.45 | ~0.42 |
| RandomForest (기본) | ~0.29 | ~0.25 |
| RandomForest (튜닝 후) | ~0.35 | ~0.32 |

---

## 🐛 문제 해결

### GPU 메모리 부족
```bash
# 샘플링 비율 줄이기
./run_full_pipeline.sh 0.3
```

### 전처리 실패
```bash
# 메모리 확인
free -h

# 데이터 파일 확인
ls -lh 02_data/00_raw/credit_card_transactions-ibm_v2.csv
```

### cuML 없음
- RandomForest 그리드 서치는 자동으로 건너뜀
- XGBoost만 사용됨

---

## 📌 체크리스트

- [ ] GPU 환경 구성 완료 (`run_gpu.sh` 작동 확인)
- [ ] 원본 데이터 존재 (`02_data/00_raw/credit_card_transactions-ibm_v2.csv`)
- [ ] 충분한 디스크 공간 (최소 20GB)
- [ ] 충분한 GPU 메모리 (최소 8GB)
- [ ] 전처리 완료 (`02_data/01_processed/preprocessed_full_featured.csv`)
- [ ] 그리드 서치 완료 (`03_models/05_gridsearch/` 확인)

---

## 🔗 관련 문서

- GPU 환경 설정: `GPU_QUICK_START.md`
- GPU 상세 가이드: `05_docs/GPU_SETUP_GUIDE.md`
- Requirements: `requirements_gpu.txt`
