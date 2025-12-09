# 72.65% Macro F1 프로젝트 구조

## 📁 프로젝트 개요
신용카드 거래 데이터를 이용한 소비 카테고리 예측 모델
- **최종 성능**: Macro F1 72.65%, Accuracy 65.79%
- **모델**: XGBoost (GPU)
- **데이터**: 시간 기반 Split (Train: 2010-2018, Test: 2018-2020)
- **카테고리**: 6개 (교통, 생활, 쇼핑, 식료품, 외식, 주유)

---

## 📂 핵심 파일 구조

```
/root/ibm_data2/
├── 00_config/              # 카테고리 매핑
│   └── category_mapping.py
│
├── 02_data/
│   └── 07_time_optimized/  # 최종 모델 데이터 (72.65% F1)
│       ├── xgboost_final.joblib
│       ├── metadata.json
│       └── preprocessed_transactions_v2.csv
│
├── scripts/                # 학습 파이프라인
│   ├── pipeline_time_based.py
│   └── pipeline_time_optimized.py
│
├── dashboard_final.py      # Streamlit 대시보드
├── GEMINI.md              # 프로젝트 메모리
├── PROJECT_STRUCTURE.md   # 프로젝트 구조 설명
├── README.md              # 프로젝트 설명
├── requirements.txt       # Python 패키지
├── .gitignore            # Git 무시 파일
│
└── 99_archive/           # 실험 파일 보관소 (Git 제외)
    ├── 01_src/           # 이전 소스 코드
    ├── 02_data/          # 이전 데이터 (00~06)
    ├── 03_models/        # 이전 모델들
    ├── 실험 스크립트들
    └── 문서들
```

---

## 🚀 실행 방법

### 1. 대시보드 실행
```bash
streamlit run dashboard_final.py --server.port 8501
```

### 2. 모델 재학습 (필요 시)
```bash
# 1단계: 시간 기반 전처리
python scripts/pipeline_time_based.py

# 2단계: 최적화 (SMOTE + Optuna)
python scripts/pipeline_time_optimized.py
```

---

## 📊 모델 성능

| 카테고리 | F1 Score | 비고 |
|---------|----------|------|
| 교통 | 95.30% | ✅ 강점 |
| 생활 | 94.53% | ✅ 강점 |
| 쇼핑 | 74.50% | |
| 외식 | 68.37% | |
| 식료품 | 53.46% | ⚠️ 개선 필요 |
| 주유 | 49.76% | ⚠️ 개선 필요 |
| **평균** | **72.65%** | |

---

## 🔧 주요 기술

- **모델**: XGBoost (GPU)
- **전처리**: 시간 기반 Train/Test Split
- **클래스 불균형**: SMOTE + Class Weight
- **최적화**: Optuna (30 trials)
- **피처**: 23개 (금액, 시간, 사용자 프로필, 시퀀스)

---

## 📝 참고

- 모든 실험 파일은 `99_archive/` 폴더에 보관
- Git에는 핵심 파일만 포함 (`.gitignore` 참조)
- 최종 모델 파일: `02_data/07_time_optimized/xgboost_final.joblib`
