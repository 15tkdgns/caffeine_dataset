# 신용카드 거래 카테고리 예측 프로젝트

IBM Credit Card Transactions 데이터를 사용한 6개 소비 카테고리 분류 모델

## 프로젝트 개요

- **데이터**: 24.4M 신용카드 거래 데이터
- **목표**: 거래를 6개 카테고리(교통, 생활, 쇼핑, 식료품, 외식, 주유)로 자동 분류
- **최종 모델**: Stacking Ensemble
- **성능**: Accuracy 49.62%, Macro F1 45.24%

## 프로젝트 구조

```
ibm_data2/
├── 00_config/              # 설정 파일
│   └── 00_mapping/        # 카테고리 매핑
├── 02_data/               # 데이터 (Git에서 제외)
│   ├── 00_raw/           # 원본 CSV
│   ├── 02_augmented/     # 전처리된 데이터
│   └── 03_advanced/      # 고급 피처
├── 03_models/             # 학습된 모델 (Git에서 제외)
│   ├── production_models/
│   ├── ensemble/
│   └── optuna_tuned/
├── 04_logs/               # 실험 로그 (Git에서 제외)
│   ├── stacking/
│   ├── optuna_deep/
│   └── analysis/
├── dashboard_final_report.py  # Streamlit 대시보드
├── stacking_ensemble.py       # 최종 모델 학습 스크립트
├── requirements.txt           # 필요 라이브러리
└── README.md                  # 본 문서
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/15tkdgns/caffeine_dataset.git
cd caffeine_dataset

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 라이브러리 설치
pip install -r requirements.txt
```

### 2. 대시보드 실행

```bash
streamlit run dashboard_final_report.py
```

브라우저에서 http://localhost:8501 접속

## 주요 실험 결과

| 모델 | Accuracy | Macro F1 | 설명 |
|------|----------|----------|------|
| Baseline (LightGBM) | 49.13% | 43.44% | 기본 모델 |
| Class Weight 조정 | 44.85% | 42.87% | 불균형 해소 시도 |
| ADASYN 증강 | 46.95% | 44.89% | 데이터 증강 |
| Accuracy 최적화 | 49.55% | 42.84% | 하이퍼파라미터 튜닝 |
| **Stacking Ensemble** | **49.62%** | **45.24%** | **최종 선정** |
| Optuna 심화 | 49.49% | 43.00% | 100 trials 튜닝 |

## 최종 모델: Stacking Ensemble

### 구성
- **Base Model 1**: LightGBM (n_estimators=500, max_depth=12)
- **Base Model 2**: XGBoost (n_estimators=500, max_depth=12)
- **Base Model 3**: LightGBM (n_estimators=400, max_depth=15)
- **Meta-Learner**: LightGBM (n_estimators=200, max_depth=5)
- **방법**: 5-Fold Cross-Validation

### 사용 피처 (27개)
- 시간 피처 (6개): Hour, DayOfWeek, Is_Weekend 등
- 금액 피처 (3개): Amount, Amount_log, Amount_bin
- 사용자 피처 (12개): User_AvgAmount, User_*_Ratio 등
- 기타 (6개): Card/MCC 관련

## 파일 설명

### 주요 스크립트
- `dashboard_final_report.py`: Streamlit 대시보드 (6개 페이지)
- `stacking_ensemble.py`: Stacking 모델 학습
- `analyze_baseline.py`: Baseline 분석
- `step2_class_weight.py`: Class Weight 실험
- `step3_adasyn.py`: ADASYN 증강 실험
- `accuracy_optimization.py`: Accuracy 최적화
- `optuna_deep.py`: Optuna 심화 튜닝

## 대시보드 페이지

1. **프로젝트 개요**: 데이터 분포, 파이프라인
2. **Stacking Ensemble 모델**: 구조, 플로우차트
3. **모델 의사결정 과정**: 예측 과정 시각화, XAI 분석
4. **실험 결과 비교**: 전체 실험 성능 비교
5. **피처 분석**: 27개 피처 중요도
6. **결론**: 최종 모델 정보, 향후 개선 방안

## 기술 스택

- **언어**: Python 3.12
- **머신러닝**: LightGBM, XGBoost
- **전처리**: Pandas, NumPy, scikit-learn
- **하이퍼파라미터 튜닝**: Optuna
- **시각화**: Plotly, Matplotlib
- **대시보드**: Streamlit

## 라이센스

MIT License

## 작성자

- 프로젝트 기간: 2025-12-05 ~ 2025-12-08
