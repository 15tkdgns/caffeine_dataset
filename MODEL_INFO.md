# 🏆 Best Model Information

## 모델 개요
- **모델 타입**: XGBoost Classifier
- **파일명**: `best_model_xgboost_acc_73.47.joblib`
- **생성일**: 2025-12-08
- **저장 형식**: Joblib (압축됨)

## 성능 지표
- **정확도 (Accuracy)**: **73.47%** ✨
- **Macro F1 Score**: 77.14%
- **Weighted F1 Score**: 73.01%

### 카테고리별 F1 Score
| 카테고리 | F1 Score |
|---------|----------|
| 교통 | 95.74% 🥇 |
| 생활 | 92.16% 🥈 |
| 쇼핑 | 73.54% 🥉 |
| 외식 | 73.05% |
| 식료품 | 68.73% |
| 주유 | 59.61% |

## 모델 특징
### Enhanced Features (24개 피처)
1. **금액 관련**: Amount_clean, Amount_log, AmountBin
2. **시간 관련**: Hour, DayOfWeek, DayOfMonth, IsWeekend, IsNight, IsBusinessHour, IsLunchTime, IsEvening, IsMorningRush
3. **사용자 패턴**: User_AvgAmount, User_StdAmount, User_TxCount
4. **카테고리 비율**: User_교통_Ratio, User_생활_Ratio, User_쇼핑_Ratio, User_식료품_Ratio, User_외식_Ratio, User_주유_Ratio
5. **시퀀스 정보**: Previous_Category, Transaction_Sequence, Time_Since_Last

## 사용 방법

### 모델 로드
```python
import joblib

# 모델 로드
model = joblib.load('best_model_xgboost_acc_73.47.joblib')

# 예측
predictions = model.predict(X_test)

# 확률 예측
probabilities = model.predict_proba(X_test)
```

### 입력 데이터 형식
- **필수 피처 개수**: 24개
- **피처 순서**: `best_model_metadata.json` 파일의 `features` 항목 참조
- **데이터 타입**: NumPy array 또는 Pandas DataFrame

## 학습 정보
- **학습 시간**: 123.15초 (약 2분)
- **데이터셋**: 신용카드 거래 데이터 (6개 카테고리)
- **전처리**: StandardScaler, SMOTE 적용

## 비교 모델
| 모델 | 정확도 | Macro F1 | 학습 시간 |
|------|--------|----------|-----------|
| **XGBoost** | **73.47%** | **77.14%** | 123초 ⚡ |
| LightGBM | 71.72% | 75.45% | 908초 |
| CatBoost | 69.39% | 72.82% | 626초 |

## 프로덕션 체크리스트
- ✅ Joblib 형식으로 저장됨 (효율적 로딩)
- ✅ 메타데이터 파일 포함 (`best_model_metadata.json`)
- ✅ 높은 정확도 (73.47%)
- ✅ 빠른 학습 시간 (2분 미만)
- ✅ 균형잡힌 카테고리 성능

## 주의사항
⚠️ **버전 호환성**: 
- Python 버전과 라이브러리 버전이 일치해야 합니다
- `requirements.txt` 파일을 함께 관리하세요

⚠️ **입력 데이터 검증**:
- 24개 피처가 정확한 순서로 제공되어야 합니다
- 결측값 처리가 필요합니다
- 스케일링이 동일하게 적용되어야 합니다

## 파일 목록
- `best_model_xgboost_acc_73.47.joblib` - 모델 파일 (76MB)
- `best_model_metadata.json` - 메타데이터 및 성능 지표
- `MODEL_INFO.md` - 이 문서

---
📅 **Last Updated**: 2025-12-10
🔧 **Saved with**: joblib
✨ **Status**: Production Ready
