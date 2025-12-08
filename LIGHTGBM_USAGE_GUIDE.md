# 🎯 LightGBM 프로덕션 모델 사용 가이드

## 📦 모델 파일

**모델 파일**: `03_models/production_models/lightgbm_cuda_production_20251205_162340.joblib`
- 크기: 24.46 MB
- 생성일: 2025-12-05 16:23:45

**메타데이터**: `03_models/production_models/lightgbm_cuda_metadata_20251205_162340.json`

---

## 📊 모델 성능

| 지표 | 값 |
|------|-----|
| **Accuracy** | **49.13%** |
| **Macro F1** | **43.44%** |
| **Weighted F1** | **47.27%** |
| **학습 시간** | 7.56분 (453초) |

### 카테고리별 F1 Score

| 카테고리 | F1 Score |
|---------|----------|
| 🚗 교통 | **64.96%** ⭐ 최고 |
| 🏠 생활 | 8.02% |
| 🛍️ 쇼핑 | 34.78% |
| 🥬 식료품 | 54.14% |
| 🍽️ 외식 | 44.34% |
| ⛽ 주유 | 54.41% |

---

## 🎯 입력 데이터 스펙

### 필수 입력 형태
- **피처 개수**: 27개 (고정)
- **데이터 타입**: `float32`
- **입력 형태**: `(n_samples, 27)` numpy array 또는 pandas DataFrame
- **결측값**: 허용 안 됨 (사전 처리 필수)
- **전처리**: StandardScaler 정규화 적용 필요

### ⭐ 27개 피처 목록 (순서대로)

#### 1. 금액 관련 (3개)
1. `Amount_scaled` - 정규화된 거래 금액
2. `Amount_log_scaled` - 로그 변환 + 정규화 금액
3. `AmountBin_encoded_scaled` - 금액 구간 인코딩

#### 2. 시간 관련 (12개)
4. `Hour_scaled` - 시간대 (0-23)
5. `DayOfWeek_scaled` - 요일 (0-6)
6. `DayOfMonth_scaled` - 일자 (1-31)
7. `IsWeekend_scaled` - 주말 여부
8. `IsLunchTime_scaled` - 점심시간 (11-14시)
9. `IsEvening_scaled` - 저녁시간 (18-22시)
10. `IsMorningRush_scaled` - 출근시간 (7-9시)
11. `IsNight_scaled` - 야간 (22시-6시)
12. `IsBusinessHour_scaled` - 업무시간 (9-18시)
13. `User_AvgAmount_scaled` - 사용자 평균 거래금액
14. `User_StdAmount_scaled` - 사용자 거래금액 표준편차
15. `User_TxCount_scaled` - 사용자 총 거래 건수

#### 3. 시퀀스 관련 (3개)
16. `Time_Since_Last_scaled` - 마지막 거래 이후 시간
17. `Transaction_Sequence_scaled` - 거래 순서
18. `User_Category_Count_scaled` - 사용자 카테고리 수

#### 4. 카테고리 관련 (9개)
19. `Current_Category_encoded_scaled` - 현재 카테고리
20. `Previous_Category_encoded_scaled` - 이전 카테고리
21. `User_FavCategory_encoded_scaled` - 선호 카테고리
22. `User_교통_Ratio_scaled` - 교통비 비율
23. `User_생활_Ratio_scaled` - 생활비 비율
24. `User_쇼핑_Ratio_scaled` - 쇼핑비 비율
25. `User_식료품_Ratio_scaled` - 식료품비 비율
26. `User_외식_Ratio_scaled` - 외식비 비율
27. `User_주유_Ratio_scaled` - 주유비 비율

---

## 📤 출력 데이터 스펙

### 예측 클래스
| 클래스 ID | 카테고리 |
|----------|---------|
| 0 | 교통 |
| 1 | 생활 |
| 2 | 쇼핑 |
| 3 | 식료품 |
| 4 | 외식 |
| 5 | 주유 |

### 예측 메서드
- `model.predict(X)` → 클래스 레이블 (0-5) 반환
- `model.predict_proba(X)` → 6개 클래스에 대한 확률 분포 반환

---

## 💻 사용 예시

### Python 코드

```python
import joblib
import numpy as np

# 1. 모델 로드
model = joblib.load('03_models/production_models/lightgbm_cuda_production_20251205_162340.joblib')

# 2. 입력 데이터 준비 (예시)
# ⚠️ 반드시 27개 피처를 올바른 순서로!
X_input = np.array([[
    # 금액 (3개)
    0.5, 0.3, 0.2,
    
    # 시간 (9개)
    0.6, 0.4, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    
    # 사용자 프로필 (3개)
    0.4, 0.3, 0.5,
    
    # 시퀀스 (3개)
    0.2, 0.7, 0.8,
    
    # 카테고리 (9개)
    2.0, 1.0, 3.0, 0.1, 0.2, 0.3, 0.15, 0.2, 0.05
]], dtype=np.float32)

# 3. 예측 (클래스)
y_pred = model.predict(X_input)
print(f"예측 카테고리 ID: {y_pred[0]}")  # 0-5

# 4. 예측 (확률)
y_proba = model.predict_proba(X_input)
print(f"예측 확률:")
categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
for cat, prob in zip(categories, y_proba[0]):
    print(f"  {cat}: {prob:.4f} ({prob*100:.2f}%)")
```

### 배치 예측

```python
# 여러 거래를 동시에 예측
X_batch = np.array([
    [0.5, 0.3, 0.2, ...],  # 거래 1
    [0.7, 0.1, 0.5, ...],  # 거래 2
    [0.2, 0.8, 0.3, ...],  # 거래 3
], dtype=np.float32)

y_pred_batch = model.predict(X_batch)
print(y_pred_batch)  # [2, 4, 0] 예시
```

### pandas DataFrame 입력

```python
import pandas as pd

# DataFrame으로 입력
df_input = pd.DataFrame({
    'Amount_scaled': [0.5],
    'Amount_log_scaled': [0.3],
    'AmountBin_encoded_scaled': [0.2],
    # ... 나머지 24개 피처
})

y_pred = model.predict(df_input)
```

---

## ⚠️ 주의사항

1. **피처 순서가 매우 중요합니다!**
   - 반드시 위에 명시된 순서대로 27개 피처를 입력해야 합니다
   - 순서가 바뀌면 예측이 완전히 잘못됩니다

2. **StandardScaler 정규화 필수**
   - 모든 피처는 학습 시 사용된 Scaler로 정규화되어야 합니다
   - 메타데이터의 `feature_statistics`를 참고하세요

3. **결측값 허용 안 됨**
   - 모든 피처는 유효한 값이 있어야 합니다
   - 예측 전에 결측값을 반드시 처리하세요 (평균, 중앙값 등)

4. **데이터 타입**
   - `float32` 타입 사용 권장
   - `float64`도 가능하지만 메모리 낭비

5. **GPU 불필요**
   - 모델이 GPU에서 학습되었지만
   - CPU에서도 예측 가능합니다

---

## 🔧 전처리 파이프라인 예시

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# 원본 데이터 (예시)
raw_data = {
    'Amount': 45.50,
    'Hour': 14,
    'DayOfWeek': 2,
    'IsWeekend': 0,
    # ... 기타 피처
}

# StandardScaler로 정규화
# ⚠️ 학습 시 사용된 Scaler를 저장해두고 재사용해야 함!
scaler = joblib.load('scaler.joblib')  # 학습 시 저장한 Scaler

# 피처 엔지니어링 + 정규화
X_processed = preprocess_and_scale(raw_data, scaler)

# 예측
y_pred = model.predict(X_processed)
```

---

## 📚 참고 파일

- **전체 메타데이터**: `03_models/production_models/lightgbm_cuda_metadata_20251205_162340.json`
  - 상세한 피처 통계 (평균, 표준편차, 최소/최대값)
  - 모델 파라미터
  - 학습 데이터 정보

- **성능 분석**: `/root/ibm_data2/MODEL_PERFORMANCE_SUMMARY.md`
  - 다른 모델과의 성능 비교
  - 상세 분석 결과

---

## 🚀 프로덕션 배포

### FastAPI 예시

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('lightgbm_cuda_production_20251205_162340.joblib')

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.features) != 27:
        raise HTTPException(400, "27개 피처 필요")
    
    X = np.array([request.features], dtype=np.float32)
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]
    
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    
    return {
        "predicted_category_id": int(y_pred),
        "predicted_category": categories[y_pred],
        "probabilities": {
            cat: float(prob) 
            for cat, prob in zip(categories, y_proba)
        }
    }
```

---

**마지막 업데이트**: 2025-12-05  
**문의**: 프로젝트 관리자
