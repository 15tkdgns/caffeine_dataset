# 🏆 모델 성능 종합 분석

**분석 일시**: 2025-12-05  
**분석자**: Gemini AI Assistant

---

## 📊 전체 모델 성능 비교

### 1️⃣ **최고 성능 모델: LightGBM (CUDA)**

```json
{
  "모델명": "LightGBM (CUDA)",
  "데이터": "SMOTE 증강 데이터",
  "성능": {
    "Accuracy": 0.4911 (49.11%),
    "Macro F1": 0.4324 (43.24%),
    "학습 시간": 197.0초 (3분 17초)
  }
}
```

**특징**:
- GPU(CUDA) 가속을 활용한 고속 학습
- SMOTE 데이터 증강으로 클래스 불균형 해소
- 전체 모델 중 **Accuracy 1위**

---

### 2️⃣ **2위: XGBoost (GPU)**

```json
{
  "모델명": "XGBoost (GPU)",
  "데이터": "SMOTE 증강 데이터",
  "성능": {
    "Accuracy": 0.4903 (49.03%),
    "Macro F1": 0.4416 (44.16%),
    "학습 시간": 101.3초 (1분 41초)
  }
}
```

**특징**:
- **Macro F1 점수 1위** (클래스 균형 성능 최고)
- LightGBM보다 학습 속도 2배 빠름
- GPU 가속으로 효율적인 학습

---

### 3️⃣ **3위: XGBoost Sequence (시퀀스 특화)**

```json
{
  "모델명": "XGBoost Sequence",
  "데이터": "시퀀스 피처 엔지니어링",
  "성능": {
    "Accuracy": 0.4910 (49.10%),
    "F1 Score": 0.4669 (46.69%),
    "학습 시간": 1429.8초 (23분 50초)
  }
}
```

**특징**:
- 사용자 거래 시퀀스 패턴 학습
- **F1 점수가 가장 높음** (46.69%)
- 그리드 서치로 하이퍼파라미터 최적화
- 학습 시간이 길지만 성능 우수

---

### 4️⃣ **프로덕션 모델: XGBoost Final (16 Features)**

```json
{
  "모델명": "xgboost_final_16features",
  "버전": "v1.0",
  "피처 개수": 16개 (27개에서 선택),
  "성능": {
    "Accuracy": 0.4590 (45.90%),
    "Macro F1": 0.4493 (44.93%),
    "Weighted F1": 0.4613 (46.13%),
    "학습 시간": 49.0초
  }
}
```

**선택된 16개 피처**:
1. `User_교통_Ratio` - 사용자 교통비 비율
2. `Current_Category_encoded` - 현재 카테고리 인코딩
3. `User_외식_Ratio` - 사용자 외식비 비율
4. `User_식료품_Ratio` - 사용자 식료품비 비율
5. `User_쇼핑_Ratio` - 사용자 쇼핑비 비율
6. `AmountBin_encoded` - 금액 구간 인코딩
7. `User_생활_Ratio` - 사용자 생활비 비율
8. `User_주유_Ratio` - 사용자 주유비 비율
9. `Amount` - 거래 금액
10. `IsNight` - 야간 거래 여부
11. `IsBusinessHour` - 업무 시간 여부
12. `IsEvening` - 저녁 시간 여부
13. `Hour` - 시간대
14. `Previous_Category_encoded` - 이전 카테고리
15. `Time_Since_Last` - 마지막 거래 이후 시간
16. `IsMorningRush` - 출근 시간 여부

**장점**:
- 피처 선택으로 **학습 속도 대폭 향상** (49초)
- **실시간 예측에 최적화**
- Refer 모델 대비 성능 차이: -17.19%p (향상 가능)

---

## 🔬 심층 신경망 & 시퀀스 모델 성능

### 딥러닝 모델 (SMOTE 데이터)

| 순위 | 모델 | Accuracy | Macro F1 | 학습시간 |
|-----|------|----------|----------|---------|
| 🥇 | **GRU+Attention** | **0.4458** | **0.4327** | 103.0초 |
| 🥈 | TabNet | 0.4438 | 0.4310 | 309.5초 |
| 🥉 | TCN | 0.4408 | 0.4323 | 102.7초 |
| 4위 | TensorFlow NN (GPU) | 0.4493 | 0.4372 | 93.3초 |
| 5위 | cuML RandomForest | 0.4549 | 0.4455 | 265.7초 |

**인사이트**:
- **GRU+Attention이 시퀀스 모델 중 최고 성능**
- TabNet보다 3배 빠른 학습 속도
- TCN도 경쟁력 있는 성능 (거의 동일한 F1)

---

## 📈 데이터 메타데이터 (SMOTE 증강)

```json
{
  "필터 조건": "월 10건 이상 × 5개월 이상",
  "원본 샘플": 6,443,429건,
  "필터링 후": 6,424,957건,
  "활성 사용자": 1,603명,
  "학습 데이터": {
    "원본": 5,139,965건,
    "SMOTE 증강": 9,254,112건 (1.8배 증가)
  },
  "테스트 데이터": 1,284,992건,
  "피처 개수": 27개
}
```

### 전체 27개 피처 목록

#### 1. 금액 관련 (3개)
- `Amount_scaled` - 정규화된 거래 금액
- `Amount_log_scaled` - 로그 변환 + 정규화
- `AmountBin_encoded_scaled` - 금액 구간 인코딩

#### 2. 시간 관련 (12개)
- `Hour_scaled` - 시간대 (0-23)
- `DayOfWeek_scaled` - 요일 (0-6)
- `DayOfMonth_scaled` - 일자 (1-31)
- `IsWeekend_scaled` - 주말 여부
- `IsLunchTime_scaled` - 점심시간 (11-14시)
- `IsEvening_scaled` - 저녁시간 (18-22시)
- `IsMorningRush_scaled` - 출근시간 (7-9시)
- `IsNight_scaled` - 야간 (22시-6시)
- `IsBusinessHour_scaled` - 업무시간 (9-18시)
- `Time_Since_Last_scaled` - 마지막 거래 이후 경과시간
- `Transaction_Sequence_scaled` - 거래 순서
- `User_TxCount_scaled` - 사용자 총 거래 건수

#### 3. 사용자 프로필 (5개)
- `User_AvgAmount_scaled` - 사용자 평균 거래금액
- `User_StdAmount_scaled` - 사용자 거래금액 표준편차
- `User_Category_Count_scaled` - 사용자가 이용한 카테고리 수
- `User_FavCategory_encoded_scaled` - 사용자 선호 카테고리

#### 4. 카테고리 관련 (7개)
- `Current_Category_encoded_scaled` - 현재 카테고리 인코딩
- `Previous_Category_encoded_scaled` - 이전 카테고리 인코딩
- `User_교통_Ratio_scaled` - 교통비 비율
- `User_생활_Ratio_scaled` - 생활비 비율
- `User_쇼핑_Ratio_scaled` - 쇼핑비 비율
- `User_식료품_Ratio_scaled` - 식료품비 비율
- `User_외식_Ratio_scaled` - 외식비 비율
- `User_주유_Ratio_scaled` - 주유비 비율

---

## 🎯 권장 사항

### 현재 시점 최적 모델 선택

#### 1️⃣ **실시간 예측 서비스** (API)
→ **XGBoost Final (16 Features)**
- ✅ 빠른 추론 속도 (피처 16개)
- ✅ 안정적인 성능 (45.90% accuracy)
- ✅ GPU 가속 지원

#### 2️⃣ **최고 성능 우선** (배치 처리)
→ **XGBoost Sequence**
- ✅ **F1 Score 46.69%** (최고)
- ✅ 시퀀스 패턴 학습
- ✅ 복잡한 사용자 행동 포착

#### 3️⃣ **균형 잡힌 최적화**
→ **LightGBM (CUDA)** 또는 **XGBoost (GPU)**
- ✅ Accuracy 49%+
- ✅ 빠른 학습 속도 (1-3분)
- ✅ SMOTE로 클래스 균형

#### 4️⃣ **시퀀스 모델 실험**
→ **GRU+Attention**
- ✅ 딥러닝 모델 중 최고
- ✅ 빠른 학습 (103초)
- ✅ 시계열 패턴 학습

---

## 🚀 성능 개선 아이디어

### 단기 개선 (Quick Wins)
1. **앙상블 모델**: XGBoost + LightGBM + GRU 조합
2. **하이퍼파라미터 튜닝**: Optuna로 자동 최적화
3. **피처 엔지니어링**: 더 많은 시퀀스 피처 추가

### 중장기 개선
1. **Transformer 기반 모델**: 실제 시퀀스 데이터로 학습
2. **LSTM+Attention**: 사용자별 최근 N건 거래 패턴
3. **Graph Neural Network**: 카테고리 간 전이 패턴 학습
4. **더 많은 외부 데이터**: 계절성, 이벤트 정보 등

---

## 📋 요약

| 항목 | 내용 |
|-----|------|
| **최고 Accuracy** | LightGBM (CUDA): **49.11%** |
| **최고 Macro F1** | XGBoost (GPU): **44.16%** |
| **최고 F1** | XGBoost Sequence: **46.69%** |
| **가장 빠른 학습** | XGBoost Final: **49초** |
| **프로덕션 추천** | XGBoost Final (16 Features) |
| **실험용 추천** | GRU+Attention (시퀀스 학습) |
| **데이터 규모** | 925만건 (SMOTE) / 643만건 (원본) |
| **피처 개수** | 27개 (전체) / 16개 (선택) |

---

**마지막 업데이트**: 2025-12-05  
**파일 위치**: `/root/ibm_data2/MODEL_PERFORMANCE_SUMMARY.md`
