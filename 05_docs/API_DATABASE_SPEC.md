# 📊 FastAPI ML 서비스 데이터베이스 명세서

## 서비스 개요

**목적**: 개인 소비 데이터 분석 및 예측 API  
**입력**: 개인 거래 내역 CSV (`2024-12-03~2025-12-03.csv`)  
**출력**: 
1. 다음 소비 카테고리 예측
2. 소비 분석 리포트
3. 맞춤형 광고 추천
4. 이상 거래 탐지

---

## 데이터베이스 설계

### 1. 사용자 관리

#### 1.1 users (사용자 테이블)
```sql
CREATE TABLE users (
    user_id VARCHAR(36) PRIMARY KEY,              -- UUID
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    
    INDEX idx_email (email),
    INDEX idx_created_at (created_at)
);
```

**용도**: 사용자 인증 및 관리

**샘플 데이터**:
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "name": "홍길동",
  "created_at": "2024-12-01 10:00:00",
  "is_active": true
}
```

---

### 2. 거래 데이터

#### 2.1 transactions (거래 내역)
```sql
CREATE TABLE transactions (
    transaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_time TIME NOT NULL,
    transaction_type ENUM('지출', '수입', '이체') NOT NULL,
    
    -- 카테고리
    category_main VARCHAR(50) NOT NULL,           -- 대분류 (원본)
    category_sub VARCHAR(50),                     -- 소분류
    category_mapped ENUM('교통', '생활', '쇼핑', '식료품', '외식', '주유') NOT NULL,  -- 6개 매핑
    
    -- 금액
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'KRW',
    
    -- 상세 정보
    description TEXT,                              -- 내용
    merchant_name VARCHAR(255),                    -- 가맹점
    payment_method VARCHAR(100),                   -- 결제수단
    memo TEXT,
    
    -- 처리 상태
    is_processed BOOLEAN DEFAULT FALSE,            -- API 처리 여부
    processed_at TIMESTAMP NULL,
    
    -- 메타
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_date (user_id, transaction_date),
    INDEX idx_category (category_mapped),
    INDEX idx_processed (is_processed)
);
```

**용도**: 원본 거래 데이터 저장 (CSV 업로드 시 삽입)

**샘플 데이터**:
```json
{
  "transaction_id": 1,
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "transaction_date": "2025-12-02",
  "transaction_time": "19:28:00",
  "transaction_type": "지출",
  "category_main": "식비",
  "category_sub": "편의점",
  "category_mapped": "외식",
  "amount": -8400.00,
  "currency": "KRW",
  "description": "GS25",
  "payment_method": "KB카드",
  "is_processed": true
}
```

---

### 3. 예측 결과

#### 3.1 predictions (예측 결과)
```sql
CREATE TABLE predictions (
    prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    transaction_id BIGINT NULL,                    -- 마지막 거래 (트리거)
    
    -- 예측 정보
    predicted_category ENUM('교통', '생활', '쇼핑', '식료품', '외식', '주유') NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,             -- 0.0000 ~ 1.0000
    
    -- 확률 분포 (JSON)
    category_probabilities JSON NOT NULL,          -- {"교통": 0.15, "생활": 0.10, ...}
    
    -- 예측 근거
    top_features JSON,                             -- 주요 피처와 영향도
    prediction_reason TEXT,                        -- 설명 텍스트
    
    -- 예측 컨텍스트
    last_category VARCHAR(20),                     -- 마지막 거래 카테고리
    time_since_last_transaction INT,               -- 마지막 거래와의 시간차 (초)
    avg_amount DECIMAL(12, 2),                     -- 사용자 평균 금액
    
    -- 모델 정보
    model_version VARCHAR(50) NOT NULL,            -- "xgboost_v1.0"
    model_accuracy DECIMAL(5, 4),
    
    -- 타임스탬프
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE SET NULL,
    INDEX idx_user_predicted (user_id, predicted_at),
    INDEX idx_category (predicted_category)
);
```

**용도**: ML 모델 예측 결과 저장

**샘플 데이터**:
```json
{
  "prediction_id": 100,
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "predicted_category": "외식",
  "confidence": 0.4567,
  "category_probabilities": {
    "교통": 0.12,
    "생활": 0.08,
    "쇼핑": 0.15,
    "식료품": 0.20,
    "외식": 0.46,
    "주유": 0.09
  },
  "top_features": {
    "User_외식_Ratio": 0.35,
    "Current_Category": "식료품",
    "IsEvening": 1
  },
  "prediction_reason": "최근 식료품 구매 이후 저녁 시간에 외식 패턴이 높습니다.",
  "last_category": "식료품",
  "model_version": "xgboost_enhanced_v1.0",
  "model_accuracy": 0.4852
}
```

---

### 4. 소비 분석 리포트

#### 4.1 spending_reports (소비 분석)
```sql
CREATE TABLE spending_reports (
    report_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    
    -- 분석 기간
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- 전체 요약
    total_transactions INT NOT NULL,
    total_spending DECIMAL(15, 2) NOT NULL,
    total_income DECIMAL(15, 2) NOT NULL,
    net_amount DECIMAL(15, 2) NOT NULL,            -- 수입 - 지출
    
    -- 카테고리별 통계 (JSON)
    category_breakdown JSON NOT NULL,              -- {"외식": {"count": 50, "amount": 500000, "ratio": 0.3}, ...}
    
    -- 패턴 분석
    most_frequent_category VARCHAR(20),
    highest_spending_category VARCHAR(20),
    avg_transaction_amount DECIMAL(12, 2),
    
    -- 시간 패턴
    weekday_vs_weekend JSON,                       -- {"weekday": 0.6, "weekend": 0.4}
    peak_hours JSON,                               -- {"09": 5, "12": 12, "18": 15, ...}
    
    -- 이상치 정보
    anomaly_count INT DEFAULT 0,
    anomaly_transactions JSON,                     -- [{"transaction_id": 123, "score": 0.95}, ...]
    
    -- 비교 분석
    vs_previous_period JSON,                       -- {"spending_change": 0.15, "pattern_change": "increased"}
    vs_user_segment JSON,                          -- {"percentile": 75, "segment": "high_spender"}
    
    -- 인사이트
    insights JSON,                                 -- [{"type": "warning", "message": "외식비 급증"}, ...]
    
    -- 생성 정보
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_period (user_id, period_end),
    UNIQUE KEY unique_user_period (user_id, period_start, period_end)
);
```

**용도**: 사용자별 소비 패턴 분석 결과

**샘플 데이터**:
```json
{
  "report_id": 50,
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "period_start": "2024-12-01",
  "period_end": "2024-12-31",
  "total_transactions": 283,
  "total_spending": 2500000.00,
  "total_income": 3000000.00,
  "net_amount": 500000.00,
  "category_breakdown": {
    "외식": {"count": 82, "amount": 750000, "ratio": 0.30},
    "쇼핑": {"count": 45, "amount": 500000, "ratio": 0.20},
    "식료품": {"count": 60, "amount": 400000, "ratio": 0.16}
  },
  "most_frequent_category": "외식",
  "highest_spending_category": "외식",
  "avg_transaction_amount": 8834.00,
  "weekday_vs_weekend": {"weekday": 0.65, "weekend": 0.35},
  "peak_hours": {"12": 25, "18": 35, "19": 30},
  "anomaly_count": 3,
  "insights": [
    {"type": "warning", "message": "이번 달 외식비가 평소보다 40% 증가했습니다."},
    {"type": "positive", "message": "식료품 지출이 10% 감소하여 절약 중입니다."}
  ]
}
```

---

### 5. 광고 추천

#### 5.1 ad_recommendations (광고 추천)
```sql
CREATE TABLE ad_recommendations (
    recommendation_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    
    -- 타겟팅 정보
    target_category VARCHAR(20) NOT NULL,          -- 추천 카테고리
    user_affinity_score DECIMAL(5, 4),             -- 사용자 친화도
    
    -- 광고 정보
    ad_id VARCHAR(100) NOT NULL,
    ad_title VARCHAR(255) NOT NULL,
    ad_description TEXT,
    ad_image_url VARCHAR(500),
    ad_url VARCHAR(500),
    
    -- 광고 타입
    ad_type ENUM('쿠폰', '할인', '신상품', '이벤트', '맞춤추천') NOT NULL,
    
    -- 광고 내용
    discount_rate DECIMAL(5, 2),                   -- 할인율 (%)
    coupon_code VARCHAR(50),
    valid_until DATE,
    
    -- 추천 근거
    recommendation_reason TEXT,
    prediction_id BIGINT,                          -- 연관 예측
    
    -- 성과 지표
    is_clicked BOOLEAN DEFAULT FALSE,
    clicked_at TIMESTAMP NULL,
    is_converted BOOLEAN DEFAULT FALSE,
    converted_at TIMESTAMP NULL,
    conversion_amount DECIMAL(12, 2),
    
    -- 우선순위
    priority INT DEFAULT 0,                        -- 높을수록 우선
    relevance_score DECIMAL(5, 4),                 -- 관련성 점수
    
    -- 타임스탬프
    recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE SET NULL,
    INDEX idx_user_recommended (user_id, recommended_at),
    INDEX idx_category (target_category),
    INDEX idx_performance (is_clicked, is_converted)
);
```

**용도**: 예측 기반 맞춤형 광고 추천

**샘플 데이터**:
```json
{
  "recommendation_id": 200,
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "target_category": "외식",
  "user_affinity_score": 0.4567,
  "ad_id": "AD_RESTAURANT_001",
  "ad_title": "인기 레스토랑 30% 할인 쿠폰",
  "ad_description": "다음 외식 시 사용 가능한 특별 할인",
  "ad_type": "쿠폰",
  "discount_rate": 30.00,
  "coupon_code": "REST30OFF",
  "valid_until": "2025-01-31",
  "recommendation_reason": "최근 외식 빈도가 높고, 다음 구매도 외식일 확률 46%",
  "priority": 10,
  "relevance_score": 0.8500,
  "is_clicked": false
}
```

---

### 6. 이상치 탐지

#### 6.1 anomalies (이상 거래)
```sql
CREATE TABLE anomalies (
    anomaly_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    transaction_id BIGINT NOT NULL,
    
    -- 이상치 정보
    anomaly_score DECIMAL(5, 4) NOT NULL,          -- 0.0000 ~ 1.0000 (높을수록 이상)
    anomaly_type ENUM('금액', '시간', '빈도', '패턴', '복합') NOT NULL,
    
    -- 상세 분석
    expected_amount DECIMAL(12, 2),                -- 예상 금액
    actual_amount DECIMAL(12, 2),                  -- 실제 금액
    deviation_ratio DECIMAL(5, 2),                 -- 편차 비율
    
    expected_category VARCHAR(20),                 -- 예상 카테고리
    actual_category VARCHAR(20),                   -- 실제 카테고리
    
    -- 컨텍스트
    user_avg_amount DECIMAL(12, 2),
    user_std_amount DECIMAL(12, 2),
    z_score DECIMAL(8, 4),                         -- 표준 점수
    
    -- 이상 근거
    anomaly_reasons JSON,                          -- [{"factor": "amount", "score": 0.8}, ...]
    anomaly_description TEXT,
    
    -- 리스크 평가
    risk_level ENUM('낮음', '중간', '높음', '매우높음') NOT NULL,
    requires_review BOOLEAN DEFAULT FALSE,
    
    -- 처리 상태
    is_reviewed BOOLEAN DEFAULT FALSE,
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP NULL,
    review_status ENUM('정상', '의심', '차단') NULL,
    review_note TEXT,
    
    -- 알림
    is_notified BOOLEAN DEFAULT FALSE,
    notified_at TIMESTAMP NULL,
    
    -- 타임스탬프
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    INDEX idx_user_detected (user_id, detected_at),
    INDEX idx_risk (risk_level, is_reviewed),
    INDEX idx_score (anomaly_score)
);
```

**용도**: 이상 거래 탐지 및 관리

**샘플 데이터**:
```json
{
  "anomaly_id": 10,
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "transaction_id": 1234,
  "anomaly_score": 0.8750,
  "anomaly_type": "금액",
  "expected_amount": 150000.00,
  "actual_amount": 500000.00,
  "deviation_ratio": 233.33,
  "z_score": 3.45,
  "anomaly_reasons": [
    {"factor": "amount", "score": 0.9, "description": "평소 지출의 3.3배"},
    {"factor": "time", "score": 0.7, "description": "새벽 시간 거래"}
  ],
  "anomaly_description": "평소 쇼핑 패턴과 다른 새벽 시간 고액 거래 감지",
  "risk_level": "높음",
  "requires_review": true,
  "is_reviewed": false
}
```

---

### 7. 사용자 프로파일

#### 7.1 user_profiles (사용자 소비 프로파일)
```sql
CREATE TABLE user_profiles (
    profile_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) UNIQUE NOT NULL,
    
    -- 기본 통계
    total_transactions INT DEFAULT 0,
    avg_transaction_amount DECIMAL(12, 2),
    std_transaction_amount DECIMAL(12, 2),
    
    -- 선호 카테고리
    favorite_category VARCHAR(20),
    category_ratios JSON,                          -- {"외식": 0.3, "쇼핑": 0.2, ...}
    
    -- 소비 패턴
    spending_level ENUM('낮음', '보통', '높음', '매우높음'),
    spending_consistency DECIMAL(5, 4),            -- 0~1 (일관성)
    
    -- 시간 패턴
    preferred_hours JSON,                          -- [18, 19, 20]
    weekday_ratio DECIMAL(5, 4),
    weekend_ratio DECIMAL(5, 4),
    
    -- 세그먼트
    user_segment VARCHAR(50),                      -- "high_spender_foodie"
    segment_percentile INT,                        -- 0-100
    
    -- 이상치 프로파일
    anomaly_sensitivity DECIMAL(5, 4) DEFAULT 0.8000,
    historical_anomaly_rate DECIMAL(5, 4),
    
    -- 예측 성향
    prediction_accuracy_for_user DECIMAL(5, 4),    -- 이 사용자에 대한 모델 정확도
    last_n_predictions JSON,                       -- 최근 예측 이력
    
    -- 업데이트
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_segment (user_segment),
    INDEX idx_spending_level (spending_level)
);
```

**용도**: ML 입력용 사용자 프로파일 (캐싱)

**샘플 데이터**:
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_transactions": 2152,
  "avg_transaction_amount": 45320.50,
  "std_transaction_amount": 78950.30,
  "favorite_category": "외식",
  "category_ratios": {
    "교통": 0.12,
    "생활": 0.08,
    "쇼핑": 0.15,
    "식료품": 0.25,
    "외식": 0.30,
    "주유": 0.10
  },
  "spending_level": "높음",
  "spending_consistency": 0.7500,
  "preferred_hours": [12, 18, 19],
  "weekday_ratio": 0.65,
  "weekend_ratio": 0.35,
  "user_segment": "high_spender_foodie",
  "segment_percentile": 85
}
```

---

### 8. 모델 메타데이터

#### 8.1 ml_models (모델 버전 관리)
```sql
CREATE TABLE ml_models (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) UNIQUE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type ENUM('xgboost', 'randomforest', 'neural_network') NOT NULL,
    
    -- 파일 정보
    model_file_path VARCHAR(500) NOT NULL,
    model_size_mb DECIMAL(10, 2),
    
    -- 성능 지표
    accuracy DECIMAL(5, 4),
    macro_f1 DECIMAL(5, 4),
    weighted_f1 DECIMAL(5, 4),
    
    -- 피처 정보
    num_features INT NOT NULL,
    feature_list JSON,                             -- ["Amount", "Hour", ...]
    feature_importance JSON,                       -- {"Amount": 0.25, ...}
    
    -- 학습 정보
    training_samples INT,
    training_date DATE,
    training_duration_mins INT,
    
    -- 상태
    is_active BOOLEAN DEFAULT FALSE,
    is_production BOOLEAN DEFAULT FALSE,
    
    -- 하이퍼파라미터
    hyperparameters JSON,
    
    -- 타임스탬프
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP NULL,
    
    UNIQUE KEY unique_name_version (model_name, model_version),
    INDEX idx_active (is_active, is_production)
);
```

**용도**: 모델 버전 관리 및 A/B 테스트

**샘플 데이터**:
```json
{
  "model_id": 3,
  "model_name": "xgboost_enhanced",
  "model_version": "v1.0",
  "model_type": "xgboost",
  "model_file_path": "/models/xgboost_enhanced_v1.0.joblib",
  "accuracy": 0.4852,
  "macro_f1": 0.4206,
  "num_features": 16,
  "feature_list": ["User_교통_Ratio", "Current_Category_encoded", ...],
  "is_active": true,
  "is_production": true,
  "hyperparameters": {
    "max_depth": 10,
    "learning_rate": 0.1,
    "n_estimators": 200
  }
}
```

---

## API 엔드포인트 설계

### 1. 거래 데이터 업로드
```
POST /api/v1/transactions/upload
Content-Type: multipart/form-data

Request:
{
  "file": <CSV 파일>,
  "user_id": "550e8400-..."
}

Response:
{
  "status": "success",
  "uploaded_count": 2152,
  "processed_count": 2150,
  "failed_count": 2,
  "upload_id": "upload_12345"
}
```

### 2. 다음 소비 예측
```
POST /api/v1/predictions/next-category

Request:
{
  "user_id": "550e8400-...",
  "context": {
    "last_transaction_id": 1234,
    "include_probabilities": true
  }
}

Response:
{
  "prediction_id": 100,
  "predicted_category": "외식",
  "confidence": 0.4567,
  "probabilities": {
    "교통": 0.12,
    "생활": 0.08,
    "쇼핑": 0.15,
    "식료품": 0.20,
    "외식": 0.46,
    "주유": 0.09
  },
  "explanation": "최근 식료품 구매 후 저녁 시간에 외식 패턴이 높습니다.",
  "model_version": "xgboost_enhanced_v1.0"
}
```

### 3. 소비 분석 리포트
```
GET /api/v1/reports/spending?user_id={user_id}&period={month}

Response:
{
  "report_id": 50,
  "period": "2024-12",
  "summary": {
    "total_spending": 2500000,
    "total_income": 3000000,
    "net_amount": 500000,
    "transaction_count": 283
  },
  "category_breakdown": [...],
  "insights": [
    {"type": "warning", "message": "외식비 급증"},
    {"type": "positive", "message": "식료품 절약 중"}
  ],
  "anomalies": {
    "count": 3,
    "high_risk_count": 1
  }
}
```

### 4. 광고 추천
```
GET /api/v1/recommendations/ads?user_id={user_id}&limit=5

Response:
{
  "recommendations": [
    {
      "ad_id": "AD_RESTAURANT_001",
      "title": "인기 레스토랑 30% 할인",
      "category": "외식",
      "relevance_score": 0.85,
      "discount_rate": 30,
      "coupon_code": "REST30OFF",
      "valid_until": "2025-01-31",
      "reason": "다음 구매 외식 확률 46%"
    }
  ],
  "total_count": 12
}
```

### 5. 이상치 탐지
```
GET /api/v1/anomalies/detect?user_id={user_id}

Response:
{
  "anomaly_count": 3,
  "high_risk_count":1,
  "anomalies": [
    {
      "transaction_id": 1234,
      "amount": 500000,
      "anomaly_score": 0.8750,
      "risk_level": "높음",
      "reason": "평소 지출의 3.3배",
      "detected_at": "2024-12-02T19:30:00"
    }
  ]
}
```

---

## 데이터 흐름

```
CSV 업로드
    ↓
transactions 테이블 삽입
    ↓
user_profiles 업데이트
    ↓
ML 모델 예측
    ↓
predictions 테이블 저장
    ↓
    ├─→ spending_reports 생성
    ├─→ ad_recommendations 생성
    └─→ anomalies 탐지
```

---

## 인덱스 전략

1. **복합 인덱스**: `(user_id, transaction_date)` - 사용자별 시계열 조회
2. **파티셔닝**: `transactions` 테이블을 월별 파티션
3. **캐싱**: Redis에 user_profiles 캐싱

---

## 보안 고려사항

1. **개인정보**: amount, description 암호화
2. **인증**: JWT 토큰 기반
3. **API 제한**: Rate limiting (100 req/min)
4. **감사 로그**: 모든 API 호출 기록

---

**작성일**: 2025-12-03  
**버전**: 1.0  
**서비스**: FastAPI ML Prediction Service
