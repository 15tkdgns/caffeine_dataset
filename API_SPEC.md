# 📚 FastAPI 완전 명세서 (테이블별 API 구성)

**프로젝트**: AI 기반 소비 예측 및 이상 거래 탐지 시스템  
**버전**: v1.0  
**Base URL**: `https://api.example.com/v1`  
**총 엔드포인트**: 85개  
**작성일**: 2025-12-03

---

## 📋 목차
1. [공통 사항](#1-공통-사항)
2. [인증 (Auth)](#2-인증-auth) - 5개 API
3. [사용자 (Users)](#3-사용자-users) - 7개 API
4. [사용자 프로필 (UserProfiles)](#4-사용자-프로필-userprofiles) - 5개 API
5. [거래 내역 (Transactions)](#5-거래-내역-transactions) - 8개 API
6. [예측 결과 (Predictions)](#6-예측-결과-predictions) - 7개 API
7. [이상 거래 (Anomalies)](#7-이상-거래-anomalies) - 7개 API
8. [예측 요청 로그 (PredictionRequests)](#8-예측-요청-로그-predictionrequests) - 5개 API
9. [모델 버전 (ModelVersions)](#9-모델-버전-modelversions) - 7개 API
10. [쿠폰 (Coupons)](#10-쿠폰-coupons) - 7개 API
11. [광고 (Ads)](#11-광고-ads) - 6개 API
12. [AI 리포트 (AiReports)](#12-ai-리포트-aireports) - 6개 API
13. [알림 (Notifications)](#13-알림-notifications) - 7개 API
14. [관리자 통계 (AdminStats)](#14-관리자-통계-adminstats) - 4개 API
15. [관리자 로그 (AdminLogs)](#15-관리자-로그-adminlogs) - 4개 API
16. [시스템 설정 (SystemConfigs)](#16-시스템-설정-systemconfigs) - 5개 API
17. [세션 (Sessions)](#17-세션-sessions) - 4개 API
18. [카테고리 (Categories)](#18-카테고리-categories) - 6개 API

---

## 1️⃣ 공통 사항

### 응답 구조
```json
{
  "success": true,
  "data": { ... },
  "error": { "code": "...", "message": "...", "details": {...} }
}
```

### 페이지네이션
```json
{
  "items": [...],
  "pagination": {
    "total_count": 100,
    "total_pages": 5,
    "current_page": 1,
    "per_page": 20
  }
}
```

### 인증
- **JWT Bearer Token**: `Authorization: Bearer {token}`
- **Access Token**: 1시간 유효
- **Refresh Token**: 30일 유효

---

## 2️⃣ 인증 (Auth) - 5개 API

### 2.1. 회원가입
**POST** `/auth/register`
```python
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=8)
```
**Response 201**: `{ "success": true, "data": { "id": "...", "email": "...", ...} }`

### 2.2. 로그인
**POST** `/auth/login`
```python
class LoginRequest(BaseModel):
    email: EmailStr
    password: str
```
**Response 200**: `{ "access_token": "...", "refresh_token": "...", "user": {...} }`

### 2.3. 토큰 갱신
**POST** `/auth/refresh`
```python
class RefreshRequest(BaseModel):
    refresh_token: str
```
**Response 200**: `{ "access_token": "...", "expires_in": 3600 }`

### 2.4. 로그아웃
**POST** `/auth/logout`
**Response 200**: `{ "success": true, "message": "로그아웃되었습니다." }`

### 2.5. 비밀번호 재설정 요청
**POST** `/auth/reset-password`
```python
class ResetPasswordRequest(BaseModel):
    email: EmailStr
```
**Response 200**: `{ "success": true, "message": "비밀번호 재설정 이메일이 발송되었습니다." }`

---

## 3️⃣ 사용자 (Users) - 7개 API

### 3.1. 사용자 목록 조회 (관리자 전용)
**GET** `/users?page=1&limit=20&role=user&is_active=true&search=홍길동`
**Response 200**: 페이지네이션된 사용자 리스트

### 3.2. 특정 사용자 조회 (관리자 전용)
**GET** `/users/{user_id}`
**Response 200**: 사용자 상세 정보 + 프로필

### 3.3. 내 정보 조회
**GET** `/users/me`
**Response 200**: 현재 로그인 사용자 정보 + 프로필

### 3.4. 내 정보 수정
**PUT** `/users/me`
```python
class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
```
**Response 200**: 업데이트된 사용자 정보

### 3.5. 비밀번호 변경
**PUT** `/users/me/password`
```python
class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)
```
**Response 200**: `{ "success": true, "message": "비밀번호가 변경되었습니다." }`

### 3.6. 사용자 계정 비활성화
**DELETE** `/users/me`
**Response 200**: `{ "success": true, "message": "계정이 비활성화되었습니다." }`

### 3.7. 사용자 활동 요약
**GET** `/users/me/summary`
**Response 200**: 거래 횟수, 총 소비액, 가입일, 최근 활동 등

---

## 4️⃣ 사용자 프로필 (UserProfiles) - 5개 API

### 4.1. 내 프로필 조회
**GET** `/profiles/me`
**Response 200**: UserProfile 상세 정보

### 4.2. 내 프로필 업데이트 (시스템 자동)
**POST** `/profiles/me/refresh`
**Response 200**: 갱신된 프로필 통계

### 4.3. 카테고리별 소비 비율
**GET** `/profiles/me/category-ratios`
**Response 200**: `{ "교통": 0.12, "외식": 0.25, ... }`

### 4.4. 월별 평균 소비액 추이
**GET** `/profiles/me/spending-trend?months=6`
**Response 200**: 최근 N개월 월평균 소비액 배열

### 4.5. 위험 점수 상세
**GET** `/profiles/me/risk-details`
**Response 200**: 위험 점수 계산 세부 내역

---

## 5️⃣ 거래 내역 (Transactions) - 8개 API

### 5.1. 거래 목록 조회 (페이지네이션)
**GET** `/transactions?page=1&limit=20&category=외식&start_date=2025-01-01&end_date=2025-12-31&min_amount=10000&max_amount=100000`
**Response 200**: 거래 리스트 + 페이지네이션

### 5.2. 특정 거래 조회
**GET** `/transactions/{transaction_id}`
**Response 200**: 거래 상세 정보

### 5.3. 거래 등록
**POST** `/transactions`
```python
class TransactionCreate(BaseModel):
    merchant: str
    amount: float
    category: str
    transaction_date: datetime
    payment_method: str
    note: Optional[str] = None
```
**Response 201**: 생성된 거래 객체

### 5.4. 거래 수정
**PUT** `/transactions/{transaction_id}`
```python
class TransactionUpdate(BaseModel):
    merchant: Optional[str] = None
    category: Optional[str] = None
    note: Optional[str] = None
```
**Response 200**: 수정된 거래 객체

### 5.5. 거래 삭제
**DELETE** `/transactions/{transaction_id}`
**Response 200**: `{ "success": true, "message": "거래가 삭제되었습니다." }`

### 5.6. 거래 통계 (기간별)
**GET** `/transactions/stats?period=month&start_date=2025-01-01&end_date=2025-12-31`
**Response 200**: 총 거래 수, 총 금액, 평균 금액, 카테고리별 집계

### 5.7. 월별 소비 추이
**GET** `/transactions/monthly-trend?months=12`
**Response 200**: 최근 N개월 월별 소비 금액 배열

### 5.8. 가맹점별 소비 Top 10
**GET** `/transactions/top-merchants?limit=10`
**Response 200**: 가맹점별 거래 횟수 및 금액 순위

---

## 6️⃣ 예측 결과 (Predictions) - 7개 API

### 6.1. 다음 카테고리 예측 (실시간)
**POST** `/predictions/next-category`
```python
class NextCategoryRequest(BaseModel):
    current_category: str
    amount: float
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    time_since_last: int
```
**Response 200**: 예측 결과 + 확률 분포

### 6.2. 예측 이력 목록
**GET** `/predictions?page=1&limit=20&is_correct=true`
**Response 200**: 예측 리스트 + 페이지네이션

### 6.3. 특정 예측 조회
**GET** `/predictions/{prediction_id}`
**Response 200**: 예측 상세 정보

### 6.4. 예측 검증 (실제 값 업데이트)
**PUT** `/predictions/{prediction_id}/verify`
```python
class PredictionVerify(BaseModel):
    actual_category: str
```
**Response 200**: 업데이트된 예측 + `is_correct` 계산

### 6.5. 예측 정확도 통계
**GET** `/predictions/accuracy-stats?model_version=v1.0&start_date=2025-01-01`
**Response 200**: 전체 정확도, 카테고리별 정확도

### 6.6. 카테고리별 예측 분포
**GET** `/predictions/category-distribution`
**Response 200**: 각 카테고리로 예측된 횟수 통계

### 6.7. 모델별 예측 성능 비교
**GET** `/predictions/model-comparison`
**Response 200**: 모델 버전별 정확도 비교

---

## 7️⃣ 이상 거래 (Anomalies) - 7개 API

### 7.1. 이상 거래 목록
**GET** `/anomalies?page=1&limit=20&status=pending&risk_level=위험&start_date=2025-01-01`
**Response 200**: 이상 거래 리스트 + 페이지네이션

### 7.2. 특정 이상 거래 조회
**GET** `/anomalies/{anomaly_id}`
**Response 200**: 이상 거래 상세 정보

### 7.3. 이상 거래 등록 (시스템 자동)
**POST** `/anomalies`
```python
class AnomalyCreate(BaseModel):
    user_id: str
    merchant: str
    amount: float
    category: str
    transaction_date: datetime
    risk_level: str
    reason: str
```
**Response 201**: 생성된 이상 거래 객체

### 7.4. 이상 거래 상태 변경 (승인/거부)
**PUT** `/anomalies/{anomaly_id}/status`
```python
class AnomalyStatusUpdate(BaseModel):
    status: str = Field(regex="^(approved|rejected)$")
    note: Optional[str] = None
```
**Response 200**: 업데이트된 이상 거래

### 7.5. 이상 거래 삭제
**DELETE** `/anomalies/{anomaly_id}`
**Response 200**: `{ "success": true }`

### 7.6. 이상 거래 통계
**GET** `/anomalies/stats?period=month`
**Response 200**: 위험도별 건수, 처리 상태별 건수

### 7.7. 위험도별 분포
**GET** `/anomalies/risk-distribution`
**Response 200**: 위험/경고/주의별 건수 및 비율

---

## 8️⃣ 예측 요청 로그 (PredictionRequests) - 5개 API

### 8.1. 요청 로그 목록
**GET** `/prediction-requests?page=1&limit=20&status=success&request_type=next_category`
**Response 200**: 요청 로그 리스트

### 8.2. 특정 요청 조회
**GET** `/prediction-requests/{request_id}`
**Response 200**: 요청 상세 정보

### 8.3. 요청 실패율 통계
**GET** `/prediction-requests/failure-rate?period=week`
**Response 200**: 성공/실패 건수 및 비율

### 8.4. 평균 응답 시간
**GET** `/prediction-requests/avg-response-time?period=day`
**Response 200**: 평균 응답 시간 (ms)

### 8.5. 사용자별 요청 횟수
**GET** `/prediction-requests/user-stats?limit=10`
**Response 200**: Top N 사용자 요청 통계

---

## 9️⃣ 모델 버전 (ModelVersions) - 7개 API

### 9.1. 모델 목록 조회
**GET** `/models?is_active=true`
**Response 200**: 모델 버전 리스트

### 9.2. 특정 모델 조회
**GET** `/models/{model_id}`
**Response 200**: 모델 상세 정보

### 9.3. 현재 활성 모델 조회
**GET** `/models/active`
**Response 200**: 현재 프로덕션 모델 정보

### 9.4. 모델 배포 (관리자 전용)
**POST** `/models/deploy`
```python
class ModelDeploy(BaseModel):
    version: str
    model_type: str
    file_path: str
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
```
**Response 201**: 배포된 모델 정보

### 9.5. 모델 비활성화
**PUT** `/models/{model_id}/deactivate`
**Response 200**: `{ "is_active": false }`

### 9.6. 모델 성능 비교
**GET** `/models/compare?version1=v1.0&version2=v1.1`
**Response 200**: 두 모델의 정확도, F1 스코어 비교

### 9.7. 모델 삭제 (관리자 전용)
**DELETE** `/models/{model_id}`
**Response 200**: `{ "success": true }`

---

## 🔟 쿠폰 (Coupons) - 7개 API

### 10.1. 쿠폰 목록 조회
**GET** `/coupons?page=1&limit=20&status=available&category=외식`
**Response 200**: 쿠폰 리스트

### 10.2. 특정 쿠폰 조회
**GET** `/coupons/{coupon_id}`
**Response 200**: 쿠폰 상세 정보

### 10.3. 쿠폰 발급 (시스템/관리자)
**POST** `/coupons`
```python
class CouponCreate(BaseModel):
    user_id: str
    merchant: str
    category: str
    discount_amount: float
    min_purchase: float
    expires_at: datetime
```
**Response 201**: 생성된 쿠폰

### 10.4. 쿠폰 사용
**POST** `/coupons/{coupon_id}/use`
**Response 200**: `{ "status": "used", "used_at": "..." }`

### 10.5. 쿠폰 취소 (사용 전)
**DELETE** `/coupons/{coupon_id}`
**Response 200**: `{ "success": true }`

### 10.6. 만료된 쿠폰 목록
**GET** `/coupons/expired`
**Response 200**: 만료된 쿠폰 리스트

### 10.7. 사용 가능 쿠폰 통계
**GET** `/coupons/stats`
**Response 200**: 카테고리별 쿠폰 개수 및 총 할인 금액

---

## 1️⃣1️⃣ 광고 (Ads) - 6개 API

### 11.1. 광고 목록 조회
**GET** `/ads?is_active=true&target_category=외식`
**Response 200**: 광고 리스트

### 11.2. 특정 광고 조회
**GET** `/ads/{ad_id}`
**Response 200**: 광고 상세 정보

### 11.3. 광고 등록 (관리자 전용)
**POST** `/ads`
```python
class AdCreate(BaseModel):
    title: str
    image_url: str
    target_category: Optional[str] = None
    start_date: datetime
    end_date: datetime
```
**Response 201**: 생성된 광고

### 11.4. 광고 수정
**PUT** `/ads/{ad_id}`
**Response 200**: 수정된 광고

### 11.5. 광고 삭제
**DELETE** `/ads/{ad_id}`
**Response 200**: `{ "success": true }`

### 11.6. 광고 클릭 추적
**POST** `/ads/{ad_id}/click`
**Response 200**: `{ "success": true, "message": "클릭이 기록되었습니다." }`

---

## 1️⃣2️⃣ AI 리포트 (AiReports) - 6개 API

### 12.1. 리포트 목록 조회
**GET** `/ai-reports?type=monthly&page=1&limit=10`
**Response 200**: 리포트 리스트

### 12.2. 특정 리포트 조회
**GET** `/ai-reports/{report_id}`
**Response 200**: 리포트 상세 (content 포함)

### 12.3. 리포트 생성 요청
**POST** `/ai-reports/generate`
```python
class ReportGenerateRequest(BaseModel):
    report_type: str = Field(regex="^(daily|weekly|monthly)$")
    user_id: Optional[str] = None  # 관리자용
```
**Response 201**: 생성된 리포트

### 12.4. 리포트 삭제
**DELETE** `/ai-reports/{report_id}`
**Response 200**: `{ "success": true }`

### 12.5. 최신 리포트 조회
**GET** `/ai-reports/latest?type=monthly`
**Response 200**: 가장 최근 리포트

### 12.6. 리포트 다운로드 (PDF/HTML)
**GET** `/ai-reports/{report_id}/download?format=pdf`
**Response 200**: PDF 또는 HTML 파일

---

## 1️⃣3️⃣ 알림 (Notifications) - 7개 API

### 13.1. 알림 목록 조회
**GET** `/notifications?page=1&limit=20&is_read=false&type=anomaly`
**Response 200**: 알림 리스트

### 13.2. 특정 알림 조회
**GET** `/notifications/{notification_id}`
**Response 200**: 알림 상세

### 13.3. 알림 생성 (시스템)
**POST** `/notifications`
```python
class NotificationCreate(BaseModel):
    user_id: str
    type: str
    title: str
    message: str
```
**Response 201**: 생성된 알림

### 13.4. 알림 읽음 처리
**PUT** `/notifications/{notification_id}/read`
**Response 200**: `{ "is_read": true }`

### 13.5. 모든 알림 읽음 처리
**PUT** `/notifications/read-all`
**Response 200**: `{ "success": true, "message": "모든 알림이 읽음 처리되었습니다." }`

### 13.6. 알림 삭제
**DELETE** `/notifications/{notification_id}`
**Response 200**: `{ "success": true }`

### 13.7. 읽지 않은 알림 개수
**GET** `/notifications/unread-count`
**Response 200**: `{ "unread_count": 3 }`

---

## 1️⃣4️⃣ 관리자 통계 (AdminStats) - 4개 API

### 14.1. 통계 목록 조회
**GET** `/admin/stats?start_date=2025-01-01&end_date=2025-12-31&metric_type=daily_active_users`
**Response 200**: 통계 리스트

### 14.2. 특정 날짜 통계 조회
**GET** `/admin/stats/{date}?metric_type=total_transactions`
**Response 200**: 해당 날짜의 통계

### 14.3. 통계 생성 (배치 작업)
**POST** `/admin/stats`
```python
class StatsCreate(BaseModel):
    stat_date: date
    metric_type: str
    value: float
    details: Optional[dict] = None
```
**Response 201**: 생성된 통계

### 14.4. 통계 집계 요약
**GET** `/admin/stats/summary?period=month`
**Response 200**: 기간별 주요 지표 요약

---

## 1️⃣5️⃣ 관리자 로그 (AdminLogs) - 4개 API

### 15.1. 로그 목록 조회
**GET** `/admin/logs?page=1&limit=50&action_type=model_deploy&admin_id=xxx`
**Response 200**: 로그 리스트

### 15.2. 특정 로그 조회
**GET** `/admin/logs/{log_id}`
**Response 200**: 로그 상세

### 15.3. 로그 생성 (시스템 자동)
**POST** `/admin/logs`
```python
class AdminLogCreate(BaseModel):
    admin_id: str
    action_type: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    description: str
    ip_address: Optional[str] = None
```
**Response 201**: 생성된 로그

### 15.4. 관리자별 활동 통계
**GET** `/admin/logs/stats-by-admin?start_date=2025-01-01`
**Response 200**: 관리자별 작업 횟수

---

## 1️⃣6️⃣ 시스템 설정 (SystemConfigs) - 5개 API

### 16.1. 설정 목록 조회
**GET** `/configs`
**Response 200**: 전체 설정 리스트

### 16.2. 특정 설정 조회
**GET** `/configs/{key}`
**Response 200**: 설정 값

### 16.3. 설정 생성 (관리자 전용)
**POST** `/configs`
```python
class ConfigCreate(BaseModel):
    key: str
    value: str
    data_type: str
    description: Optional[str] = None
    is_editable: bool = True
```
**Response 201**: 생성된 설정

### 16.4. 설정 값 수정
**PUT** `/configs/{key}`
```python
class ConfigUpdate(BaseModel):
    value: str
```
**Response 200**: 수정된 설정

### 16.5. 설정 삭제
**DELETE** `/configs/{key}`
**Response 200**: `{ "success": true }`

---

## 1️⃣7️⃣ 세션 (Sessions) - 4개 API

### 17.1. 세션 목록 조회 (관리자 전용)
**GET** `/sessions?user_id=xxx&is_expired=false`
**Response 200**: 세션 리스트

### 17.2. 특정 세션 조회
**GET** `/sessions/{session_id}`
**Response 200**: 세션 상세

### 17.3. 세션 생성 (로그인 시 자동)
**POST** `/sessions`
```python
class SessionCreate(BaseModel):
    user_id: str
    refresh_token: str
    device_info: Optional[str] = None
    ip_address: Optional[str] = None
    expires_at: datetime
```
**Response 201**: 생성된 세션

### 17.4. 세션 삭제 (로그아웃)
**DELETE** `/sessions/{session_id}`
**Response 200**: `{ "success": true }`

---

## 1️⃣8️⃣ 카테고리 (Categories) - 6개 API

### 18.1. 카테고리 목록 조회
**GET** `/categories?is_active=true`
**Response 200**: 카테고리 리스트

### 18.2. 특정 카테고리 조회
**GET** `/categories/{category_id}`
**Response 200**: 카테고리 상세

### 18.3. 카테고리 생성 (관리자 전용)
**POST** `/categories`
```python
class CategoryCreate(BaseModel):
    code: str
    name: str
    description: Optional[str] = None
    color_hex: Optional[str] = None
    icon: Optional[str] = None
```
**Response 201**: 생성된 카테고리

### 18.4. 카테고리 수정
**PUT** `/categories/{category_id}`
**Response 200**: 수정된 카테고리

### 18.5. 카테고리 삭제
**DELETE** `/categories/{category_id}`
**Response 200**: `{ "success": true }`

### 18.6. 카테고리별 거래 통계
**GET** `/categories/{category_id}/transaction-stats?period=month`
**Response 200**: 해당 카테고리의 거래 건수 및 금액

---

## 📊 API 요약

| 섹션 | 엔드포인트 수 |
|------|--------------|
| 인증 | 5 |
| 사용자 | 7 |
| 사용자 프로필 | 5 |
| 거래 내역 | 8 |
| 예측 결과 | 7 |
| 이상 거래 | 7 |
| 예측 요청 로그 | 5 |
| 모델 버전 | 7 |
| 쿠폰 | 7 |
| 광고 | 6 |
| AI 리포트 | 6 |
| 알림 | 7 |
| 관리자 통계 | 4 |
| 관리자 로그 | 4 |
| 시스템 설정 | 5 |
| 세션 | 4 |
| 카테고리 | 6 |
| **총합** | **85개** |

---

## 🔴 에러 코드

| HTTP | 코드 | 설명 |
|------|------|------|
| 400 | INVALID_REQUEST | 파라미터 오류 |
| 401 | UNAUTHORIZED | 인증 실패 |
| 403 | FORBIDDEN | 권한 부족 |
| 404 | NOT_FOUND | 리소스 없음 |
| 409 | CONFLICT | 중복 데이터 |
| 422 | VALIDATION_ERROR | 검증 실패 |
| 429 | RATE_LIMIT_EXCEEDED | 호출 제한 초과 |
| 500 | INTERNAL_SERVER_ERROR | 서버 오류 |

---

## 🔧 Rate Limiting

- **일반 사용자**: 100 req/min
- **관리자**: 1000 req/min
- **예측 API**: 10 req/min (리소스 집약적)

---

## 📌 구현 가이드

```python
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, date

app = FastAPI(title="AI 소비 예측 API", version="1.0")
security = HTTPBearer()

# Dependency for current user
async def get_current_user(token = Depends(security)):
    # JWT 검증 로직
    return user

# Example: Transaction List
@app.get("/transactions")
async def list_transactions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user = Depends(get_current_user)
):
    # DB 조회 로직
    return {"success": True, "data": {...}}
```
