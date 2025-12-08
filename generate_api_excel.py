"""
API 명세서를 테이블별 시트로 분리하여 엑셀 생성
총 20개 시트: Overview + 전체 목록 + 16개 테이블 + 섹션 요약 + 에러 코드
"""
import pandas as pd
from datetime import datetime

output_path = "05_docs/API_SPEC.xlsx"

def ep(method, path, desc, req="-", res="-", auth="Yes", rate="100/min", table=""):
    return {
        "Table": table,
        "Method": method,
        "Path": path,
        "Description": desc,
        "Request": req,
        "Response": res,
        "Auth": auth,
        "Rate": rate
    }

# ============================================================
# Overview
# ============================================================
overview_data = {
    "항목": ["프로젝트명", "버전", "Base URL", "총 테이블", "총 엔드포인트", "생성일시"],
    "내용": [
        "AI 기반 소비 예측 및 이상 거래 탐지 시스템",
        "v1.0",
        "https://api.example.com/v1",
        "16개 (Auth 포함 17개 섹션)",
        "85개",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
}
df_overview = pd.DataFrame(overview_data)

# ============================================================
# 전체 API 목록
# ============================================================
all_apis = []

# ============================================================
# 1. Auth (5개) - 별도 섹션
# ============================================================
auth_apis = []
auth_apis.append(ep("POST", "/auth/register", "회원가입", "RegisterRequest", "UserOut (201)", "No", "100/min", "Auth"))
auth_apis.append(ep("POST", "/auth/login", "로그인", "LoginRequest", "TokenResponse (200)", "No", "100/min", "Auth"))
auth_apis.append(ep("POST", "/auth/refresh", "토큰 갱신", "RefreshRequest", "AccessToken (200)", "Yes", "100/min", "Auth"))
auth_apis.append(ep("POST", "/auth/logout", "로그아웃", "-", "Message (200)", "Yes", "100/min", "Auth"))
auth_apis.append(ep("POST", "/auth/reset-password", "비밀번호 재설정 요청", "ResetPasswordRequest", "Message (200)", "No", "100/min", "Auth"))
all_apis.extend(auth_apis)

# ============================================================
# 2. Users 테이블 (7개)
# ============================================================
users_apis = []
users_apis.append(ep("GET", "/users", "사용자 목록 조회 (페이지네이션)", "-", "UserList (200)", "Admin", "1000/min", "Users"))
users_apis.append(ep("GET", "/users/{user_id}", "특정 사용자 조회", "-", "UserDetail (200)", "Admin", "1000/min", "Users"))
users_apis.append(ep("GET", "/users/me", "내 정보 조회", "-", "UserOut + Profile (200)", "Yes", "100/min", "Users"))
users_apis.append(ep("PUT", "/users/me", "내 정보 수정", "UserUpdate", "UserOut (200)", "Yes", "100/min", "Users"))
users_apis.append(ep("PUT", "/users/me/password", "비밀번호 변경", "PasswordChange", "Message (200)", "Yes", "100/min", "Users"))
users_apis.append(ep("DELETE", "/users/me", "계정 비활성화", "-", "Message (200)", "Yes", "100/min", "Users"))
users_apis.append(ep("GET", "/users/me/summary", "사용자 활동 요약", "-", "Summary (200)", "Yes", "100/min", "Users"))
all_apis.extend(users_apis)

# ============================================================
# 3. UserProfiles 테이블 (5개)
# ============================================================
profiles_apis = []
profiles_apis.append(ep("GET", "/profiles/me", "내 프로필 조회", "-", "ProfileOut (200)", "Yes", "100/min", "UserProfiles"))
profiles_apis.append(ep("POST", "/profiles/me/refresh", "프로필 통계 갱신", "-", "ProfileOut (200)", "Yes", "100/min", "UserProfiles"))
profiles_apis.append(ep("GET", "/profiles/me/category-ratios", "카테고리별 소비 비율", "-", "Dict (200)", "Yes", "100/min", "UserProfiles"))
profiles_apis.append(ep("GET", "/profiles/me/spending-trend", "월별 소비 추이", "-", "Array (200)", "Yes", "100/min", "UserProfiles"))
profiles_apis.append(ep("GET", "/profiles/me/risk-details", "위험 점수 상세", "-", "RiskDetail (200)", "Yes", "100/min", "UserProfiles"))
all_apis.extend(profiles_apis)

# ============================================================
# 4. Transactions 테이블 (8개)
# ============================================================
trans_apis = []
trans_apis.append(ep("GET", "/transactions", "거래 목록 조회 (페이지네이션, 필터)", "-", "TransactionList (200)", "Yes", "100/min", "Transactions"))
trans_apis.append(ep("GET", "/transactions/{id}", "특정 거래 조회", "-", "Transaction (200)", "Yes", "100/min", "Transactions"))
trans_apis.append(ep("POST", "/transactions", "거래 등록", "TransactionCreate", "Transaction (201)", "Yes", "100/min", "Transactions"))
trans_apis.append(ep("PUT", "/transactions/{id}", "거래 수정", "TransactionUpdate", "Transaction (200)", "Yes", "100/min", "Transactions"))
trans_apis.append(ep("DELETE", "/transactions/{id}", "거래 삭제", "-", "Message (200)", "Yes", "100/min", "Transactions"))
trans_apis.append(ep("GET", "/transactions/stats", "거래 통계 (기간별)", "-", "Stats (200)", "Yes", "100/min", "Transactions"))
trans_apis.append(ep("GET", "/transactions/monthly-trend", "월별 소비 추이", "-", "Array (200)", "Yes", "100/min", "Transactions"))
trans_apis.append(ep("GET", "/transactions/top-merchants", "가맹점별 Top 10", "-", "MerchantRank (200)", "Yes", "100/min", "Transactions"))
all_apis.extend(trans_apis)

# ============================================================
# 5. Predictions 테이블 (7개)
# ============================================================
pred_apis = []
pred_apis.append(ep("POST", "/predictions/next-category", "다음 카테고리 예측 (실시간)", "NextCategoryRequest", "PredictionOut (200)", "Yes", "10/min", "Predictions"))
pred_apis.append(ep("GET", "/predictions", "예측 이력 목록", "-", "PredictionList (200)", "Yes", "100/min", "Predictions"))
pred_apis.append(ep("GET", "/predictions/{id}", "특정 예측 조회", "-", "Prediction (200)", "Yes", "100/min", "Predictions"))
pred_apis.append(ep("PUT", "/predictions/{id}/verify", "예측 검증 (실제 값 업데이트)", "VerifyRequest", "Prediction (200)", "Yes", "100/min", "Predictions"))
pred_apis.append(ep("GET", "/predictions/accuracy-stats", "예측 정확도 통계", "-", "AccuracyStats (200)", "Yes", "100/min", "Predictions"))
pred_apis.append(ep("GET", "/predictions/category-distribution", "카테고리별 예측 분포", "-", "Distribution (200)", "Yes", "100/min", "Predictions"))
pred_apis.append(ep("GET", "/predictions/model-comparison", "모델별 성능 비교", "-", "Comparison (200)", "Yes", "100/min", "Predictions"))
all_apis.extend(pred_apis)

# ============================================================
# 6. Anomalies 테이블 (7개)
# ============================================================
anom_apis = []
anom_apis.append(ep("GET", "/anomalies", "이상 거래 목록", "-", "AnomalyList (200)", "Yes", "100/min", "Anomalies"))
anom_apis.append(ep("GET", "/anomalies/{id}", "특정 이상 거래 조회", "-", "Anomaly (200)", "Yes", "100/min", "Anomalies"))
anom_apis.append(ep("POST", "/anomalies", "이상 거래 등록 (시스템)", "AnomalyCreate", "Anomaly (201)", "System", "100/min", "Anomalies"))
anom_apis.append(ep("PUT", "/anomalies/{id}/status", "이상 거래 상태 변경", "StatusUpdate", "Anomaly (200)", "Yes", "100/min", "Anomalies"))
anom_apis.append(ep("DELETE", "/anomalies/{id}", "이상 거래 삭제", "-", "Message (200)", "Yes", "100/min", "Anomalies"))
anom_apis.append(ep("GET", "/anomalies/stats", "이상 거래 통계", "-", "Stats (200)", "Yes", "100/min", "Anomalies"))
anom_apis.append(ep("GET", "/anomalies/risk-distribution", "위험도별 분포", "-", "Distribution (200)", "Yes", "100/min", "Anomalies"))
all_apis.extend(anom_apis)

# ============================================================
# 7. PredictionRequests 테이블 (5개)
# ============================================================
preq_apis = []
preq_apis.append(ep("GET", "/prediction-requests", "요청 로그 목록", "-", "RequestList (200)", "Yes", "100/min", "PredictionRequests"))
preq_apis.append(ep("GET", "/prediction-requests/{id}", "특정 요청 조회", "-", "Request (200)", "Yes", "100/min", "PredictionRequests"))
preq_apis.append(ep("GET", "/prediction-requests/failure-rate", "요청 실패율 통계", "-", "FailureRate (200)", "Yes", "100/min", "PredictionRequests"))
preq_apis.append(ep("GET", "/prediction-requests/avg-response-time", "평균 응답 시간", "-", "AvgTime (200)", "Yes", "100/min", "PredictionRequests"))
preq_apis.append(ep("GET", "/prediction-requests/user-stats", "사용자별 요청 통계", "-", "UserStats (200)", "Yes", "100/min", "PredictionRequests"))
all_apis.extend(preq_apis)

# ============================================================
# 8. ModelVersions 테이블 (7개)
# ============================================================
model_apis = []
model_apis.append(ep("GET", "/models", "모델 목록 조회", "-", "ModelList (200)", "Admin", "1000/min", "ModelVersions"))
model_apis.append(ep("GET", "/models/{id}", "특정 모델 조회", "-", "Model (200)", "Admin", "1000/min", "ModelVersions"))
model_apis.append(ep("GET", "/models/active", "현재 활성 모델 조회", "-", "Model (200)", "Yes", "100/min", "ModelVersions"))
model_apis.append(ep("POST", "/models/deploy", "모델 배포", "ModelDeploy", "Model (201)", "Admin", "1000/min", "ModelVersions"))
model_apis.append(ep("PUT", "/models/{id}/deactivate", "모델 비활성화", "-", "Model (200)", "Admin", "1000/min", "ModelVersions"))
model_apis.append(ep("GET", "/models/compare", "모델 성능 비교", "-", "Comparison (200)", "Admin", "1000/min", "ModelVersions"))
model_apis.append(ep("DELETE", "/models/{id}", "모델 삭제", "-", "Message (200)", "Admin", "1000/min", "ModelVersions"))
all_apis.extend(model_apis)

# ============================================================
# 9. Coupons 테이블 (7개)
# ============================================================
coupon_apis = []
coupon_apis.append(ep("GET", "/coupons", "쿠폰 목록 조회", "-", "CouponList (200)", "Yes", "100/min", "Coupons"))
coupon_apis.append(ep("GET", "/coupons/{id}", "특정 쿠폰 조회", "-", "Coupon (200)", "Yes", "100/min", "Coupons"))
coupon_apis.append(ep("POST", "/coupons", "쿠폰 발급", "CouponCreate", "Coupon (201)", "System", "100/min", "Coupons"))
coupon_apis.append(ep("POST", "/coupons/{id}/use", "쿠폰 사용", "-", "Coupon (200)", "Yes", "100/min", "Coupons"))
coupon_apis.append(ep("DELETE", "/coupons/{id}", "쿠폰 취소", "-", "Message (200)", "Yes", "100/min", "Coupons"))
coupon_apis.append(ep("GET", "/coupons/expired", "만료된 쿠폰 목록", "-", "CouponList (200)", "Yes", "100/min", "Coupons"))
coupon_apis.append(ep("GET", "/coupons/stats", "쿠폰 통계", "-", "Stats (200)", "Yes", "100/min", "Coupons"))
all_apis.extend(coupon_apis)

# ============================================================
# 10. Ads 테이블 (6개)
# ============================================================
ads_apis = []
ads_apis.append(ep("GET", "/ads", "광고 목록 조회", "-", "AdList (200)", "Yes", "100/min", "Ads"))
ads_apis.append(ep("GET", "/ads/{id}", "특정 광고 조회", "-", "Ad (200)", "Yes", "100/min", "Ads"))
ads_apis.append(ep("POST", "/ads", "광고 등록", "AdCreate", "Ad (201)", "Admin", "1000/min", "Ads"))
ads_apis.append(ep("PUT", "/ads/{id}", "광고 수정", "AdUpdate", "Ad (200)", "Admin", "1000/min", "Ads"))
ads_apis.append(ep("DELETE", "/ads/{id}", "광고 삭제", "-", "Message (200)", "Admin", "1000/min", "Ads"))
ads_apis.append(ep("POST", "/ads/{id}/click", "광고 클릭 추적", "-", "Message (200)", "Yes", "100/min", "Ads"))
all_apis.extend(ads_apis)

# ============================================================
# 11. AiReports 테이블 (6개)
# ============================================================
report_apis = []
report_apis.append(ep("GET", "/ai-reports", "리포트 목록 조회", "-", "ReportList (200)", "Yes", "100/min", "AiReports"))
report_apis.append(ep("GET", "/ai-reports/{id}", "특정 리포트 조회", "-", "Report (200)", "Yes", "100/min", "AiReports"))
report_apis.append(ep("POST", "/ai-reports/generate", "리포트 생성 요청", "GenerateRequest", "Report (201)", "Yes", "100/min", "AiReports"))
report_apis.append(ep("DELETE", "/ai-reports/{id}", "리포트 삭제", "-", "Message (200)", "Yes", "100/min", "AiReports"))
report_apis.append(ep("GET", "/ai-reports/latest", "최신 리포트 조회", "-", "Report (200)", "Yes", "100/min", "AiReports"))
report_apis.append(ep("GET", "/ai-reports/{id}/download", "리포트 다운로드", "-", "File (200)", "Yes", "100/min", "AiReports"))
all_apis.extend(report_apis)

# ============================================================
# 12. Notifications 테이블 (7개)
# ============================================================
notif_apis = []
notif_apis.append(ep("GET", "/notifications", "알림 목록 조회", "-", "NotificationList (200)", "Yes", "100/min", "Notifications"))
notif_apis.append(ep("GET", "/notifications/{id}", "특정 알림 조회", "-", "Notification (200)", "Yes", "100/min", "Notifications"))
notif_apis.append(ep("POST", "/notifications", "알림 생성", "NotificationCreate", "Notification (201)", "System", "100/min", "Notifications"))
notif_apis.append(ep("PUT", "/notifications/{id}/read", "알림 읽음 처리", "-", "Notification (200)", "Yes", "100/min", "Notifications"))
notif_apis.append(ep("PUT", "/notifications/read-all", "모든 알림 읽음", "-", "Message (200)", "Yes", "100/min", "Notifications"))
notif_apis.append(ep("DELETE", "/notifications/{id}", "알림 삭제", "-", "Message (200)", "Yes", "100/min", "Notifications"))
notif_apis.append(ep("GET", "/notifications/unread-count", "읽지 않은 알림 개수", "-", "Count (200)", "Yes", "100/min", "Notifications"))
all_apis.extend(notif_apis)

# ============================================================
# 13. AdminStats 테이블 (4개)
# ============================================================
stats_apis = []
stats_apis.append(ep("GET", "/admin/stats", "통계 목록 조회", "-", "StatsList (200)", "Admin", "1000/min", "AdminStats"))
stats_apis.append(ep("GET", "/admin/stats/{date}", "특정 날짜 통계", "-", "Stats (200)", "Admin", "1000/min", "AdminStats"))
stats_apis.append(ep("POST", "/admin/stats", "통계 생성 (배치)", "StatsCreate", "Stats (201)", "System", "1000/min", "AdminStats"))
stats_apis.append(ep("GET", "/admin/stats/summary", "통계 집계 요약", "-", "Summary (200)", "Admin", "1000/min", "AdminStats"))
all_apis.extend(stats_apis)

# ============================================================
# 14. AdminLogs 테이블 (4개)
# ============================================================
logs_apis = []
logs_apis.append(ep("GET", "/admin/logs", "로그 목록 조회", "-", "LogList (200)", "Admin", "1000/min", "AdminLogs"))
logs_apis.append(ep("GET", "/admin/logs/{id}", "특정 로그 조회", "-", "Log (200)", "Admin", "1000/min", "AdminLogs"))
logs_apis.append(ep("POST", "/admin/logs", "로그 생성 (자동)", "LogCreate", "Log (201)", "System", "1000/min", "AdminLogs"))
logs_apis.append(ep("GET", "/admin/logs/stats-by-admin", "관리자별 활동 통계", "-", "AdminStats (200)", "Admin", "1000/min", "AdminLogs"))
all_apis.extend(logs_apis)

# ============================================================
# 15. SystemConfigs 테이블 (5개)
# ============================================================
config_apis = []
config_apis.append(ep("GET", "/configs", "설정 목록 조회", "-", "ConfigList (200)", "Admin", "1000/min", "SystemConfigs"))
config_apis.append(ep("GET", "/configs/{key}", "특정 설정 조회", "-", "Config (200)", "Admin", "1000/min", "SystemConfigs"))
config_apis.append(ep("POST", "/configs", "설정 생성", "ConfigCreate", "Config (201)", "Admin", "1000/min", "SystemConfigs"))
config_apis.append(ep("PUT", "/configs/{key}", "설정 수정", "ConfigUpdate", "Config (200)", "Admin", "1000/min", "SystemConfigs"))
config_apis.append(ep("DELETE", "/configs/{key}", "설정 삭제", "-", "Message (200)", "Admin", "1000/min", "SystemConfigs"))
all_apis.extend(config_apis)

# ============================================================
# 16. Sessions 테이블 (4개)
# ============================================================
session_apis = []
session_apis.append(ep("GET", "/sessions", "세션 목록 조회", "-", "SessionList (200)", "Admin", "1000/min", "Sessions"))
session_apis.append(ep("GET", "/sessions/{id}", "특정 세션 조회", "-", "Session (200)", "Yes", "100/min", "Sessions"))
session_apis.append(ep("POST", "/sessions", "세션 생성 (자동)", "SessionCreate", "Session (201)", "System", "100/min", "Sessions"))
session_apis.append(ep("DELETE", "/sessions/{id}", "세션 삭제 (로그아웃)", "-", "Message (200)", "Yes", "100/min", "Sessions"))
all_apis.extend(session_apis)

# ============================================================
# 17. Categories 테이블 (6개)
# ============================================================
cat_apis = []
cat_apis.append(ep("GET", "/categories", "카테고리 목록", "-", "CategoryList (200)", "Yes", "100/min", "Categories"))
cat_apis.append(ep("GET", "/categories/{id}", "특정 카테고리 조회", "-", "Category (200)", "Yes", "100/min", "Categories"))
cat_apis.append(ep("POST", "/categories", "카테고리 생성", "CategoryCreate", "Category (201)", "Admin", "1000/min", "Categories"))
cat_apis.append(ep("PUT", "/categories/{id}", "카테고리 수정", "CategoryUpdate", "Category (200)", "Admin", "1000/min", "Categories"))
cat_apis.append(ep("DELETE", "/categories/{id}", "카테고리 삭제", "-", "Message (200)", "Admin", "1000/min", "Categories"))
cat_apis.append(ep("GET", "/categories/{id}/transaction-stats", "카테고리별 거래 통계", "-", "Stats (200)", "Yes", "100/min", "Categories"))
all_apis.extend(cat_apis)

# ============================================================
# DataFrame 생성
# ============================================================
df_all = pd.DataFrame(all_apis)

# 섹션별 요약
summary_data = {
    "테이블/섹션": [
        "Auth", "Users", "UserProfiles", "Transactions", "Predictions",
        "Anomalies", "PredictionRequests", "ModelVersions", "Coupons", "Ads",
        "AiReports", "Notifications", "AdminStats", "AdminLogs", "SystemConfigs",
        "Sessions", "Categories"
    ],
    "API 개수": [5, 7, 5, 8, 7, 7, 5, 7, 7, 6, 6, 7, 4, 4, 5, 4, 6],
    "주요 기능": [
        "인증, 토큰 관리",
        "사용자 CRUD, 프로필",
        "소비 패턴 통계, 위험 점수",
        "거래 CRUD, 통계, 추이",
        "카테고리 예측, 정확도",
        "이상 거래 탐지, 승인/거부",
        "API 로그, 성능 분석",
        "모델 배포, 버전 관리",
        "쿠폰 발급/사용, 통계",
        "광고 관리, 클릭 추적",
        "AI 리포트 생성/조회",
        "알림 관리, 읽음 처리",
        "대시보드 통계",
        "관리자 활동 로그",
        "시스템 설정 관리",
        "세션 관리",
        "카테고리 마스터"
    ]
}
df_summary = pd.DataFrame(summary_data)

# 에러 코드
errors = {
    "HTTP": [400, 401, 403, 404, 409, 422, 429, 500],
    "Code": [
        "INVALID_REQUEST", "UNAUTHORIZED", "FORBIDDEN", "NOT_FOUND",
        "CONFLICT", "VALIDATION_ERROR", "RATE_LIMIT_EXCEEDED", "INTERNAL_SERVER_ERROR"
    ],
    "설명": [
        "파라미터 형식 오류", "인증 실패/토큰 만료", "권한 부족", "리소스 없음",
        "중복 데이터", "Pydantic 검증 실패", "호출 제한 초과", "서버 오류"
    ]
}
df_errors = pd.DataFrame(errors)

# ============================================================
# 엑셀 생성 - 테이블별 시트
# ============================================================
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # 1. Overview
    df_overview.to_excel(writer, sheet_name="📌 Overview", index=False)
    
    # 2. 전체 API 목록
    df_all.to_excel(writer, sheet_name="📋 All APIs (85)", index=False)
    
    # 3. 섹션별 요약
    df_summary.to_excel(writer, sheet_name="📊 섹션별 요약", index=False)
    
    # 4-20. 테이블별 시트
    pd.DataFrame(auth_apis).to_excel(writer, sheet_name="1️⃣ Auth (5)", index=False)
    pd.DataFrame(users_apis).to_excel(writer, sheet_name="2️⃣ Users (7)", index=False)
    pd.DataFrame(profiles_apis).to_excel(writer, sheet_name="3️⃣ UserProfiles (5)", index=False)
    pd.DataFrame(trans_apis).to_excel(writer, sheet_name="4️⃣ Transactions (8)", index=False)
    pd.DataFrame(pred_apis).to_excel(writer, sheet_name="5️⃣ Predictions (7)", index=False)
    pd.DataFrame(anom_apis).to_excel(writer, sheet_name="6️⃣ Anomalies (7)", index=False)
    pd.DataFrame(preq_apis).to_excel(writer, sheet_name="7️⃣ PredictionRequests (5)", index=False)
    pd.DataFrame(model_apis).to_excel(writer, sheet_name="8️⃣ ModelVersions (7)", index=False)
    pd.DataFrame(coupon_apis).to_excel(writer, sheet_name="9️⃣ Coupons (7)", index=False)
    pd.DataFrame(ads_apis).to_excel(writer, sheet_name="🔟 Ads (6)", index=False)
    pd.DataFrame(report_apis).to_excel(writer, sheet_name="1️⃣1️⃣ AiReports (6)", index=False)
    pd.DataFrame(notif_apis).to_excel(writer, sheet_name="1️⃣2️⃣ Notifications (7)", index=False)
    pd.DataFrame(stats_apis).to_excel(writer, sheet_name="1️⃣3️⃣ AdminStats (4)", index=False)
    pd.DataFrame(logs_apis).to_excel(writer, sheet_name="1️⃣4️⃣ AdminLogs (4)", index=False)
    pd.DataFrame(config_apis).to_excel(writer, sheet_name="1️⃣5️⃣ SystemConfigs (5)", index=False)
    pd.DataFrame(session_apis).to_excel(writer, sheet_name="1️⃣6️⃣ Sessions (4)", index=False)
    pd.DataFrame(cat_apis).to_excel(writer, sheet_name="1️⃣7️⃣ Categories (6)", index=False)
    
    # 21. 에러 코드
    df_errors.to_excel(writer, sheet_name="🔴 Error Codes", index=False)

print(f"✅ API 명세서 엑셀 생성 완료: {output_path}")
print(f"   총 21개 시트:")
print(f"   - Overview, All APIs, 섹션별 요약")
print(f"   - 17개 테이블별 시트 (Auth + 16개 DB 테이블)")
print(f"   - Error Codes")
print(f"   총 {len(all_apis)}개 엔드포인트")
