"""
DB 테이블 명세서를 엑셀 파일로 생성
각 테이블을 별도 시트로 구성
"""

import pandas as pd
from datetime import datetime

# 엑셀 파일 경로
output_file = '05_docs/DB_SCHEMA_SPEC.xlsx'

# Excel Writer 생성
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # ==========================================
    # Sheet 1: 테이블 목록
    # ==========================================
    overview_data = {
        '번호': list(range(1, 17)),
        '테이블명': [
            'Users', 'UserProfiles', 'Transactions', 'Predictions', 
            'Anomalies', 'PredictionRequests', 'ModelVersions',
            'Coupons', 'Ads', 'AiReports', 'Notifications',
            'AdminStats', 'AdminLogs', 'SystemConfigs',
            'Sessions', 'Categories'
        ],
        '한글명': [
            '사용자', '사용자 프로필', '거래 내역', '예측 결과',
            '이상 거래', '예측 API 로그', '모델 버전',
            '쿠폰', '광고', 'AI 리포트', '알림',
            '관리자 통계', '관리자 로그', '시스템 설정',
            '세션', '카테고리 마스터'
        ],
        '분류': [
            'Core', 'Core', 'Core', 'Core',
            'Core', 'Core', 'Core',
            'Marketing', 'Marketing', 'Marketing', 'Marketing',
            'Admin', 'Admin', 'Admin',
            '선택', '선택'
        ],
        '설명': [
            '사용자 계정 및 인증',
            '사용자 소비 패턴 통계 (Feature Store)',
            '신용카드 거래 내역',
            '모델의 다음 거래 예측 결과',
            '이상 거래 탐지 내역',
            '예측 API 호출 로그',
            'ML 모델 메타데이터 및 버전 관리',
            '맞춤형 쿠폰 정보',
            '광고 배너 정보',
            'LLM 생성 소비 가이드',
            '사용자 알림 이력',
            '관리자 대시보드용 통계 요약',
            '관리자 활동 감사 추적',
            '시스템 전역 설정값',
            '사용자 세션/토큰 (JWT 대신)',
            '카테고리 중앙 관리'
        ],
        '필수여부': [
            '필수', '필수', '필수', '필수',
            '필수', '필수', '필수',
            '필수', '필수', '필수', '필수',
            '필수', '필수', '필수',
            '선택', '권장'
        ]
    }
    
    df_overview = pd.DataFrame(overview_data)
    df_overview.to_excel(writer, sheet_name='테이블 목록', index=False)
    
    # ==========================================
    # 각 테이블별 상세 스펙
    # ==========================================
    
    # 1. Users
    users_data = {
        '컬럼명': ['id', 'name', 'email', 'password_hash', 'role', 'is_active', 'created_at', 'updated_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(100)', 'VARCHAR(255)', 'VARCHAR(255)', 'VARCHAR(20)', 'BOOLEAN', 'TIMESTAMP', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],
        'Key': ['PK', '', 'UK', '', '', '', '', ''],
        'Default': ['', '', '', '', "'user'", 'TRUE', 'CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP'],
        '설명': ['사용자 고유 ID (UUID)', '사용자 이름', '이메일 주소 (로그인 ID)', '비밀번호 해시', '권한 (user/admin)', '계정 활성 상태', '생성 일시', '수정 일시']
    }
    pd.DataFrame(users_data).to_excel(writer, sheet_name='Users', index=False)
    
    # 2. UserProfiles
    profiles_data = {
        '컬럼명': ['user_id', 'avg_monthly_spending', 'favorite_category', 'transaction_count', 'last_transaction_date', 'category_ratios', 'risk_score', 'updated_at'],
        '데이터타입': ['VARCHAR(36)', 'DECIMAL(15,2)', 'VARCHAR(50)', 'INTEGER', 'TIMESTAMP', 'JSON', 'DECIMAL(5,2)', 'TIMESTAMP'],
        'Nullable': ['N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N'],
        'Key': ['PK, FK', '', '', '', '', '', '', ''],
        'Default': ['', '', '', '0', '', '', '0', 'CURRENT_TIMESTAMP'],
        '설명': ['사용자 ID', '월 평균 소비액', '가장 많이 소비하는 카테고리', '총 거래 횟수', '마지막 거래 일시', '카테고리별 소비 비율 (JSON)', '이상 거래 위험 점수 (0~100)', '프로필 갱신일']
    }
    pd.DataFrame(profiles_data).to_excel(writer, sheet_name='UserProfiles', index=False)
    
    # 3. Transactions
    trans_data = {
        '컬럼명': ['id', 'user_id', 'merchant', 'amount', 'category', 'transaction_date', 'payment_method', 'note', 'is_anomaly', 'created_at', 'updated_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(100)', 'DECIMAL(15,2)', 'VARCHAR(50)', 'TIMESTAMP', 'VARCHAR(50)', 'TEXT', 'BOOLEAN', 'TIMESTAMP', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y', 'N', 'N'],
        'Key': ['PK', 'FK', '', '', '', '', '', '', '', '', ''],
        'Default': ['', '', '', '', '', '', '', '', 'FALSE', 'CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP'],
        '설명': ['거래 고유 ID', '사용자 ID', '가맹점명', '거래 금액', '소비 카테고리', '거래 일시', '결제 수단', '비고 (사용자 메모)', '이상 거래 여부', '데이터 생성일', '데이터 수정일']
    }
    pd.DataFrame(trans_data).to_excel(writer, sheet_name='Transactions', index=False)
    
    # 4. Predictions
    pred_data = {
        '컬럼명': ['id', 'user_id', 'current_category', 'predicted_category', 'confidence', 'model_version', 'actual_category', 'is_correct', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(50)', 'VARCHAR(50)', 'DECIMAL(5,4)', 'VARCHAR(50)', 'VARCHAR(50)', 'BOOLEAN', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y', 'N'],
        'Key': ['PK', 'FK', '', '', '', 'FK', '', '', ''],
        'Default': ['', '', '', '', '', '', '', '', 'CURRENT_TIMESTAMP'],
        '설명': ['예측 ID', '사용자 ID', '현재(마지막) 거래 카테고리', '예측된 다음 카테고리', '예측 확률 (0~1)', '사용된 모델 버전', '실제 다음 구매 카테고리 (검증용)', '예측 정확도', '예측 일시']
    }
    pd.DataFrame(pred_data).to_excel(writer, sheet_name='Predictions', index=False)
    
    # 5. Anomalies
    anom_data = {
        '컬럼명': ['id', 'user_id', 'user_name', 'merchant', 'amount', 'category', 'transaction_date', 'risk_level', 'reason', 'status', 'created_at', 'updated_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(100)', 'VARCHAR(100)', 'DECIMAL(15,2)', 'VARCHAR(50)', 'TIMESTAMP', 'VARCHAR(20)', 'TEXT', 'VARCHAR(20)', 'TIMESTAMP', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],
        'Key': ['PK', 'FK', '', '', '', '', '', '', '', '', '', ''],
        'Default': ['', '', '', '', '', '', '', '', '', "'pending'", 'CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP'],
        '설명': ['이상거래 ID', '사용자 ID', '사용자 이름 (편의성)', '가맹점명', '거래 금액', '카테고리', '거래 일시', '위험도 (위험/경고/주의)', '탐지 사유', '처리 상태 (pending/approved/rejected)', '탐지 일시', '상태 변경일']
    }
    pd.DataFrame(anom_data).to_excel(writer, sheet_name='Anomalies', index=False)
    
    # 6. PredictionRequests
    preq_data = {
        '컬럼명': ['id', 'user_id', 'request_type', 'input_features', 'prediction_id', 'response_time_ms', 'status', 'error_message', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(20)', 'JSON', 'VARCHAR(36)', 'INTEGER', 'VARCHAR(20)', 'TEXT', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'N'],
        'Key': ['PK', 'FK', '', '', 'FK', '', '', '', ''],
        'Default': ['', '', '', '', '', '', '', '', 'CURRENT_TIMESTAMP'],
        '설명': ['요청 ID', '사용자 ID', '요청 유형 (next_category/anomaly_detection)', '입력 피처 (JSON)', '예측 결과 ID', '응답 시간 (ms)', '요청 상태 (success/failed/timeout)', '에러 메시지', '요청 일시']
    }
    pd.DataFrame(preq_data).to_excel(writer, sheet_name='PredictionRequests', index=False)
    
    # 7. ModelVersions
    model_data = {
        '컬럼명': ['id', 'version', 'model_type', 'file_path', 'accuracy', 'macro_f1', 'is_active', 'deployed_at', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(50)', 'VARCHAR(50)', 'VARCHAR(255)', 'DECIMAL(5,4)', 'DECIMAL(5,4)', 'BOOLEAN', 'TIMESTAMP', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'N'],
        'Key': ['PK', 'UK', '', '', '', '', '', '', ''],
        'Default': ['', '', '', '', '', '', 'FALSE', '', 'CURRENT_TIMESTAMP'],
        '설명': ['모델 ID', '모델 버전 (v1.2.3)', '모델 유형 (xgboost/random_forest)', '모델 파일 경로', '정확도', 'Macro F1 Score', '현재 사용 중', '배포 일시', '생성 일시']
    }
    pd.DataFrame(model_data).to_excel(writer, sheet_name='ModelVersions', index=False)
    
    # 8. Coupons
    coupon_data = {
        '컬럼명': ['id', 'user_id', 'merchant', 'category', 'discount_amount', 'min_purchase', 'expires_at', 'used_at', 'status', 'created_at', 'updated_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(100)', 'VARCHAR(50)', 'DECIMAL(15,2)', 'DECIMAL(15,2)', 'TIMESTAMP', 'TIMESTAMP', 'VARCHAR(20)', 'TIMESTAMP', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'N', 'N', 'N'],
        'Key': ['PK', 'FK', '', '', '', '', '', '', '', '', ''],
        'Default': ['', '', '', '', '', '', '', '', "'available'", 'CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP'],
        '설명': ['쿠폰 ID', '사용자 ID', '사용 가능 가맹점', '적용 카테고리', '할인 금액', '최소 구매 금액', '만료 일시', '사용 일시', '쿠폰 상태 (available/used/expired)', '발급 일시', '상태 변경일']
    }
    pd.DataFrame(coupon_data).to_excel(writer, sheet_name='Coupons', index=False)
    
    # 9. Ads
    ads_data = {
        '컬럼명': ['id', 'title', 'image_url', 'target_category', 'start_date', 'end_date', 'is_active', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(100)', 'VARCHAR(255)', 'VARCHAR(50)', 'TIMESTAMP', 'TIMESTAMP', 'BOOLEAN', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'Y', 'N', 'N', 'N', 'N'],
        'Key': ['PK', '', '', '', '', '', '', ''],
        'Default': ['', '', '', '', '', '', 'TRUE', 'CURRENT_TIMESTAMP'],
        '설명': ['광고 ID', '광고 제목', '이미지 URL', '타겟 카테고리', '게시 시작일', '게시 종료일', '활성 여부', '생성 일시']
    }
    pd.DataFrame(ads_data).to_excel(writer, sheet_name='Ads', index=False)
    
    # 10. AiReports
    report_data = {
        '컬럼명': ['id', 'user_id', 'report_type', 'content', 'summary', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(20)', 'TEXT', 'VARCHAR(255)', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'Y', 'N'],
        'Key': ['PK', 'FK', '', '', '', ''],
        'Default': ['', '', '', '', '', 'CURRENT_TIMESTAMP'],
        '설명': ['리포트 ID', '사용자 ID', '리포트 유형 (daily/weekly/monthly)', '리포트 내용 (LLM 생성 텍스트)', '요약', '생성 일시']
    }
    pd.DataFrame(report_data).to_excel(writer, sheet_name='AiReports', index=False)
    
    # 11. Notifications
    notif_data = {
        '컬럼명': ['id', 'user_id', 'type', 'title', 'message', 'is_read', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(20)', 'VARCHAR(100)', 'TEXT', 'BOOLEAN', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'N', 'N', 'N'],
        'Key': ['PK', 'FK', '', '', '', '', ''],
        'Default': ['', '', '', '', '', 'FALSE', 'CURRENT_TIMESTAMP'],
        '설명': ['알림 ID', '사용자 ID', '알림 유형 (anomaly/coupon/report/system)', '알림 제목', '알림 내용', '읽음 여부', '생성 일시']
    }
    pd.DataFrame(notif_data).to_excel(writer, sheet_name='Notifications', index=False)
    
    # 12. AdminStats
    stats_data = {
        '컬럼명': ['id', 'stat_date', 'metric_type', 'value', 'details', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'DATE', 'VARCHAR(50)', 'DECIMAL(15,2)', 'JSON', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'N', 'Y', 'N'],
        'Key': ['PK', '', '', '', '', ''],
        'Default': ['', '', '', '', '', 'CURRENT_TIMESTAMP'],
        '설명': ['통계 ID', '통계 기준일', '지표 유형 (daily_active_users/total_transactions/anomaly_count)', '지표 값', '상세 정보 (JSON)', '생성 일시']
    }
    pd.DataFrame(stats_data).to_excel(writer, sheet_name='AdminStats', index=False)
    
    # 13. AdminLogs
    log_data = {
        '컬럼명': ['id', 'admin_id', 'action_type', 'resource_type', 'resource_id', 'description', 'ip_address', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(50)', 'VARCHAR(50)', 'VARCHAR(36)', 'TEXT', 'VARCHAR(45)', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'N'],
        'Key': ['PK', 'FK', '', '', '', '', '', ''],
        'Default': ['', '', '', '', '', '', '', 'CURRENT_TIMESTAMP'],
        '설명': ['로그 ID', '관리자 ID', '작업 유형 (model_deploy/user_manage/config_update)', '대상 리소스 (model/user/system)', '대상 리소스 ID', '작업 설명', '접속 IP', '작업 일시']
    }
    pd.DataFrame(log_data).to_excel(writer, sheet_name='AdminLogs', index=False)
    
    # 14. SystemConfigs
    config_data = {
        '컬럼명': ['key', 'value', 'data_type', 'description', 'is_editable', 'updated_by', 'updated_at'],
        '데이터타입': ['VARCHAR(100)', 'TEXT', 'VARCHAR(20)', 'TEXT', 'BOOLEAN', 'VARCHAR(36)', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'Y', 'N', 'Y', 'N'],
        'Key': ['PK', '', '', '', '', 'FK', ''],
        'Default': ['', '', '', '', 'TRUE', '', 'CURRENT_TIMESTAMP'],
        '설명': ['설정 키 (anomaly_threshold/model_version)', '설정 값 (JSON 또는 문자열)', '데이터 타입 (string/number/boolean/json)', '설명', '수정 가능 여부', '수정한 관리자 ID', '수정 일시']
    }
    pd.DataFrame(config_data).to_excel(writer, sheet_name='SystemConfigs', index=False)
    
    # 15. Sessions (선택)
    session_data = {
        '컬럼명': ['id', 'user_id', 'refresh_token', 'device_info', 'ip_address', 'expires_at', 'created_at'],
        '데이터타입': ['VARCHAR(36)', 'VARCHAR(36)', 'VARCHAR(255)', 'VARCHAR(255)', 'VARCHAR(45)', 'TIMESTAMP', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'Y', 'Y', 'N', 'N'],
        'Key': ['PK', 'FK', 'UK', '', '', '', ''],
        'Default': ['', '', '', '', '', '', 'CURRENT_TIMESTAMP'],
        '설명': ['세션 ID', '사용자 ID', 'Refresh Token', '디바이스 정보', '접속 IP', '만료 시간', '생성 일시']
    }
    pd.DataFrame(session_data).to_excel(writer, sheet_name='Sessions (선택)', index=False)
    
    # 16. Categories (선택)
    cat_data = {
        '컬럼명': ['id', 'code', 'name', 'description', 'color_hex', 'icon', 'is_active', 'created_at'],
        '데이터타입': ['INTEGER', 'VARCHAR(20)', 'VARCHAR(50)', 'TEXT', 'VARCHAR(7)', 'VARCHAR(50)', 'BOOLEAN', 'TIMESTAMP'],
        'Nullable': ['N', 'N', 'N', 'Y', 'Y', 'Y', 'N', 'N'],
        'Key': ['PK (AI)', 'UK', '', '', '', '', '', ''],
        'Default': ['AUTO_INCREMENT', '', '', '', '', '', 'TRUE', 'CURRENT_TIMESTAMP'],
        '설명': ['카테고리 ID (자동증가)', '카테고리 코드 (transport/life/shopping 등)', '카테고리 이름 (한글)', '설명', 'UI 표시 색상', '아이콘 이름', '활성 상태', '생성 일시']
    }
    pd.DataFrame(cat_data).to_excel(writer, sheet_name='Categories (선택)', index=False)

print(f"✅ 엑셀 파일 생성 완료: {output_file}")
print(f"   총 17개 시트 (테이블 목록 + 16개 테이블)")
