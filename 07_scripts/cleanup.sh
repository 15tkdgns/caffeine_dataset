#!/bin/bash
# 프로젝트 정리 및 용량 확보 스크립트

echo "=============================================================="
echo "프로젝트 정리 및 용량 확보"
echo "=============================================================="
echo ""

# 현재 디스크 사용량
echo "현재 디스크 사용량:"
df -h / | tail -1
echo ""

# 1. 불필요한 전처리 파일 삭제 (sequence만 남기기)
echo "[1/6] 불필요한 전처리 파일 삭제..."
rm -f 02_data/01_processed/preprocessed_full_featured.csv
rm -f 02_data/01_processed/preprocessed_transactions.csv
rm -f 02_data/01_processed/preprocessed_transactions_v2.csv
echo "  ✅ 중복 전처리 파일 삭제"

# 2. Zone.Identifier 파일 삭제 (Windows 메타데이터)
echo "[2/6] Zone.Identifier 파일 삭제..."
find /root/ibm_data2 -name "*:Zone.Identifier" -delete
echo "  ✅ Windows 메타데이터 삭제"

# 3. 불필요한 원본 데이터 삭제
echo "[3/6] 불필요한 원본 데이터 삭제..."
rm -f 02_data/00_raw/User0_credit_card_transactions.csv
rm -f 02_data/00_raw/sd254_*.csv
echo "  ✅ 샘플 데이터 삭제"

# 4. 불필요한 모델 폴더 삭제
echo "[4/6] 사용하지 않는 모델 폴더 삭제..."
rm -rf 03_models/00_nn
rm -rf 03_models/04_rf
rm -rf 03_models/05_gridsearch
echo "  ✅ 사용하지 않는 모델 삭제"

# 5. 로그 파일 정리
echo "[5/6] 오래된 로그 파일 정리..."
find 04_logs -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
echo "  ✅ 오래된 로그 삭제"

# 6. temp 파일 삭제
echo "[6/6] 임시 파일 정리..."
rm -f temp.txt
rm -rf 99_trash
echo "  ✅ 임시 파일 삭제"

echo ""
echo "=============================================================="
echo "정리 완료"
echo "=============================================================="
echo ""

# 정리 후 디스크 사용량
echo "정리 후 디스크 사용량:"
df -h / | tail -1
echo ""

# 주요 폴더 크기
echo "주요 폴더 크기:"
du -sh 02_data/00_raw 02_data/01_processed 03_models
echo ""

echo "✅ 프로젝트 정리 완료!"
echo ""
echo "남은 파일:"
echo "  - 02_data/00_raw/credit_card_transactions-ibm_v2.csv (원본)"
echo "  - 02_data/01_processed/preprocessed_sequence.csv (15개 피처)"
echo "  - 02_data/01_processed/preprocessed_enhanced.csv (27개 피처)"
echo "  - 03_models/02_xgb (XGBoost 모델)"
echo "  - 03_models/06_sequence (시퀀스 모델)"
