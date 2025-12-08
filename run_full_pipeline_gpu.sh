#!/bin/bash

# 전체 머신러닝 파이프라인 실행 스크립트
# 전처리 → 필터링 → SMOTE → 모델 비교 → Top 3 분석

echo "======================================================================"
echo "전체 머신러닝 파이프라인 실행"
echo "======================================================================"

# 환경 확인
echo -e "\n[단계 0] GPU 환경 확인"
python3 01_src/utils/gpu_check.py
if [ $? -ne 0 ]; then
    echo "❌ GPU 사용 불가. CPU 모드로 진행하거나 GPU 설정을 확인하세요."
    exit 1
fi

# 1. 전처리 (이미 완료된 경우 건너뛰기)
if [ ! -f "02_data/01_processed/preprocessed_enhanced.csv" ]; then
    echo -e "\n[단계 1] 전처리 실행"
    python3 01_src/00_preprocessing/03_preprocess_enhanced.py
    if [ $? -ne 0 ]; then
        echo "❌ 전처리 실패"
        exit 1
    fi
else
    echo -e "\n[단계 1] 전처리 완료 (건너뛰기)"
fi

# 2. 활동성 필터링
echo -e "\n[단계 2] 활동성 필터링 (월 10건 이상, 5개월 이상)"
python3 01_src/00_preprocessing/08_filter_active_monthly.py
if [ $? -ne 0 ]; then
    echo "⚠️ 필터링 실패. 원본 데이터 사용"
    DATA_FILE="02_data/01_processed/preprocessed_enhanced.csv"
else
    DATA_FILE="02_data/01_processed/preprocessed_filtered_monthly.csv"
fi

# 3. SMOTE 증강
echo -e "\n[단계 3] SMOTE 데이터 증강"
python3 01_src/00_preprocessing/09_apply_smote.py
if [ $? -ne 0 ]; then
    echo "⚠️ SMOTE 실패. 원본 데이터 사용"
else
    echo "✅ SMOTE 증강 완료"
fi

# 4. GPU 모델 비교
echo -e "\n[단계 4] GPU 모델 학습 및 비교"
python3 01_src/01_training/10_compare_gpu_models.py
if [ $? -ne 0 ]; then
    echo "❌ 모델 학습 실패"
    exit 1
fi

# 5. 결과 확인
echo -e "\n======================================================================"
echo "파이프라인 실행 완료"
echo "======================================================================"
echo ""
echo "결과 파일:"
echo "  - 필터링 데이터: $DATA_FILE"
echo "  - SMOTE 증강: 02_data/01_processed/preprocessed_smote_augmented.csv"
echo "  - 모델 비교: 03_models/comparison/gpu_models_comparison.csv"
echo ""
echo "다음 단계: Top 3 모델 상세 분석"
echo "  python3 01_src/01_training/11_analyze_top3.py"
