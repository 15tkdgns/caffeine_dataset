#!/bin/bash
# GPU 전체 파이프라인 실행 스크립트
# 전처리 → 그리드 서치 → 최종 모델 학습

set -e  # 오류 발생 시 중단

echo "======================================================================"
echo "GPU 기반 전체 머신러닝 파이프라인"
echo "======================================================================"
echo ""

# 설정
SAMPLE_FRAC=${1:-1.0}  # 첫 번째 인자로 샘플 비율 지정 가능 (기본: 1.0 = 전체)
PYTHON_CMD="./run_gpu.sh"

echo "파이프라인 설정:"
echo "  - 데이터 샘플 비율: ${SAMPLE_FRAC} (1.0 = 전체 데이터)"
echo "  - GPU 환경: gemini_gpu"
echo ""

# 단계 1: 전처리
echo "======================================================================"
echo "[단계 1/3] 전체 데이터 전처리 (피처 엔지니어링)"
echo "======================================================================"
echo ""

python3 01_src/00_preprocessing/01_preprocess_full.py

if [ $? -ne 0 ]; then
    echo "❌ 전처리 실패"
    exit 1
fi

echo ""
echo "✅ 전처리 완료"
echo ""

# 단계 2: 그리드 서치 (GPU)
echo "======================================================================"
echo "[단계 2/3] GPU 그리드 서치 (하이퍼파라미터 최적화)"
echo "======================================================================"
echo ""

${PYTHON_CMD} 01_src/01_training/02_gridsearch_gpu.py

if [ $? -ne 0 ]; then
    echo "❌ 그리드 서치 실패"
    exit 1
fi

echo ""
echo "✅ 그리드 서치 완료"
echo ""

# 단계 3: 최종 모델 학습 (옵션)
echo "======================================================================"
echo "[단계 3/3] 추가 모델 학습 (선택적)"
echo "======================================================================"
echo ""
echo "필요한 경우 다음 명령어로 특정 모델을 다시 학습할 수 있습니다:"
echo ""
echo "  # Neural Network"
echo "  ${PYTHON_CMD} 01_src/01_training/00_train_nn.py --device gpu"
echo ""
echo "  # XGBoost"
echo "  ${PYTHON_CMD} 01_src/01_training/01_train_tree.py --model xgboost --device gpu"
echo ""
echo "  # RandomForest"
echo "  ${PYTHON_CMD} 01_src/01_training/01_train_tree.py --model randomforest --device gpu"
echo ""

# 최종 요약
echo "======================================================================"
echo "파이프라인 완료!"
echo "======================================================================"
echo ""
echo "📊 결과 확인:"
echo "  - 전처리 데이터: 02_data/01_processed/preprocessed_full_featured.csv"
echo "  - 그리드 서치 결과: 03_models/05_gridsearch/"
echo "  - 메타데이터: 02_data/01_processed/preprocessing_metadata.txt"
echo ""
echo "🎯 다음 단계:"
echo "  1. 03_models/05_gridsearch/ 폴더에서 최적 모델 확인"
echo "  2. metadata_*.json 파일에서 최적 하이퍼파라미터 확인"
echo "  3. cv_results_*.csv 파일로 상세 분석"
echo ""
