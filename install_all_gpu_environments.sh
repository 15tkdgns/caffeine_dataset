#!/bin/bash
# Master Installation Script for All GPU Environments
# Installs all 3 environments sequentially

echo "======================================================================"
echo "Master GPU Environment Installation"
echo "Strategy B: 3 Separate Environments"
echo "======================================================================"

START_TIME=$(date +%s)

# 스크립트에 실행 권한 부여
chmod +x install_env1_basic_gpu.sh
chmod +x install_env2_rapids.sh
chmod +x install_env3_lightgbm_cuda.sh

# 환경 1: Basic GPU
echo -e "\n\n"
echo "######################################################################"
echo "# Installing Environment 1: Basic GPU (TensorFlow, XGBoost, CatBoost)"
echo "######################################################################"
bash install_env1_basic_gpu.sh

if [ $? -ne 0 ]; then
    echo "❌ Environment 1 installation failed"
    exit 1
fi

# 환경 2: RAPIDS cuML
echo -e "\n\n"
echo "######################################################################"
echo "# Installing Environment 2: RAPIDS cuML"
echo "######################################################################"
bash install_env2_rapids.sh

if [ $? -ne 0 ]; then
    echo "⚠️ Environment 2 installation failed (non-critical)"
    echo "   Continuing with remaining environments..."
fi

# 환경 3: LightGBM CUDA
echo -e "\n\n"
echo "######################################################################"
echo "# Installing Environment 3: LightGBM CUDA"
echo "######################################################################"
bash install_env3_lightgbm_cuda.sh

if [ $? -ne 0 ]; then
    echo "⚠️ Environment 3 installation failed (non-critical)"
    echo "   Continuing..."
fi

# 최종 요약
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "\n\n"
echo "======================================================================"
echo "Installation Summary"
echo "======================================================================"
echo "Total time: $(($DURATION / 60)) minutes $(($DURATION % 60)) seconds"
echo ""
echo "Created environments:"
echo "  1. gpu_basic      - TensorFlow, XGBoost, CatBoost"
echo "  2. rapids_cuml    - cuML RandomForest"
echo "  3. lightgbm_cuda  - LightGBM CUDA"
echo ""
echo "To activate an environment:"
echo "  conda activate gpu_basic"
echo "  conda activate rapids_cuml"
echo "  conda activate lightgbm_cuda"
echo ""
echo "Next steps:"
echo "  1. Test each environment with: python3 01_src/utils/gpu_check.py"
echo "  2. Run model comparison: bash run_all_environments.sh"
echo "======================================================================"
