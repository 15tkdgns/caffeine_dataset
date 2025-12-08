#!/bin/bash
# Run models across all environments and combine results

echo "======================================================================"
echo "Multi-Environment Model Training"
echo "======================================================================"

OUTPUT_DIR="03_models/multi_env_comparison"
mkdir -p $OUTPUT_DIR

START_TIME=$(date +%s)

# Environment 1: Basic GPU
echo -e "\n######################################################################"
echo "# Environment 1: Basic GPU (TensorFlow, XGBoost, CatBoost)"
echo "######################################################################"

conda run -n gpu_basic python3 01_src/01_training/12_train_by_environment.py

if [ $? -eq 0 ]; then
    echo "✅ Environment 1 complete"
else
    echo "❌ Environment 1 failed"
fi

# Environment 2: RAPIDS
echo -e "\n######################################################################"
echo "# Environment 2: RAPIDS cuML"
echo "######################################################################"

conda run -n rapids_cuml python3 01_src/01_training/12_train_by_environment.py

if [ $? -eq 0 ]; then
    echo "✅ Environment 2 complete"
else
    echo "⚠️ Environment 2 failed (continuing...)"
fi

# Environment 3: LightGBM CUDA
echo -e "\n######################################################################"
echo "# Environment 3: LightGBM CUDA"
echo "######################################################################"

conda run -n lightgbm_cuda python3 01_src/01_training/12_train_by_environment.py

if [ $? -eq 0 ]; then
    echo "✅ Environment 3 complete"
else
    echo "⚠️ Environment 3 failed (continuing...)"
fi

# 결과 통합
echo -e "\n######################################################################"
echo "# Combining Results"
echo "######################################################################"

python3 << 'EOF'
import json
import pandas as pd
import os

output_dir = '03_models/multi_env_comparison'
all_results = {}

# 각 환경 결과 로드
for env_file in ['env1_results.json', 'env2_results.json', 'env3_results.json']:
    filepath = os.path.join(output_dir, env_file)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            results = json.load(f)
            all_results.update(results)
        print(f"✅ Loaded {env_file}")
    else:
        print(f"⚠️ {env_file} not found")

# 통합 결과 저장
combined_file = os.path.join(output_dir, 'combined_results.json')
with open(combined_file, 'w') as f:
    json.dump(all_results, f, indent=2)

# CSV 저장
df = pd.DataFrame(all_results).T
csv_file = os.path.join(output_dir, 'combined_results.csv')
df.to_csv(csv_file)

print(f"\n✅ Combined results saved:")
print(f"  - {combined_file}")
print(f"  - {csv_file}")

# 요약 출력
print("\n" + "="*70)
print("Results Summary")
print("="*70)
print(df[['accuracy', 'macro_f1', 'train_time', 'environment']].to_string())

# Top 3
print("\n" + "="*70)
print("Top 3 Models (by Accuracy)")
print("="*70)
top3 = df.nlargest(3, 'accuracy')
for idx, (name, row) in enumerate(top3.iterrows(), 1):
    print(f"\n{idx}. {name}")
    print(f"   Accuracy: {row['accuracy']:.4f}")
    print(f"   Macro F1: {row['macro_f1']:.4f}")
    print(f"   Train Time: {row['train_time']:.1f}s")
    print(f"   Environment: {row['environment']}")
EOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "\n======================================================================"
echo "Multi-Environment Training Complete!"
echo "======================================================================"
echo "Total time: $(($DURATION / 60))m $(($DURATION % 60))s"
echo ""
echo "Results: $OUTPUT_DIR/combined_results.csv"
echo ""
echo "Next step: Top 3 analysis"
echo "  python3 01_src/01_training/11_analyze_top3.py"
