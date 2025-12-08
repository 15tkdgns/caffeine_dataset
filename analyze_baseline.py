"""
STEP 1: Baseline 분석
- Confusion Matrix
- 클래스별 혼동 패턴
- Feature Importance
- 생활 카테고리 오분류 분석
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import json

print("="*80)
print("📊 STEP 1: Baseline 모델 상세 분석")
print("="*80)

# ============================================================
# 1. 데이터 및 모델 로드
# ============================================================
print("\n[1/5] 데이터 및 모델 로드")

X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

# Baseline LightGBM 로드
model = joblib.load('03_models/production_models/lightgbm_cuda_production_20251205_162340.joblib')
print(f"  ✅ 모델 로드 완료")
print(f"  테스트 데이터: {len(X_test):,}건")

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']

# ============================================================
# 2. Confusion Matrix
# ============================================================
print("\n[2/5] Confusion Matrix 분석")

cm = confusion_matrix(y_test, y_pred)

# 정규화된 confusion matrix (행 기준)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(f"\n  Confusion Matrix (절대값):")
cm_df = pd.DataFrame(cm, index=category_names, columns=category_names)
print(cm_df)

print(f"\n  Confusion Matrix (정규화 - 행 기준):")
cm_norm_df = pd.DataFrame(cm_normalized, index=category_names, columns=category_names)
print(cm_norm_df.round(3))

# 생활 카테고리 분석
생활_idx = 1
print(f"\n  🔍 생활 카테고리 상세 분석:")
print(f"     정답 생활 건수: {cm[생활_idx].sum():,}건")
print(f"     정확히 맞춘 건수: {cm[생활_idx, 생활_idx]:,}건")
print(f"     정확도: {cm[생활_idx, 생활_idx] / cm[생활_idx].sum() * 100:.2f}%")

# 생활이 오분류된 카테고리
print(f"\n  생활 → 다른 카테고리 오분류:")
for i, cat in enumerate(category_names):
    if i != 생활_idx:
        count = cm[생활_idx, i]
        ratio = count / cm[생활_idx].sum() * 100
        print(f"     → {cat}: {count:,}건 ({ratio:.2f}%)")

# 다른 카테고리 → 생활 오분류
print(f"\n  다른 카테고리 → 생활 오분류:")
for i, cat in enumerate(category_names):
    if i != 생활_idx:
        count = cm[i, 생활_idx]
        ratio = count / cm[i].sum() * 100
        if count > 0:
            print(f"     {cat} → 생활: {count:,}건 ({ratio:.2f}%)")

# ============================================================
# 3. 클래스 분포
# ============================================================
print("\n[3/5] 클래스 분포 분석")

unique, counts = np.unique(y_test, return_counts=True)
total = len(y_test)

print(f"\n  테스트 데이터 클래스 분포:")
for cat_id, cat_name, count in zip(unique, category_names, counts):
    ratio = count / total * 100
    print(f"     {cat_name:6s}: {count:,}건 ({ratio:.2f}%)")

# 가장 불균형한 클래스
max_count = counts.max()
min_count = counts.min()
imbalance_ratio = max_count / min_count
print(f"\n  불균형 비율: {imbalance_ratio:.2f}:1 (최대/최소)")

# ============================================================
# 4. Feature Importance
# ============================================================
print("\n[4/5] Feature Importance 분석")

# LightGBM feature importance
importance = model.feature_importances_

# 상위 20개 피처
top_k = 20
top_indices = np.argsort(importance)[::-1][:top_k]

print(f"\n  Top {top_k} 중요 피처:")
for rank, idx in enumerate(top_indices, 1):
    print(f"     {rank:2d}. Feature {idx:2d}: {importance[idx]:.4f}")

# Feature importance 저장
importance_data = {
    'feature_importance': {
        f'feature_{i}': float(imp) 
        for i, imp in enumerate(importance)
    },
    'top_20_features': [int(idx) for idx in top_indices]
}

# ============================================================
# 5. 생활 카테고리 오분류 심층 분석
# ============================================================
print("\n[5/5] 생활 카테고리 오분류 심층 분석")

# 생활로 예측된 샘플 분석
생활_mask = (y_test == 생활_idx)
생활_pred_mask = (y_pred == 생활_idx)

# True Positive (정답)
tp_mask = 생활_mask & 생활_pred_mask
tp_count = tp_mask.sum()

# False Negative (생활인데 다른 것으로 예측)
fn_mask = 생활_mask & ~생활_pred_mask
fn_count = fn_mask.sum()

# False Positive (생활이 아닌데 생활로 예측)
fp_mask = ~생활_mask & 생활_pred_mask
fp_count = fp_mask.sum()

print(f"\n  생활 카테고리 예측 결과:")
print(f"     True Positive:  {tp_count:,}건")
print(f"     False Negative: {fn_count:,}건 (생활을 못 맞춤)")
print(f"     False Positive: {fp_count:,}건 (다른 걸 생활로 오판)")

# FN 분석 (생활을 못 맞춘 경우)
if fn_count > 0:
    fn_predictions = y_pred[fn_mask]
    print(f"\n  생활을 못 맞춘 경우, 어떤 카테고리로 예측했는지:")
    fn_unique, fn_counts = np.unique(fn_predictions, return_counts=True)
    for pred_id, count in sorted(zip(fn_unique, fn_counts), key=lambda x: -x[1]):
        ratio = count / fn_count * 100
        print(f"     → {category_names[pred_id]}: {count:,}건 ({ratio:.2f}%)")

# FP 분석 (생활로 잘못 예측한 경우)
if fp_count > 0:
    fp_true_labels = y_test[fp_mask]
    print(f"\n  생활로 잘못 예측한 경우, 실제로는:")
    fp_unique, fp_counts = np.unique(fp_true_labels, return_counts=True)
    for true_id, count in sorted(zip(fp_unique, fp_counts), key=lambda x: -x[1]):
        ratio = count / fp_count * 100
        print(f"     실제 {category_names[true_id]}: {count:,}건 ({ratio:.2f}%)")

# ============================================================
# 6. 결과 저장
# ============================================================
print("\n[6/6] 결과 저장")

import os
os.makedirs('04_logs/analysis', exist_ok=True)

# Confusion Matrix 저장
analysis_results = {
    'confusion_matrix': {
        'absolute': cm.tolist(),
        'normalized': cm_normalized.tolist()
    },
    'class_distribution': {
        cat: int(count) for cat, count in zip(category_names, counts)
    },
    'living_category_analysis': {
        'true_positive': int(tp_count),
        'false_negative': int(fn_count),
        'false_positive': int(fp_count),
        'accuracy': float(tp_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0 else 0,
        'precision': float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0,
        'recall': float(tp_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0 else 0
    },
    'most_confused_with_living': {},
    'feature_importance': importance_data
}

# 생활과 가장 헷갈리는 클래스 (양방향)
for i, cat in enumerate(category_names):
    if i != 생활_idx:
        # 생활 → cat
        생활_to_cat = int(cm[생활_idx, i])
        # cat → 생활
        cat_to_생활 = int(cm[i, 생활_idx])
        
        if 생활_to_cat > 0 or cat_to_생활 > 0:
            analysis_results['most_confused_with_living'][cat] = {
                'living_to_category': 생활_to_cat,
                'category_to_living': cat_to_생활,
                'total_confusion': 생활_to_cat + cat_to_생활
            }

# JSON 저장
with open('04_logs/analysis/baseline_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

print(f"  ✅ 분석 결과 저장: 04_logs/analysis/baseline_analysis.json")

# ============================================================
# 7. 핵심 발견 요약
# ============================================================
print("\n" + "="*80)
print("🎯 핵심 발견 사항")
print("="*80)

# 생활과 가장 헷갈리는 Top 3 클래스
confused_sorted = sorted(
    analysis_results['most_confused_with_living'].items(),
    key=lambda x: x[1]['total_confusion'],
    reverse=True
)[:3]

print(f"\n✅ 생활 카테고리와 가장 헷갈리는 클래스 (Top 3):")
for rank, (cat, conf) in enumerate(confused_sorted, 1):
    total = conf['total_confusion']
    생활_to = conf['living_to_category']
    to_생활 = conf['category_to_living']
    print(f"   {rank}. {cat}: 총 {total:,}건 혼동")
    print(f"      - 생활 → {cat}: {생활_to:,}건")
    print(f"      - {cat} → 생활: {to_생활:,}건")

print(f"\n✅ Feature Importance Top 5:")
for i in range(min(5, len(top_indices))):
    idx = top_indices[i]
    print(f"   {i+1}. Feature {idx}: {importance[idx]:.4f}")

print(f"\n✅ 클래스 불균형:")
print(f"   최대: {category_names[counts.argmax()]} ({counts.max():,}건)")
print(f"   최소: {category_names[counts.argmin()]} ({counts.min():,}건)")
print(f"   비율: {imbalance_ratio:.2f}:1")

print("\n" + "="*80)
print("✅ Baseline 분석 완료!")
print("="*80)
print(f"\n📦 다음 단계:")
print(f"   STEP 2: Class Weight 적용")
print(f"   STEP 3: SMOTE/ADASYN (생활 카테고리 증강)")
print(f"   STEP 4: Focal Loss 도입")
print("="*80)
