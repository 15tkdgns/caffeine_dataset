"""
STEP 4: Focal Loss 적용
- 어려운 샘플(생활 카테고리)에 집중
- XGBoost + LightGBM에 Focal Loss 적용
- γ = 2.0 사용
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import time
import joblib
import json
from datetime import datetime

print("="*80)
print("🔥 STEP 4: Focal Loss 적용")
print("="*80)

# ============================================================
# 1. Focal Loss 구현
# ============================================================
print("\n[1/5] Focal Loss 구현")

def focal_loss_lgb(y_true, y_pred, gamma=2.0, alpha=None):
    """
    LightGBM용 Focal Loss
    γ (gamma): focusing parameter (default=2.0)
    """
    # y_pred는 raw score (logits)
    # Softmax 적용
    exp_preds = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    # One-hot encoding
    n_samples = y_true.shape[0]
    n_classes = probs.shape[1]
    y_true_one_hot = np.zeros((n_samples, n_classes))
    y_true_one_hot[np.arange(n_samples), y_true.astype(int)] = 1
    
    # Focal loss gradient
    p_t = np.sum(probs * y_true_one_hot, axis=1, keepdims=True)
    grad = probs - y_true_one_hot
    grad = grad * (1 - p_t) ** (gamma - 1) * (gamma * p_t * np.log(p_t + 1e-15) + p_t - 1)
    
    # Hessian (approximation)
    hess = probs * (1 - probs) * (1 - p_t) ** gamma
    
    return grad.flatten(), hess.flatten()

print(f"  ✅ Focal Loss (γ=2.0) 준비 완료")

# ============================================================
# 2. 데이터 로드
# ============================================================
print("\n[2/5] 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

# STEP 2 전략적 언더샘플링 적용
sampling_ratios = {0: 1.0, 1: 1.0, 2: 0.7, 3: 0.7, 4: 0.7, 5: 1.0}

indices_to_keep = []
for class_id in range(6):
    class_mask = (y_train == class_id)
    class_indices = np.where(class_mask)[0]
    n_samples = len(class_indices)
    n_keep = int(n_samples * sampling_ratios[class_id])
    np.random.seed(42)
    kept_indices = np.random.choice(class_indices, n_keep, replace=False)
    indices_to_keep.extend(kept_indices)

X_train_sampled = X_train[np.array(indices_to_keep)]
y_train_sampled = y_train[np.array(indices_to_keep)]

print(f"  학습 데이터: {len(X_train_sampled):,}건")
print(f"  테스트 데이터: {len(X_test):,}건")

category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']

# ============================================================
# 3. XGBoost with Focal Loss (Custom Objective)
# ============================================================
print("\n[3/5] XGBoost with Focal Loss")

def focal_loss_xgb(preds, dtrain, gamma=2.0):
    """XGBoost용 Focal Loss"""
    labels = dtrain.get_label()
    n_classes = 6
    
    # Reshape predictions
    preds = preds.reshape(len(labels), n_classes)
    
    # Softmax
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    # One-hot
    n_samples = len(labels)
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), labels.astype(int)] = 1
    
    # Gradient
    p_t = np.sum(probs * y_one_hot, axis=1, keepdims=True)
    grad = probs - y_one_hot
    grad = grad * (1 - p_t) ** (gamma - 1) * (gamma * p_t * np.log(p_t + 1e-15) + p_t - 1)
    
    # Hessian
    hess = probs * (1 - probs) * (1 - p_t) ** gamma
    
    return grad.flatten(), hess.flatten()

# 일반 XGBoost (비교용)
print("  [XGBoost 일반] 학습 시작...")
start = time.time()

xgb_normal = xgb.XGBClassifier(
    device='cuda',
    tree_method='hist',
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_normal.fit(X_train_sampled, y_train_sampled)
train_time_xgb_normal = time.time() - start

y_pred_xgb_normal = xgb_normal.predict(X_test)
acc_xgb_normal = accuracy_score(y_test, y_pred_xgb_normal)
f1_xgb_normal = f1_score(y_test, y_pred_xgb_normal, average='macro')
cat_f1_xgb_normal = f1_score(y_test, y_pred_xgb_normal, average=None)

print(f"  ✅ 완료: {train_time_xgb_normal:.2f}초")
print(f"     Accuracy: {acc_xgb_normal:.4f}, Macro F1: {f1_xgb_normal:.4f}, 생활 F1: {cat_f1_xgb_normal[1]:.4f}")

# XGBoost with Focal Loss
print("\n  [XGBoost Focal Loss] 학습 시작...")
start = time.time()

dtrain = xgb.DMatrix(X_train_sampled, label=y_train_sampled)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'device': 'cuda',
    'tree_method': 'hist',
    'max_depth': 10,
    'learning_rate': 0.05,  # Focal Loss는 낮은 LR 권장
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'num_class': 6,
    'seed': 42
}

xgb_focal = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    obj=focal_loss_xgb,
    verbose_eval=False
)

train_time_xgb_focal = time.time() - start

# 예측
y_pred_probs_focal = xgb_focal.predict(dtest)
y_pred_xgb_focal = np.argmax(y_pred_probs_focal, axis=1)

acc_xgb_focal = accuracy_score(y_test, y_pred_xgb_focal)
f1_xgb_focal = f1_score(y_test, y_pred_xgb_focal, average='macro')
cat_f1_xgb_focal = f1_score(y_test, y_pred_xgb_focal, average=None)

print(f"  ✅ 완료: {train_time_xgb_focal:.2f}초")
print(f"     Accuracy: {acc_xgb_focal:.4f}, Macro F1: {f1_xgb_focal:.4f}, 생활 F1: {cat_f1_xgb_focal[1]:.4f}")

# ============================================================
# 4. LightGBM (일반 + Class Weight)
# ============================================================
print("\n[4/5] LightGBM with Enhanced Class Weight")

# Class Weight 계산
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_sampled), y=y_train_sampled)
class_weights[1] *= 2.5  # 생활 카테고리 강화

sample_weights = np.array([class_weights[y] for y in y_train_sampled])

print("  학습 시작...")
start = time.time()

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train_sampled, y_train_sampled, sample_weight=sample_weights)
train_time_lgb = time.time() - start

y_pred_lgb = lgb_model.predict(X_test)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
cat_f1_lgb = f1_score(y_test, y_pred_lgb, average=None)

print(f"  ✅ 완료: {train_time_lgb:.2f}초")
print(f"     Accuracy: {acc_lgb:.4f}, Macro F1: {f1_lgb:.4f}, 생활 F1: {cat_f1_lgb[1]:.4f}")

# ============================================================
# 5. 결과 저장 및 비교
# ============================================================
print("\n[5/5] 결과 저장")

import os
os.makedirs('04_logs/step4_focal_loss', exist_ok=True)

# 최고 성능 모델 선택
best_scores = {
    'xgb_normal': acc_xgb_normal,
    'xgb_focal': acc_xgb_focal,
    'lgb': acc_lgb
}
best_model_name = max(best_scores, key=best_scores.get)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 메타데이터
metadata = {
    'experiment': 'STEP 4: Focal Loss',
    'results': {
        'xgb_normal': {
            'accuracy': float(acc_xgb_normal),
            'macro_f1': float(f1_xgb_normal),
            'living_f1': float(cat_f1_xgb_normal[1]),
            'category_f1': {cat: float(f1) for cat, f1 in zip(category_names, cat_f1_xgb_normal)}
        },
        'xgb_focal_loss': {
            'accuracy': float(acc_xgb_focal),
            'macro_f1': float(f1_xgb_focal),
            'living_f1': float(cat_f1_xgb_focal[1]),
            'category_f1': {cat: float(f1) for cat, f1 in zip(category_names, cat_f1_xgb_focal)}
        },
        'lgb_weighted': {
            'accuracy': float(acc_lgb),
            'macro_f1': float(f1_lgb),
            'living_f1': float(cat_f1_lgb[1]),
            'category_f1': {cat: float(f1) for cat, f1 in zip(category_names, cat_f1_lgb)}
        }
    }
}

with open(f'04_logs/step4_focal_loss/metadata_{timestamp}.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"  ✅ 메타데이터 저장 완료")

# ============================================================
# 최종 비교
# ============================================================
print("\n" + "="*80)
print("🏆 STEP 4 결과 비교")
print("="*80)

print(f"\n{'모델':<35} {'Accuracy':>12} {'Macro F1':>12} {'생활 F1':>12}")
print("-"*75)
print(f"{'Baseline (Original)':<35} {0.4913:>12.4f} {0.4344:>12.4f} {0.0802:>12.4f}")
print(f"{'XGBoost (일반)':<35} {acc_xgb_normal:>12.4f} {f1_xgb_normal:>12.4f} {cat_f1_xgb_normal[1]:>12.4f}")
print(f"{'XGBoost (Focal Loss γ=2)':<35} {acc_xgb_focal:>12.4f} {f1_xgb_focal:>12.4f} {cat_f1_xgb_focal[1]:>12.4f}")
print(f"{'LightGBM (Weight 2.5x)':<35} {acc_lgb:>12.4f} {f1_lgb:>12.4f} {cat_f1_lgb[1]:>12.4f}")
print("-"*75)

# Focal Loss 효과
focal_vs_normal_acc = (acc_xgb_focal - acc_xgb_normal) * 100
focal_vs_normal_f1 = (cat_f1_xgb_focal[1] - cat_f1_xgb_normal[1]) * 100

print(f"\n📊 Focal Loss 효과:")
print(f"  Accuracy:  {focal_vs_normal_acc:+.2f}%p")
print(f"  생활 F1:   {focal_vs_normal_f1:+.2f}%p")

if focal_vs_normal_f1 > 0:
    print(f"  ✅ Focal Loss가 생활 카테고리 개선!")
else:
    print(f"  ⚠️ Focal Loss 효과 제한적")

print("\n" + "="*80)
print("✅ STEP 4 완료!")
print("="*80)
print(f"\n📦 다음 단계: Stacking Ensemble")
print("="*80)
