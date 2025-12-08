"""
STEP 1: Stacking Ensemble (3-Model)
- Base: LightGBM 최적화, XGBoost 최적화, Baseline LightGBM
- Meta-learner: Logistic Regression, LightGBM
- 목표: 50% 달성
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
import joblib
import time

print("="*80)
print("🏗️ STEP 1: Stacking Ensemble (50% 도전)")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/5] 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

# 원본 데이터만 사용 (SMOTE 제거)
original_size = 5139965
X_train_original = X_train[:original_size]
y_train_original = y_train[:original_size]

print(f"  학습: {len(X_train_original):,}건")
print(f"  테스트: {len(X_test):,}건")

# ============================================================
# 2. Base Models 학습
# ============================================================
print("\n[2/5] Base Models 학습")

# Model 1: LightGBM 최적화
print("  [1/3] LightGBM 최적화...")
lgb_model1 = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.05,
    num_leaves=256,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
start = time.time()
lgb_model1.fit(X_train_original, y_train_original)
time1 = time.time() - start
print(f"    ✅ 완료: {time1:.2f}초")

# Model 2: XGBoost 최적화
print("  [2/3] XGBoost 최적화...")
xgb_model1 = xgb.XGBClassifier(
    device='cuda',
    tree_method='hist',
    n_estimators=500,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)
start = time.time()
xgb_model1.fit(X_train_original, y_train_original)
time2 = time.time() - start
print(f"    ✅ 완료: {time2:.2f}초")

# Model 3: LightGBM 다른 설정
print("  [3/3] LightGBM 다른 설정...")
lgb_model2 = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=15,
    learning_rate=0.08,
    num_leaves=200,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=30,
    random_state=123,
    n_jobs=-1,
    verbose=-1
)
start = time.time()
lgb_model2.fit(X_train_original, y_train_original)
time3 = time.time() - start
print(f"    ✅ 완료: {time3:.2f}초")

# Base Models 단독 성능
print("\n  Base Models 단독 성능:")
y_pred_lgb1 = lgb_model1.predict(X_test)
y_pred_xgb1 = xgb_model1.predict(X_test)
y_pred_lgb2 = lgb_model2.predict(X_test)

acc_lgb1 = accuracy_score(y_test, y_pred_lgb1)
acc_xgb1 = accuracy_score(y_test, y_pred_xgb1)
acc_lgb2 = accuracy_score(y_test, y_pred_lgb2)

print(f"    LightGBM-1: {acc_lgb1:.4f} ({acc_lgb1*100:.2f}%)")
print(f"    XGBoost-1:  {acc_xgb1:.4f} ({acc_xgb1*100:.2f}%)")
print(f"    LightGBM-2: {acc_lgb2:.4f} ({acc_lgb2*100:.2f}%)")

# ============================================================
# 3. Meta Features 생성 (Out-of-Fold Predictions)
# ============================================================
print("\n[3/5] Meta Features 생성 (5-Fold CV)")

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold predictions 저장
oof_lgb1 = np.zeros((len(X_train_original), 6))
oof_xgb1 = np.zeros((len(X_train_original), 6))
oof_lgb2 = np.zeros((len(X_train_original), 6))

print("  5-Fold Cross-Validation 진행 중...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_original), 1):
    print(f"    Fold {fold}/5...", end=" ")
    
    X_tr, X_val = X_train_original[train_idx], X_train_original[val_idx]
    y_tr, y_val = y_train_original[train_idx], y_train_original[val_idx]
    
    # LightGBM-1
    lgb_fold = lgb.LGBMClassifier(
        n_estimators=500, max_depth=12, learning_rate=0.05,
        num_leaves=256, subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_fold.fit(X_tr, y_tr)
    oof_lgb1[val_idx] = lgb_fold.predict_proba(X_val)
    
    # XGBoost-1
    xgb_fold = xgb.XGBClassifier(
        device='cuda', tree_method='hist',
        n_estimators=500, max_depth=12, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1
    )
    xgb_fold.fit(X_tr, y_tr)
    oof_xgb1[val_idx] = xgb_fold.predict_proba(X_val)
    
    # LightGBM-2
    lgb2_fold = lgb.LGBMClassifier(
        n_estimators=400, max_depth=15, learning_rate=0.08,
        num_leaves=200, subsample=0.85, colsample_bytree=0.85,
        random_state=123, n_jobs=-1, verbose=-1
    )
    lgb2_fold.fit(X_tr, y_tr)
    oof_lgb2[val_idx] = lgb2_fold.predict_proba(X_val)
    
    print("✓")

# Test set meta features
test_lgb1 = lgb_model1.predict_proba(X_test)
test_xgb1 = xgb_model1.predict_proba(X_test)
test_lgb2 = lgb_model2.predict_proba(X_test)

# Meta features
X_meta_train = np.hstack([oof_lgb1, oof_xgb1, oof_lgb2])
X_meta_test = np.hstack([test_lgb1, test_xgb1, test_lgb2])

print(f"  ✅ Meta Features: {X_meta_train.shape}")

# ============================================================
# 4. Meta-Learner 학습
# ============================================================
print("\n[4/5] Meta-Learner 학습")

# Method 1: Logistic Regression
print("  [1/3] Logistic Regression...")
lr_meta = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_meta.fit(X_meta_train, y_train_original)

y_pred_lr = lr_meta.predict(X_meta_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')
print(f"    Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")

# Method 2: LightGBM Meta
print("  [2/3] LightGBM Meta...")
lgb_meta = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_meta.fit(X_meta_train, y_train_original)

y_pred_lgb_meta = lgb_meta.predict(X_meta_test)
acc_lgb_meta = accuracy_score(y_test, y_pred_lgb_meta)
f1_lgb_meta = f1_score(y_test, y_pred_lgb_meta, average='macro')
print(f"    Accuracy: {acc_lgb_meta:.4f} ({acc_lgb_meta*100:.2f}%)")

# Method 3: Weighted Average (가장 단순)
print("  [3/3] Weighted Average...")
# 개별 성능 기반 가중치
weights = np.array([acc_lgb1, acc_xgb1, acc_lgb2])
weights = weights / weights.sum()

test_avg = weights[0] * test_lgb1 + weights[1] * test_xgb1 + weights[2] * test_lgb2
y_pred_avg = np.argmax(test_avg, axis=1)
acc_avg = accuracy_score(y_test, y_pred_avg)
f1_avg = f1_score(y_test, y_pred_avg, average='macro')
print(f"    Accuracy: {acc_avg:.4f} ({acc_avg*100:.2f}%)")
print(f"    가중치: LGB1={weights[0]:.3f}, XGB={weights[1]:.3f}, LGB2={weights[2]:.3f}")

# ============================================================
# 5. 최종 결과
# ============================================================
print("\n[5/5] 최종 결과")

results = [
    ("Baseline (기존)", 0.4913),
    ("LightGBM-1 (단독)", acc_lgb1),
    ("XGBoost-1 (단독)", acc_xgb1),
    ("LightGBM-2 (단독)", acc_lgb2),
    ("Stacking (LogisticRegression)", acc_lr),
    ("Stacking (LightGBM Meta)", acc_lgb_meta),
    ("Stacking (Weighted Avg)", acc_avg),
]

print("\n" + "="*80)
print("🏆 Stacking Ensemble 결과")
print("="*80)

print(f"\n{'모델':<40} {'Accuracy':>12} {'50% 달성':>12}")
print("-"*68)
for name, acc in results:
    status = "✅" if acc >= 0.50 else "❌"
    print(f"{name:<40} {acc:>12.4f} {status:>12}")
print("-"*68)

# 최고 성능
best_name, best_acc = max(results, key=lambda x: x[1])
print(f"\n🏆 최고 성능: {best_name}")
print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

if best_acc >= 0.50:
    print(f"\n✅✅✅ 50% 달성 성공!")
    improvement = (best_acc - 0.4913) * 100
    print(f"   Baseline 대비: +{improvement:.2f}%p")
else:
    shortage = (0.50 - best_acc) * 100
    print(f"\n⚠️ 50% 미달성 ({shortage:.2f}%p 부족)")
    print(f"   다음 단계: Optuna 심화 필요")

# 최고 모델 저장
print(f"\n💾 최고 모델 저장...")
import os
os.makedirs('04_logs/stacking', exist_ok=True)

if best_name.startswith("Stacking"):
    if "Logistic" in best_name:
        joblib.dump(lr_meta, '04_logs/stacking/meta_lr.joblib')
    elif "LightGBM" in best_name:
        joblib.dump(lgb_meta, '04_logs/stacking/meta_lgb.joblib')
    
    # Base models도 저장
    joblib.dump(lgb_model1, '04_logs/stacking/base_lgb1.joblib')
    joblib.dump(xgb_model1, '04_logs/stacking/base_xgb1.joblib')
    joblib.dump(lgb_model2, '04_logs/stacking/base_lgb2.joblib')
    
    print(f"   ✅ Base models + Meta-learner 저장 완료")

print("\n" + "="*80)
print("✅ STEP 1 완료!")
print("="*80)
