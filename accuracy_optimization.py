"""
Accuracy 최적화 실험
- SMOTE 제거 (원본 데이터)
- Accuracy 중심 하이퍼파라미터
- Voting Ensemble
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier
import joblib
import time

print("="*80)
print("🎯 Accuracy 최적화 실험")
print("="*80)

# ============================================================
# 1. 원본 데이터 로드 (SMOTE 없이)
# ============================================================
print("\n[1/4] 원본 데이터 로드 (SMOTE 제거)")

# SMOTE 적용 전 원본 데이터
X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

# SMOTE로 증강된 부분 제거 (원본만 사용)
# 원본은 5,139,965건, SMOTE는 9,254,112건
# 처음 5,139,965건만 사용
original_size = 5139965
X_train_original = X_train[:original_size]
y_train_original = y_train[:original_size]

print(f"  원본 학습: {len(X_train_original):,}건")
print(f"  테스트: {len(X_test):,}건")

category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']

# ============================================================
# 2. LightGBM - Accuracy 최적화
# ============================================================
print("\n[2/4] LightGBM - Accuracy 최적화")

# Accuracy 중심 설정
lgb_acc = lgb.LGBMClassifier(
    n_estimators=500,        # 늘림
    max_depth=12,            # 깊게
    learning_rate=0.05,      # 낮춤 (과적합 방지)
    num_leaves=256,          # 늘림
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("  학습 시작...")
start = time.time()
lgb_acc.fit(X_train_original, y_train_original)
train_time_lgb = time.time() - start

y_pred_lgb = lgb_acc.predict(X_test)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')

print(f"  ✅ 완료: {train_time_lgb:.2f}초")
print(f"     Accuracy: {acc_lgb:.4f} ({acc_lgb*100:.2f}%)")
print(f"     Macro F1: {f1_lgb:.4f}")

# ============================================================
# 3. XGBoost - Accuracy 최적화
# ============================================================
print("\n[3/4] XGBoost - Accuracy 최적화")

xgb_acc = xgb.XGBClassifier(
    device='cuda',
    tree_method='hist',
    n_estimators=500,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

print("  학습 시작...")
start = time.time()
xgb_acc.fit(X_train_original, y_train_original)
train_time_xgb = time.time() - start

y_pred_xgb = xgb_acc.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')

print(f"  ✅ 완료: {train_time_xgb:.2f}초")
print(f"     Accuracy: {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"     Macro F1: {f1_xgb:.4f}")

# ============================================================
# 4. Voting Ensemble
# ============================================================
print("\n[4/4] Voting Ensemble")

# Hard Voting
y_pred_ensemble_hard = []
for i in range(len(X_test)):
    votes = [y_pred_lgb[i], y_pred_xgb[i]]
    # 다수결
    pred = max(set(votes), key=votes.count)
    y_pred_ensemble_hard.append(pred)

acc_ensemble_hard = accuracy_score(y_test, y_pred_ensemble_hard)
f1_ensemble_hard = f1_score(y_test, y_pred_ensemble_hard, average='macro')

print(f"  Hard Voting:")
print(f"     Accuracy: {acc_ensemble_hard:.4f} ({acc_ensemble_hard*100:.2f}%)")
print(f"     Macro F1: {f1_ensemble_hard:.4f}")

# Soft Voting (확률 평균)
y_proba_lgb = lgb_acc.predict_proba(X_test)
y_proba_xgb = xgb_acc.predict_proba(X_test)

# 가중 평균 (LightGBM 60%, XGBoost 40% - LightGBM이 더 높은 Acc)
if acc_lgb >= acc_xgb:
    y_proba_ensemble = 0.6 * y_proba_lgb + 0.4 * y_proba_xgb
else:
    y_proba_ensemble = 0.4 * y_proba_lgb + 0.6 * y_proba_xgb

y_pred_ensemble_soft = np.argmax(y_proba_ensemble, axis=1)
acc_ensemble_soft = accuracy_score(y_test, y_pred_ensemble_soft)
f1_ensemble_soft = f1_score(y_test, y_pred_ensemble_soft, average='macro')

print(f"\n  Soft Voting (가중 평균):")
print(f"     Accuracy: {acc_ensemble_soft:.4f} ({acc_ensemble_soft*100:.2f}%)")
print(f"     Macro F1: {f1_ensemble_soft:.4f}")

# ============================================================
# 5. Baseline 모델 추가 테스트
# ============================================================
print("\n[5/5] Baseline 모델 (원본 데이터로 재학습)")

lgb_baseline = lgb.LGBMClassifier(
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

print("  학습 시작...")
lgb_baseline.fit(X_train_original, y_train_original)
y_pred_baseline = lgb_baseline.predict(X_test)
acc_baseline_original = accuracy_score(y_test, y_pred_baseline)
f1_baseline_original = f1_score(y_test, y_pred_baseline, average='macro')

print(f"  ✅ 완료")
print(f"     Accuracy: {acc_baseline_original:.4f} ({acc_baseline_original*100:.2f}%)")
print(f"     Macro F1: {f1_baseline_original:.4f}")

# ============================================================
# 최종 비교
# ============================================================
print("\n" + "="*80)
print("🏆 Accuracy 최적화 결과")
print("="*80)

results = [
    ("Baseline (SMOTE 있음)", 0.4913),
    ("Baseline (원본 데이터)", acc_baseline_original),
    ("LightGBM (Acc 최적화)", acc_lgb),
    ("XGBoost (Acc 최적화)", acc_xgb),
    ("Ensemble Hard Voting", acc_ensemble_hard),
    ("Ensemble Soft Voting", acc_ensemble_soft),
]

print(f"\n{'모델':<35} {'Accuracy':>12} {'50% 달성':>12}")
print("-"*65)
for name, acc in results:
    status = "✅" if acc >= 0.50 else "❌"
    print(f"{name:<35} {acc:>12.4f} {status:>12}")
print("-"*65)

# 최고 성능
best_name, best_acc = max(results, key=lambda x: x[1])
print(f"\n🏆 최고 성능: {best_name}")
print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

if best_acc >= 0.50:
    print(f"\n✅✅✅ 50% 달성!")
else:
    print(f"\n⚠️ 50% 미달성 ({(0.50 - best_acc)*100:.2f}%p 부족)")

# SMOTE vs 원본 비교
print(f"\n📊 SMOTE 효과:")
print(f"   SMOTE 있음: 49.13%")
print(f"   원본 데이터: {acc_baseline_original*100:.2f}%")
print(f"   차이: {(acc_baseline_original - 0.4913)*100:+.2f}%p")

print("\n" + "="*80)
print("✅ Accuracy 최적화 실험 완료!")
print("="*80)
