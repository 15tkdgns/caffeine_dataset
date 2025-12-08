"""
카테고리 재정의 실험
- 생활 카테고리 통합/제거 시 성능 변화 확인
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time

print("="*80)
print("🔬 카테고리 재정의 실험")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/4] 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

# 샘플링 (빠른 실험)
sample_size = min(2000000, len(X_train))
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]

print(f"  학습 샘플: {len(X_train_sample):,}건")
print(f"  테스트: {len(X_test):,}건")

category_names_6 = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']

# ============================================================
# 2. Baseline (6개 카테고리)
# ============================================================
print("\n[2/4] Baseline (6개 카테고리)")

model_6 = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

start = time.time()
model_6.fit(X_train_sample, y_train_sample)
train_time_6 = time.time() - start

y_pred_6 = model_6.predict(X_test)
acc_6 = accuracy_score(y_test, y_pred_6)
f1_6 = f1_score(y_test, y_pred_6, average='macro')

print(f"  Accuracy: {acc_6:.4f} ({acc_6*100:.2f}%)")
print(f"  Macro F1: {f1_6:.4f}")
print(f"  학습 시간: {train_time_6:.2f}초")

# ============================================================
# 3. 시나리오 1: 생활 → 쇼핑 통합 (5개)
# ============================================================
print("\n[3/4] 시나리오 1: 생활 → 쇼핑 통합 (5개 카테고리)")

# 레이블 변환: 생활(1) → 쇼핑(2)
y_train_5a = y_train_sample.copy()
y_test_5a = y_test.copy()

# 생활(1) → 쇼핑(2)로 변경
y_train_5a[y_train_5a == 1] = 2
y_test_5a[y_test_5a == 1] = 2

# 레이블 재정렬: 0, 2, 3, 4, 5 → 0, 1, 2, 3, 4
label_map_5a = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
y_train_5a = np.array([label_map_5a[y] for y in y_train_5a])
y_test_5a = np.array([label_map_5a[y] for y in y_test_5a])

category_names_5a = ['교통', '쇼핑(+생활)', '식료품', '외식', '주유']

model_5a = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

start = time.time()
model_5a.fit(X_train_sample, y_train_5a)
train_time_5a = time.time() - start

y_pred_5a = model_5a.predict(X_test)
acc_5a = accuracy_score(y_test_5a, y_pred_5a)
f1_5a = f1_score(y_test_5a, y_pred_5a, average='macro')

print(f"  Accuracy: {acc_5a:.4f} ({acc_5a*100:.2f}%)")
print(f"  Macro F1: {f1_5a:.4f}")
print(f"  학습 시간: {train_time_5a:.2f}초")

# 카테고리별 F1
cat_f1_5a = f1_score(y_test_5a, y_pred_5a, average=None)
print(f"\n  카테고리별 F1:")
for cat, f1 in zip(category_names_5a, cat_f1_5a):
    print(f"     {cat:12s}: {f1:.4f}")

# ============================================================
# 4. 시나리오 2: 생활 제외 (5개)
# ============================================================
print("\n[4/4] 시나리오 2: 생활 제외 (5개 카테고리)")

# 생활 데이터 제거
train_mask = y_train_sample != 1
test_mask = y_test != 1

X_train_5b = X_train_sample[train_mask]
y_train_5b = y_train_sample[train_mask]
X_test_5b = X_test[test_mask]
y_test_5b = y_test[test_mask]

# 레이블 재정렬: 0, 2, 3, 4, 5 → 0, 1, 2, 3, 4
label_map_5b = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
y_train_5b = np.array([label_map_5b[y] for y in y_train_5b])
y_test_5b = np.array([label_map_5b[y] for y in y_test_5b])

category_names_5b = ['교통', '쇼핑', '식료품', '외식', '주유']

model_5b = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

start = time.time()
model_5b.fit(X_train_5b, y_train_5b)
train_time_5b = time.time() - start

y_pred_5b = model_5b.predict(X_test_5b)
acc_5b = accuracy_score(y_test_5b, y_pred_5b)
f1_5b = f1_score(y_test_5b, y_pred_5b, average='macro')

print(f"  학습 데이터: {len(X_train_5b):,}건 (생활 제외)")
print(f"  테스트 데이터: {len(X_test_5b):,}건")
print(f"\n  Accuracy: {acc_5b:.4f} ({acc_5b*100:.2f}%)")
print(f"  Macro F1: {f1_5b:.4f}")
print(f"  학습 시간: {train_time_5b:.2f}초")

# 카테고리별 F1
cat_f1_5b = f1_score(y_test_5b, y_pred_5b, average=None)
print(f"\n  카테고리별 F1:")
for cat, f1 in zip(category_names_5b, cat_f1_5b):
    print(f"     {cat:12s}: {f1:.4f}")

# ============================================================
# 결과 비교
# ============================================================
print("\n" + "="*80)
print("🏆 카테고리 재정의 실험 결과")
print("="*80)

print(f"\n{'시나리오':<35} {'카테고리':<8} {'Accuracy':>12} {'Macro F1':>12}")
print("-"*70)
print(f"{'Baseline':<35} {'6개':<8} {acc_6:>12.4f} {f1_6:>12.4f}")
print(f"{'생활→쇼핑 통합':<35} {'5개':<8} {acc_5a:>12.4f} {f1_5a:>12.4f}")
print(f"{'생활 제외':<35} {'5개':<8} {acc_5b:>12.4f} {f1_5b:>12.4f}")
print("-"*70)

# 개선도
print(f"\n📊 Baseline 대비 개선:")
print(f"  생활→쇼핑 통합: Accuracy {(acc_5a - acc_6)*100:+.2f}%p, Macro F1 {(f1_5a - f1_6)*100:+.2f}%p")
print(f"  생활 제외:      Accuracy {(acc_5b - acc_6)*100:+.2f}%p, Macro F1 {(f1_5b - f1_6)*100:+.2f}%p")

# 50% 달성 여부
print(f"\n🎯 50% 달성 여부:")
if acc_5a >= 0.50:
    print(f"  ✅ 생활→쇼핑 통합: {acc_5a*100:.2f}% (50% 돌파!)")
else:
    print(f"  ⚠️ 생활→쇼핑 통합: {acc_5a*100:.2f}%")

if acc_5b >= 0.50:
    print(f"  ✅ 생활 제외: {acc_5b*100:.2f}% (50% 돌파!)")
else:
    print(f"  ⚠️ 생활 제외: {acc_5b*100:.2f}%")

print("\n" + "="*80)
print("✅ 카테고리 재정의 실험 완료!")
print("="*80)
