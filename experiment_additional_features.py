"""
추가 피처 실험: Merchant + 시퀀스 패턴
- 원본 CSV에서 Merchant City, State 정보 활용
- Rolling statistics (시퀀스)
- 빠른 테스트 (1M 샘플)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time

print("="*80)
print("🚀 추가 피처 실험: Merchant + 시퀀스")
print("="*80)

# ============================================================
# 1. 원본 CSV 로드 (Merchant 정보 포함)
# ============================================================
print("\n[1/5] 원본 CSV 로드")

# 필요한 컬럼만 로드
df = pd.read_csv('02_data/00_raw/credit_card_transactions-ibm_v2.csv',
                 usecols=['User', 'Year', 'Month', 'Day', 'Time', 'Amount',
                         'Use Chip', 'Merchant Name', 'Merchant City', 
                         'Merchant State', 'MCC'])

print(f"  원본 데이터: {len(df):,}건")

# 카테고리 매핑
import sys
sys.path.append('00_config/00_mapping')
from category_mapping import get_category_mapping

category_map = get_category_mapping()
df['Category'] = df['MCC'].map(category_map)
df = df[df['Category'].notna()].copy()

print(f"  필터링 후: {len(df):,}건")

# ============================================================
# 2. 기본 전처리 + Merchant 피처
# ============================================================
print("\n[2/5] 기본 전처리 + Merchant 피처")

# Amount 정제
df['Amount'] = df['Amount'].str.replace('$', '').str.replace(',', '').astype(float)

# 시간 피처
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
df['DayOfWeek'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.dayofweek

# Merchant 피처
print("  Merchant 피처 생성...")

# 1) Merchant Name 인코딩 (해시값이지만 패턴 있음)
le_merchant = LabelEncoder()
df['Merchant_ID'] = le_merchant.fit_transform(df['Merchant Name'].astype(str))

# 2) Merchant City/State 인코딩
le_city = LabelEncoder()
le_state = LabelEncoder()
df['Merchant_City_ID'] = le_city.fit_transform(df['Merchant City'].astype(str))
df['Merchant_State_ID'] = le_state.fit_transform(df['Merchant State'].astype(str))

# 3) Merchant 빈도
merchant_freq = df['Merchant Name'].value_counts()
df['Merchant_Frequency'] = df['Merchant Name'].map(merchant_freq)

# 4) City 평균 금액
city_avg = df.groupby('Merchant City')['Amount'].mean()
df['City_Avg_Amount'] = df['Merchant City'].map(city_avg)

# 5) State 평균 금액
state_avg = df.groupby('Merchant State')['Amount'].mean()
df['State_Avg_Amount'] = df['Merchant State'].map(state_avg)

# ============================================================
# 3. 시퀀스 패턴 피처
# ============================================================
print("\n[3/5] 시퀀스 패턴 피처")

# 날짜 생성 및 정렬
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df.sort_values(['User', 'Date', 'Time']).reset_index(drop=True)

print("  시퀀스 피처 생성...")

# 1) Rolling statistics (최근 3개 거래)
df['Rolling_Amount_Mean_3'] = df.groupby('User')['Amount'].transform(
    lambda x: x.rolling(3, min_periods=1).mean().shift(1)
).fillna(0)

df['Rolling_Amount_Std_3'] = df.groupby('User')['Amount'].transform(
    lambda x: x.rolling(3, min_periods=1).std().shift(1)
).fillna(0)

# 2) 같은 Merchant 재방문
df['Same_Merchant_Count'] = df.groupby(['User', 'Merchant Name']).cumcount()

# 3) 같은 State 연속 거래
df['Same_State_Streak'] = (df.groupby('User')['Merchant State'].shift() == df['Merchant State']).astype(int)

# 4) 시간 간격
df['Hours_Since_Last'] = df.groupby('User')['Date'].diff().dt.total_seconds() / 3600
df['Hours_Since_Last'] = df['Hours_Since_Last'].fillna(24)  # 첫 거래는 24시간

print(f"  ✅ 총 {len(df.columns)}개 컬럼 생성")

# ============================================================
# 4. 활성 사용자 필터링 + Train/Test 분할
# ============================================================
print("\n[4/5] 데이터 준비")

# 활성 사용자
tx_per_month = df.groupby(['User', 'Year', 'Month']).size()
active_months = tx_per_month[tx_per_month >= 10].reset_index().groupby('User').size()
active_users = active_months[active_months >= 5].index

df = df[df['User'].isin(active_users)].copy()
print(f"  활성 사용자 데이터: {len(df):,}건")

# 피처 선택
feature_columns = [
    'Amount', 'Hour', 'DayOfWeek',
    # Merchant 피처
    'Merchant_ID', 'Merchant_City_ID', 'Merchant_State_ID',
    'Merchant_Frequency', 'City_Avg_Amount', 'State_Avg_Amount',
    # 시퀀스 피처
    'Rolling_Amount_Mean_3', 'Rolling_Amount_Std_3',
    'Same_Merchant_Count', 'Same_State_Streak', 'Hours_Since_Last'
]

# 간단한 Train/Test 분할 (최근 20%)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# 사용자별 통계 추가
user_stats = train_df.groupby('User')['Amount'].agg(['mean', 'std', 'count'])
train_df['User_AvgAmount'] = train_df['User'].map(user_stats['mean'])
train_df['User_StdAmount'] = train_df['User'].map(user_stats['std']).fillna(0)
test_df['User_AvgAmount'] = test_df['User'].map(user_stats['mean']).fillna(train_df['Amount'].mean())
test_df['User_StdAmount'] = test_df['User'].map(user_stats['std']).fillna(0)

feature_columns.extend(['User_AvgAmount', 'User_StdAmount'])

X_train = train_df[feature_columns].fillna(0).values
y_train = train_df['Category'].astype('category').cat.codes.values
X_test = test_df[feature_columns].fillna(0).values
y_test = test_df['Category'].astype('category').cat.codes.values

print(f"  학습: {len(X_train):,}건")
print(f"  테스트: {len(X_test):,}건")
print(f"  피처: {len(feature_columns)}개")

# 샘플링 (빠른 테스트)
sample_size = min(1000000, len(X_train))
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]

print(f"  샘플: {len(X_train_sample):,}건 (테스트용)")

# ============================================================
# 5. LightGBM 학습 및 평가
# ============================================================
print("\n[5/5] LightGBM 학습 및 평가")

# 기존 27개 피처 vs 새로운 피처 비교
print("\n  [비교 1] 기존 피처 로드...")
X_train_old = np.load('02_data/02_augmented/X_train_smote.npy')[:sample_size]
y_train_old = np.load('02_data/02_augmented/y_train_smote.npy')[:sample_size]
X_test_old = np.load('02_data/02_augmented/X_test.npy')

# 기존 피처로 학습
lgb_old = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

start = time.time()
lgb_old.fit(X_train_old, y_train_old)
time_old = time.time() - start

y_pred_old = lgb_old.predict(X_test_old)
acc_old = accuracy_score(np.load('02_data/02_augmented/y_test.npy'), y_pred_old)

print(f"    ✅ 완료: {time_old:.2f}초")
print(f"    Accuracy: {acc_old:.4f} ({acc_old*100:.2f}%)")

# 새로운 피처로 학습
print("\n  [비교 2] 새로운 피처 (Merchant + 시퀀스)...")
lgb_new = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

start = time.time()
lgb_new.fit(X_train_sample, y_train_sample)
time_new = time.time() - start

y_pred_new = lgb_new.predict(X_test)
acc_new = accuracy_score(y_test, y_pred_new)
f1_new = f1_score(y_test, y_pred_new, average='macro')

print(f"    ✅ 완료: {time_new:.2f}초")
print(f"    Accuracy: {acc_new:.4f} ({acc_new*100:.2f}%)")
print(f"    Macro F1: {f1_new:.4f}")

# Feature Importance
importances = lgb_new.feature_importances_
top_features = sorted(zip(feature_columns, importances), key=lambda x: -x[1])[:10]

print(f"\n  Top 10 중요 피처:")
for rank, (feat, imp) in enumerate(top_features, 1):
    print(f"    {rank:2d}. {feat:25s}: {imp:.0f}")

# ============================================================
# 최종 비교
# ============================================================
print("\n" + "="*80)
print("🏆 추가 피처 실험 결과")
print("="*80)

results = [
    ("기존 피처 (27개)", 27, acc_old),
    ("새 피처 (Merchant+시퀀스)", len(feature_columns), acc_new),
]

print(f"\n{'모델':<30} {'피처 수':>10} {'Accuracy':>12} {'50% 달성':>12}")
print("-"*70)
for name, n_feat, acc in results:
    status = "✅" if acc >= 0.50 else "❌"
    print(f"{name:<30} {n_feat:>10} {acc:>12.4f} {status:>12}")
print("-"*70)

improvement = (acc_new - acc_old) * 100
print(f"\n📊 개선도:")
print(f"  Accuracy: {improvement:+.2f}%p")

if acc_new >= 0.50:
    print(f"\n🎉🎉🎉 50% 달성 성공!")
    print(f"   Merchant + 시퀀스 피처가 효과적!")
elif acc_new > acc_old:
    print(f"\n✅ 성능 개선 확인!")
    print(f"   추가 피처가 효과 있음")
else:
    print(f"\n⚠️ 개선 효과 제한적")
    print(f"   다른 접근 필요")

print("\n" + "="*80)
print("✅ 추가 피처 실험 완료!")
print("="*80)
