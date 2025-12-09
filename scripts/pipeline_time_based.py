"""
시간 기반 Train/Test Split (데이터 유출 제거)
Train 데이터로만 피처 계산 → Test에 적용
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import json
import os

print("="*80)
print("⏰ 시간 기반 Train/Test Split (데이터 유출 제거)")
print("="*80)

# ============================================================
# 1. 데이터 로드 및 필터링
# ============================================================
print("\n[1/6] 데이터 로드")

df = pd.read_csv('02_data/00_raw/credit_card_transactions-ibm_v2.csv')
print(f"  원본: {len(df):,}건")

# 날짜 생성
df['Date'] = pd.to_datetime(
    df['Year'].astype(str) + '-' + 
    df['Month'].astype(str).str.zfill(2) + '-' + 
    df['Day'].astype(str).str.zfill(2)
)

# 최근 10년
max_date = df['Date'].max()
cutoff_date = max_date - timedelta(days=365*10)
df = df[df['Date'] >= cutoff_date].copy()

# 카테고리 매핑
mcc_to_category = {
    range(5411, 5500): '식료품', range(5811, 5900): '외식',
    range(5200, 5300): '쇼핑', range(5300, 5400): '쇼핑', range(5600, 5700): '쇼핑',
    range(5500, 5600): '주유', range(4000, 4100): '교통', range(4100, 4200): '교통',
    range(4800, 4900): '생활', range(6000, 6100): '생활'
}

def get_category(mcc):
    for mcc_range, cat in mcc_to_category.items():
        if mcc in mcc_range:
            return cat
    return None

df['Category'] = df['MCC'].apply(get_category)
df = df[df['Category'].notna()].copy()

# 로열 고객
user_stats = df.groupby('User').agg({'Date': ['count', 'min', 'max']}).reset_index()
user_stats.columns = ['User', 'tx_count', 'first_date', 'last_date']
user_stats['monthly_avg'] = user_stats['tx_count'] / ((user_stats['last_date'] - user_stats['first_date']).dt.days / 30 + 1)
loyal_users = user_stats[user_stats['monthly_avg'] >= 10]['User'].values
df = df[df['User'].isin(loyal_users)].copy()

cat_list = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
cat_to_idx = {cat: i for i, cat in enumerate(cat_list)}
df['Category_idx'] = df['Category'].map(cat_to_idx)

print(f"  필터링 후: {len(df):,}건, {len(loyal_users):,}명")

# ============================================================
# 2. 시간 기반 Train/Test Split (80/20)
# ============================================================
print("\n[2/6] ⏰ 시간 기반 Train/Test Split")

# 80% 시점 날짜 계산
df_sorted = df.sort_values('Date')
split_idx = int(len(df_sorted) * 0.8)
split_date = df_sorted.iloc[split_idx]['Date']

train_df = df[df['Date'] < split_date].copy()
test_df = df[df['Date'] >= split_date].copy()

print(f"  Split 날짜: {split_date.date()}")
print(f"  Train: {len(train_df):,}건 ({train_df['Date'].min().date()} ~ {train_df['Date'].max().date()})")
print(f"  Test:  {len(test_df):,}건 ({test_df['Date'].min().date()} ~ {test_df['Date'].max().date()})")

# ============================================================
# 3. 피처 엔지니어링 (Train만 사용!)
# ============================================================
print("\n[3/6] 피처 엔지니어링 (Train 데이터만 사용)")

def add_features(df_input, train_stats=None):
    """피처 추가 (train_stats가 있으면 사용, 없으면 계산)"""
    df = df_input.copy()
    
    # 금액
    df['Amount_clean'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Amount_log'] = np.log1p(df['Amount_clean'])
    df['AmountBin'] = pd.cut(df['Amount_clean'], bins=[0, 10, 50, 100, 200, 500, float('inf')], labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(0)
    
    # 시간
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] < 6)).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 18)).astype(int)
    
    # 사용자 프로필 (train_stats 사용 또는 계산)
    if train_stats is None:
        # Train 데이터: 계산
        user_profile = df.groupby('User').agg({'Amount_clean': ['mean', 'std'], 'Category_idx': 'count'}).reset_index()
        user_profile.columns = ['User', 'User_AvgAmount', 'User_StdAmount', 'User_TxCount']
        
        user_cat_counts = df.groupby(['User', 'Category']).size().unstack(fill_value=0)
        user_cat_total = user_cat_counts.sum(axis=1)
        for cat in cat_list:
            if cat in user_cat_counts.columns:
                user_profile[f'User_{cat}_Ratio'] = (user_cat_counts[cat] / user_cat_total).values
            else:
                user_profile[f'User_{cat}_Ratio'] = 0.0
        
        train_stats = user_profile
    
    # 병합
    df = df.merge(train_stats, on='User', how='left')
    
    # 결측값 처리 (Test에 없는 사용자)
    df['User_AvgAmount'] = df['User_AvgAmount'].fillna(df['Amount_clean'].mean())
    df['User_StdAmount'] = df['User_StdAmount'].fillna(df['Amount_clean'].std())
    df['User_TxCount'] = df['User_TxCount'].fillna(0)
    for cat in cat_list:
        df[f'User_{cat}_Ratio'] = df[f'User_{cat}_Ratio'].fillna(0)
    
    return df, train_stats

# Train 피처 계산
train_df, train_stats = add_features(train_df, train_stats=None)
print(f"  ✅ Train 피처 계산 완료")

# Test는 Train 통계 사용!
test_df, _ = add_features(test_df, train_stats=train_stats)
print(f"  ✅ Test에 Train 통계 적용")

# ============================================================
# 4. 데이터 준비
# ============================================================
print("\n[4/6] 데이터 준비")

feature_cols = [
    'Amount_clean', 'Amount_log', 'AmountBin',
    'Hour', 'DayOfWeek', 'DayOfMonth',
    'IsWeekend', 'IsNight', 'IsBusinessHour',
    'User_AvgAmount', 'User_StdAmount', 'User_TxCount',
    'User_교통_Ratio', 'User_생활_Ratio', 'User_쇼핑_Ratio',
    'User_식료품_Ratio', 'User_외식_Ratio', 'User_주유_Ratio'
]

print(f"  피처: {len(feature_cols)}개")

# 결측값 처리
train_df[feature_cols] = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
test_df[feature_cols] = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df['Category_idx'].values.astype(np.int32)
X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['Category_idx'].values.astype(np.int32)

# 스케일링 (Train으로만 fit)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Train scaler 사용

print(f"  학습: {len(X_train):,}, 테스트: {len(X_test):,}")

# ============================================================
# 5. 모델 학습
# ============================================================
print("\n[5/6] XGBoost GPU 학습")

model = xgb.XGBClassifier(
    device='cuda', tree_method='hist',
    n_estimators=300, max_depth=10, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

import time
start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start

print(f"  ✅ 학습 완료: {train_time:.1f}초")

# ============================================================
# 6. 평가
# ============================================================
print("\n[6/6] 성능 평가")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
category_f1 = f1_score(y_test, y_pred, average=None)

print(f"\n  📊 전체 성능:")
print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    Macro F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")

print(f"\n  카테고리별 F1:")
for cat, f1 in zip(cat_list, category_f1):
    print(f"    {cat:6s}: {f1:.4f} ({f1*100:.2f}%)")

# 저장
output_dir = '02_data/06_time_based'
os.makedirs(output_dir, exist_ok=True)

import joblib
joblib.dump(model, f'{output_dir}/xgboost_time_based.joblib')

metadata = {
    'split_method': 'time_based',
    'split_date': str(split_date.date()),
    'train_period': f"{train_df['Date'].min().date()} ~ {train_df['Date'].max().date()}",
    'test_period': f"{test_df['Date'].min().date()} ~ {test_df['Date'].max().date()}",
    'n_features': len(feature_cols),
    'features': feature_cols,
    'accuracy': float(accuracy),
    'macro_f1': float(macro_f1),
    'category_f1': {cat: float(f1) for cat, f1 in zip(cat_list, category_f1)},
    'train_time': train_time,
    'created_at': datetime.now().isoformat()
}

with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ============================================================
# 요약
# ============================================================
print("\n" + "="*80)
print("✅ 시간 기반 Split 완료!")
print("="*80)

print(f"\n🎯 목표 달성 여부:")
print(f"   목표: F1 85%")
print(f"   결과: F1 {macro_f1*100:.2f}%")
if macro_f1 >= 0.85:
    print(f"   ✅ 목표 달성!")
else:
    print(f"   ⚠️ 추가 개선 필요 ({(0.85 - macro_f1)*100:.2f}%p)")

print(f"\n📂 저장: {output_dir}")
print("="*80)
