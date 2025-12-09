"""
시간 기반 + 추가 개선
1. 추가 피처 (과거 정보만)
2. SMOTE 오버샘플링
3. Class Weight
4. Optuna 튜닝
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
import json
import joblib
import time
import os

print("="*80)
print("🚀 시간 기반 Split + F1 85% 최적화")
print("="*80)

# ============================================================
# 1. 데이터 로드 및 시간 기반 Split
# ============================================================
print("\n[1/7] 데이터 로드 및 시간 기반 Split")

df = pd.read_csv('02_data/00_raw/credit_card_transactions-ibm_v2.csv')

df['Date'] = pd.to_datetime(
    df['Year'].astype(str) + '-' + 
    df['Month'].astype(str).str.zfill(2) + '-' + 
    df['Day'].astype(str).str.zfill(2)
)

max_date = df['Date'].max()
cutoff_date = max_date - timedelta(days=365*10)
df = df[df['Date'] >= cutoff_date].copy()

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

user_stats = df.groupby('User').agg({'Date': ['count', 'min', 'max']}).reset_index()
user_stats.columns = ['User', 'tx_count', 'first_date', 'last_date']
user_stats['monthly_avg'] = user_stats['tx_count'] / ((user_stats['last_date'] - user_stats['first_date']).dt.days / 30 + 1)
loyal_users = user_stats[user_stats['monthly_avg'] >= 10]['User'].values
df = df[df['User'].isin(loyal_users)].copy()

cat_list = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
cat_to_idx = {cat: i for i, cat in enumerate(cat_list)}
df['Category_idx'] = df['Category'].map(cat_to_idx)

# 시간 기반 Split
df_sorted = df.sort_values('Date')
split_idx = int(len(df_sorted) * 0.8)
split_date = df_sorted.iloc[split_idx]['Date']

train_df = df[df['Date'] < split_date].copy().sort_values(['User', 'Date', 'Time']).reset_index(drop=True)
test_df = df[df['Date'] >= split_date].copy().sort_values(['User', 'Date', 'Time']).reset_index(drop=True)

print(f"  Split: {split_date.date()}")
print(f"  Train: {len(train_df):,}건")
print(f"  Test:  {len(test_df):,}건")

# ============================================================
# 2. 확장 피처 엔지니어링 (과거 정보만!)
# ============================================================
print("\n[2/7] 확장 피처 엔지니어링")

def add_extended_features(df_input, train_stats=None, is_train=True):
    df = df_input.copy()
    
    # 기본 피처
    df['Amount_clean'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Amount_log'] = np.log1p(df['Amount_clean'])
    df['AmountBin'] = pd.cut(df['Amount_clean'], bins=[0, 10, 50, 100, 200, 500, float('inf')], labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(0)
    
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] < 6)).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 18)).astype(int)
    df['IsLunchTime'] = ((df['Hour'] >= 11) & (df['Hour'] <= 14)).astype(int)
    
    # 사용자 프로필
    if train_stats is None:
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
    
    df = df.merge(train_stats, on='User', how='left')
    df['User_AvgAmount'] = df['User_AvgAmount'].fillna(df['Amount_clean'].mean())
    df['User_StdAmount'] = df['User_StdAmount'].fillna(df['Amount_clean'].std())
    df['User_TxCount'] = df['User_TxCount'].fillna(0)
    for cat in cat_list:
        df[f'User_{cat}_Ratio'] = df[f'User_{cat}_Ratio'].fillna(0)
    
    # 이동 평균 (과거만!)
    df['Last5_AvgAmount'] = df.groupby('User')['Amount_clean'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).fillna(0)
    
    df['Last10_AvgAmount'] = df.groupby('User')['Amount_clean'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    ).fillna(0)
    
    # 이전 카테고리
    df['Previous_Category'] = df.groupby('User')['Category_idx'].shift(1).fillna(-1).astype(int)
    
    # 시간대 그룹
    df['HourBin'] = pd.cut(df['Hour'], bins=[-1, 6, 9, 12, 14, 18, 24], labels=[0, 1, 2, 3, 4, 5]).astype(float)
    
    return df, train_stats

train_df, train_stats = add_extended_features(train_df, train_stats=None, is_train=True)
test_df, _ = add_extended_features(test_df, train_stats=train_stats, is_train=False)

print(f"  ✅ 확장 피처 생성 완료")

# ============================================================
# 3. 데이터 준비
# ============================================================
print("\n[3/7] 데이터 준비")

feature_cols = [
    'Amount_clean', 'Amount_log', 'AmountBin',
    'Hour', 'DayOfWeek', 'DayOfMonth',
    'IsWeekend', 'IsNight', 'IsBusinessHour', 'IsLunchTime',
    'User_AvgAmount', 'User_StdAmount', 'User_TxCount',
    'User_교통_Ratio', 'User_생활_Ratio', 'User_쇼핑_Ratio',
    'User_식료품_Ratio', 'User_외식_Ratio', 'User_주유_Ratio',
    'Last5_AvgAmount', 'Last10_AvgAmount', 'Previous_Category', 'HourBin'
]

print(f"  피처: {len(feature_cols)}개")

train_df[feature_cols] = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
test_df[feature_cols] = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df['Category_idx'].values.astype(np.int32)
X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['Category_idx'].values.astype(np.int32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

# ============================================================
# 4. SMOTE 오버샘플링
# ============================================================
print("\n[4/7] SMOTE 오버샘플링")

unique, counts = np.unique(y_train, return_counts=True)
print(f"  원본 분포:")
for idx, (u, c) in enumerate(zip(unique, counts)):
    print(f"    {cat_list[idx]:6s}: {c:,}건 ({c/len(y_train)*100:.1f}%)")

# 샘플링 전략 (최소 클래스를 평균의 70%로)
avg_count = int(counts.mean() * 0.7)
sampling_strategy = {idx: max(count, avg_count) for idx, count in enumerate(counts)}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"  SMOTE 후: {len(X_train_smote):,}건")

# ============================================================
# 5. Class Weight
# ============================================================
print("\n[5/7] Class Weight 계산")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
class_weight_dict[5] *= 3.0  # 주유 3배

sample_weights = np.array([class_weight_dict[y] for y in y_train_smote])

print(f"  주요 가중치:")
print(f"    주유: {class_weight_dict[5]:.2f}")
print(f"    식료품: {class_weight_dict[3]:.2f}")

# ============================================================
# 6. Optuna 튜닝
# ============================================================
print("\n[6/7] Optuna 하이퍼파라미터 튜닝 (30 trials)")

# 샘플링
sample_size = min(500000, len(X_train_smote))
sample_idx = np.random.choice(len(X_train_smote), sample_size, replace=False)

def objective(trial):
    params = {
        'device': 'cuda',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_smote[sample_idx], y_train_smote[sample_idx], sample_weight=sample_weights[sample_idx])
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

start = time.time()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
tuning_time = time.time() - start

print(f"✅ 튜닝 완료: {tuning_time/60:.1f}분")
print(f"  최적 F1: {study.best_value:.4f}")

# ============================================================
# 7. 최종 모델 학습
# ============================================================
print("\n[7/7] 최종 모델 학습")

best_params = study.best_params.copy()
best_params['device'] = 'cuda'
best_params['tree_method'] = 'hist'
best_params['random_state'] = 42

model = xgb.XGBClassifier(**best_params)
model.fit(X_train_smote, y_train_smote, sample_weight=sample_weights)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
category_f1 = f1_score(y_test, y_pred, average=None)

print(f"\n📊 최종 성능:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Macro F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")

print(f"\n카테고리별 F1:")
for cat, f1 in zip(cat_list, category_f1):
    print(f"  {cat:6s}: {f1:.4f} ({f1*100:.2f}%)")

# 저장
output_dir = '02_data/07_time_optimized'
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, f'{output_dir}/xgboost_final.joblib')

metadata = {
    'method': 'time_based + SMOTE + Class Weight + Optuna',
    'split_date': str(split_date.date()),
    'n_features': len(feature_cols),
    'features': feature_cols,
    'accuracy': float(accuracy),
    'macro_f1': float(macro_f1),
    'category_f1': {cat: float(f1) for cat, f1 in zip(cat_list, category_f1)},
    'best_params': best_params,
    'created_at': datetime.now().isoformat()
}

with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print("✅ 최적화 완료!")
print("="*80)

print(f"\n🎯 목표 달성:")
print(f"  목표: F1 85%")
print(f"  결과: F1 {macro_f1*100:.2f}%")
if macro_f1 >= 0.85:
    print(f"  ✅ 목표 달성!")
else:
    print(f"  ⚠️ 부족: {(0.85 - macro_f1)*100:.2f}%p")

print(f"\n📂 저장: {output_dir}")
print("="*80)
