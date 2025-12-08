"""
활성 사용자 필터링 + SMOTE 증강 파이프라인
조건: 월 10건 이상, 5개월 이상 활동 사용자
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import json

print("="*70)
print("활성 사용자 필터링 + SMOTE 증강 파이프라인")
print("조건: 월 10건 이상, 5개월 이상 활동")
print("="*70)

# 1. 원본 데이터에서 날짜 정보 로드
print("\n[1/5] 원본 데이터에서 날짜 정보 로드")
raw_file = '02_data/00_raw/credit_card_transactions-ibm_v2.csv'
df_raw = pd.read_csv(raw_file)
print(f"  원본 데이터: {len(df_raw):,}개 레코드")
print(f"  컬럼: {list(df_raw.columns)}")

# 2. 월별 거래 수 계산
print("\n[2/5] 월별 거래 수 계산")
df_raw['YearMonth'] = df_raw['Year'].astype(str) + '-' + df_raw['Month'].astype(str).str.zfill(2)
monthly_tx = df_raw.groupby(['User', 'YearMonth']).size().reset_index(name='tx_count')

# 월 10건 이상인 월만 카운트
active_months = monthly_tx[monthly_tx['tx_count'] >= 10].groupby('User').size()
print(f"  총 사용자: {df_raw['User'].nunique()}명")

# 5개월 이상 활동한 사용자
MIN_MONTHS = 5
active_users = active_months[active_months >= MIN_MONTHS].index.tolist()
print(f"  활성 사용자 (월 10건 이상 x {MIN_MONTHS}개월): {len(active_users)}명")

# 3. 전처리된 데이터에서 활성 사용자 필터링
print("\n[3/5] 전처리된 데이터 필터링")
df = pd.read_csv('02_data/01_processed/preprocessed_enhanced.csv')
df_filtered = df[df['User'].isin(active_users)]

print(f"  원본 전처리 데이터: {len(df):,}개")
print(f"  활성 사용자 데이터: {len(df_filtered):,}개 ({len(df_filtered)/len(df)*100:.1f}%)")

# 카테고리 분포
print("\n  카테고리 분포:")
for cat, count in df_filtered['Next_Category'].value_counts().items():
    print(f"    {cat}: {count:,} ({count/len(df_filtered)*100:.1f}%)")

# 4. SMOTE 증강
print("\n[4/5] SMOTE 증강")
feature_cols = [col for col in df_filtered.columns if col.endswith('_scaled')]
X = df_filtered[feature_cols].values.astype('float32')
y = df_filtered['Next_Category_encoded'].values.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  학습 데이터: {len(X_train):,}개")
print(f"  테스트 데이터: {len(X_test):,}개")

# SMOTE 적용
print("\n  SMOTE 적용 중...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"  증강 전: {len(X_train):,}개")
print(f"  증강 후: {len(X_train_smote):,}개 ({len(X_train_smote)/len(X_train):.1f}배)")

# 증강 후 클래스 분포
print("\n  증강 후 클래스 분포:")
unique, counts = np.unique(y_train_smote, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    클래스 {u}: {c:,}")

# 5. 저장
print("\n[5/5] 데이터 저장")
output_dir = '02_data/02_augmented'
os.makedirs(output_dir, exist_ok=True)

# numpy 배열로 저장
np.save(f'{output_dir}/X_train_smote.npy', X_train_smote)
np.save(f'{output_dir}/y_train_smote.npy', y_train_smote)
np.save(f'{output_dir}/X_test.npy', X_test)
np.save(f'{output_dir}/y_test.npy', y_test)

# 메타데이터 저장
metadata = {
    'filter_condition': f'월 10건 이상 x {MIN_MONTHS}개월 이상',
    'original_samples': len(df),
    'filtered_samples': len(df_filtered),
    'active_users': len(active_users),
    'train_original': len(X_train),
    'train_smote': len(X_train_smote),
    'test_samples': len(X_test),
    'feature_count': len(feature_cols),
    'feature_names': feature_cols
}

with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ 저장 완료: {output_dir}/")
print(f"     - X_train_smote.npy ({X_train_smote.shape})")
print(f"     - y_train_smote.npy ({y_train_smote.shape})")
print(f"     - X_test.npy ({X_test.shape})")
print(f"     - y_test.npy ({y_test.shape})")

print("\n" + "="*70)
print("✅ 활성 사용자 필터링 + SMOTE 증강 완료!")
print("="*70)
