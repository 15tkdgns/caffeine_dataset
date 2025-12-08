"""
시퀀스 예측 전처리 파이프라인
X: 현재 거래 정보 (카테고리 포함)
Y: 다음 거래 카테고리
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os

def get_category_mapping():
    """6개 비즈니스 카테고리 MCC 매핑"""
    mcc_map = {
        # 식료품
        5411: '식료품', 5499: '식료품', 5462: '식료품',
        # 주유
        5541: '주유', 5542: '주유',
        # 교통
        4121: '교통', 4111: '교통', 4784: '교통', 4131: '교통',
        # 생활
        5912: '생활', 5200: '생활', 5722: '생활',
        # 외식
        5812: '외식', 5813: '외식', 5814: '외식', 5815: '외식',
        # 쇼핑
        5311: '쇼핑', 5310: '쇼핑', 5300: '쇼핑',
        5651: '쇼핑', 5942: '쇼핑', 5964: '쇼핑', 5945: '쇼핑',
    }
    return mcc_map


def create_sequence_features(df):
    """
    시퀀스 예측을 위한 피처 생성
    현재 거래 → 다음 거래 카테고리 예측
    """
    print("\n시퀀스 피처 생성 중...")
    
    # 1. 날짜/시간 파싱
    df['Transaction_DateTime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str) + '-' + 
        df['Day'].astype(str) + ' ' + 
        df['Time'], 
        errors='coerce'
    )
    
    # 2. 사용자별 시간순 정렬
    df = df.sort_values(['User', 'Transaction_DateTime']).reset_index(drop=True)
    
    # 3. 현재 카테고리 (X에 포함)
    df['Current_Category'] = df['Category']
    
    # 4. 다음 카테고리 (Y, 타겟)
    df['Next_Category'] = df.groupby('User')['Category'].shift(-1)
    
    # 5. 이전 카테고리 (추가 피처)
    df['Previous_Category'] = df.groupby('User')['Category'].shift(1)
    
    # 6. 시간 간격 (이전 거래와의 시간차)
    df['Time_Since_Last'] = df.groupby('User')['Transaction_DateTime'].diff().dt.total_seconds()
    df['Time_Since_Last'] = df['Time_Since_Last'].fillna(0)
    
    # 7. 사용자별 거래 순서
    df['Transaction_Sequence'] = df.groupby('User').cumcount() + 1
    
    # 8. 마지막 거래 제거 (다음 카테고리가 없음)
    original_count = len(df)
    df = df.dropna(subset=['Next_Category'])
    print(f"  - 마지막 거래 제거: {original_count - len(df):,}건")
    
    return df


def advanced_feature_engineering(df):
    """고급 피처 엔지니어링"""
    print("\n고급 피처 엔지니어링...")
    
    # 1. 시간 특성
    df['Hour'] = df['Transaction_DateTime'].dt.hour
    df['DayOfWeek'] = df['Transaction_DateTime'].dt.dayofweek
    df['DayOfMonth'] = df['Transaction_DateTime'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsNight'] = df['Hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
    df['IsBusinessHour'] = df['Hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    
    # 2. 금액 특성
    df['Amount'] = df['Amount'].replace({r'\$': '', r',': ''}, regex=True).astype(float)
    df['Amount'] = df['Amount'].fillna(0).clip(lower=0)
    df['Amount_log'] = np.log1p(df['Amount'])
    
    # 3. 사용자별 통계
    print("  사용자별 통계...")
    user_stats = df.groupby('User')['Amount'].agg([
        ('User_AvgAmount', 'mean'),
        ('User_StdAmount', 'std'),
    ]).reset_index()
    user_stats['User_StdAmount'] = user_stats['User_StdAmount'].fillna(0)
    df = df.merge(user_stats, on='User', how='left')
    
    # 4. 카테고리별 사용자 빈도
    print("  카테고리별 사용자 통계...")
    df['User_Category_Count'] = df.groupby(['User', 'Current_Category']).cumcount() + 1
    
    return df


def preprocess_sequence_pipeline(data_file, output_dir='02_data/01_processed'):
    """시퀀스 예측 전처리 파이프라인"""
    print("="*70)
    print("시퀀스 예측 전처리 파이프라인")
    print("="*70)
    
    start_time = datetime.now()
    
    # 1. 데이터 로드
    print(f"\n[1/6] 데이터 로드: {data_file}")
    df = pd.read_csv(data_file)
    print(f"  - 전체 데이터: {len(df):,}건")
    
    # 2. 카테고리 매핑
    print("\n[2/6] 카테고리 매핑")
    mcc_map = get_category_mapping()
    df['Category'] = df['MCC'].map(mcc_map)
    
    original_count = len(df)
    df.dropna(subset=['Category'], inplace=True)
    print(f"  - 유효 데이터: {len(df):,}건")
    print(f"  - 제외 데이터: {original_count - len(df):,}건")
    
    # 3. 시퀀스 피처 생성
    print("\n[3/6] 시퀀스 피처 생성")
    df = create_sequence_features(df)
    
    # 4. 고급 피처 엔지니어링
    print("\n[4/6] 고급 피처 엔지니어링")
    df = advanced_feature_engineering(df)
    
    # 5. 인코딩 및 스케일링
    print("\n[5/6] 인코딩 및 스케일링")
    
    # 카테고리 인코딩
    label_encoder = LabelEncoder()
    df['Current_Category_encoded'] = label_encoder.fit_transform(df['Current_Category'])
    df['Next_Category_encoded'] = label_encoder.transform(df['Next_Category'])
    
    # 이전 카테고리 인코딩 (NaN 처리)
    df['Previous_Category'] = df['Previous_Category'].fillna('NONE')
    df['Previous_Category_encoded'] = df['Previous_Category'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
    )
    
    # 사용할 특성
    numeric_features = [
        'Amount', 'Amount_log',
        'Hour', 'DayOfWeek', 'DayOfMonth',
        'IsWeekend', 'IsNight', 'IsBusinessHour',
        'User_AvgAmount', 'User_StdAmount',
        'Time_Since_Last', 'Transaction_Sequence',
        'User_Category_Count',
        'Current_Category_encoded',  # ★ 현재 카테고리 포함
        'Previous_Category_encoded'
    ]
    
    # NaN 처리
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features])
    scaled_feature_names = [f"{f}_scaled" for f in numeric_features]
    
    # 6. 최종 데이터 저장
    print("\n[6/6] 데이터 저장")
    
    processed_df = pd.DataFrame(X_scaled, columns=scaled_feature_names)
    processed_df['Next_Category_encoded'] = df['Next_Category_encoded'].values  # Y (타겟)
    processed_df['Next_Category'] = df['Next_Category'].values
    processed_df['Current_Category'] = df['Current_Category'].values
    
    if 'User' in df.columns:
        processed_df['User'] = df['User'].values
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'preprocessed_sequence.csv')
    
    print(f"  - 저장 경로: {output_file}")
    processed_df.to_csv(output_file, index=False)
    
    # 메타정보
    metadata = {
        'total_samples': len(processed_df),
        'num_features': len(numeric_features),
        'feature_names': numeric_features,
        'categories': list(label_encoder.classes_),
        'processing_time': str(datetime.now() - start_time)
    }
    
    metadata_file = os.path.join(output_dir, 'preprocessing_sequence_metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    # 요약
    print("\n" + "="*70)
    print("전처리 완료")
    print("="*70)
    print(f"총 시퀀스: {len(processed_df):,}개")
    print(f"총 특성: {len(numeric_features)}개 (현재 카테고리 포함)")
    print(f"타겟: 다음 카테고리 (6개 클래스)")
    print(f"\n카테고리 분포 (다음 거래):")
    for i, cat in enumerate(label_encoder.classes_):
        count = (processed_df['Next_Category_encoded'] == i).sum()
        print(f"  - {cat}: {count:,}건 ({count/len(processed_df)*100:.2f}%)")
    print(f"\n처리 시간: {datetime.now() - start_time}")
    print("="*70)
    
    return output_file, metadata


if __name__ == '__main__':
    data_file = '02_data/00_raw/credit_card_transactions-ibm_v2.csv'
    output_dir = '02_data/01_processed'
    
    output_file, metadata = preprocess_sequence_pipeline(data_file, output_dir)
    
    print(f"\n✅ 시퀀스 데이터 준비 완료: {output_file}")
    print(f"다음 단계: GPU 학습으로 다음 카테고리 예측")
