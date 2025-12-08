"""
전체 데이터셋 전처리 파이프라인 (피처 엔지니어링 포함)
GPU 학습을 위한 고급 전처리 수행
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os
import sys

def get_category_mapping():
    """6개 비즈니스 카테고리 MCC 매핑"""
    mcc_map = {
        # 식료품 (마트, 편의점, 슈퍼마켓)
        5411: '식료품', 5499: '식료품', 5462: '식료품',
        # 주유 (주유소, 충전소)
        5541: '주유', 5542: '주유',
        # 교통 (톨게이트, 대중교통, 택시)
        4121: '교통', 4111: '교통', 4784: '교통', 4131: '교통',
        # 생활 (생활용품, 약국, 드럭스토어)
        5912: '생활', 5200: '생활', 5722: '생활',
        # 외식 (음식점, 카페, 패스트푸드)
        5812: '외식', 5813: '외식', 5814: '외식', 5815: '외식',
        # 쇼핑 (백화점, 온라인, 의류)
        5311: '쇼핑', 5310: '쇼핑', 5300: '쇼핑',
        5651: '쇼핑', 5942: '쇼핑', 5964: '쇼핑', 5945: '쇼핑',
    }
    return mcc_map


def advanced_feature_engineering(df):
    """
    고급 피처 엔지니어링 수행
    
    생성 피처:
    - 시간 관련: Hour, DayOfWeek, IsWeekend, IsNight, IsBusinessHour
    - 금액 관련: Amount_log, Amount_zscore
    - 사용자/카드 통계: User별 거래빈도, 평균금액, 표준편차
    - 시계열 특성: 요일별 패턴, 시간대별 패턴
    """
    print("\n고급 피처 엔지니어링 시작...")
    
    # 1. 날짜/시간 특성
    print("  [1/6] 날짜/시간 특성 생성...")
    df['Transaction_DateTime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str) + '-' + 
        df['Day'].astype(str) + ' ' + 
        df['Time'], 
        errors='coerce'
    )
    
    df['Hour'] = df['Transaction_DateTime'].dt.hour
    df['DayOfWeek'] = df['Transaction_DateTime'].dt.dayofweek
    df['DayOfMonth'] = df['Transaction_DateTime'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsNight'] = df['Hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
    df['IsBusinessHour'] = df['Hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    
    # 2. 금액 특성
    print("  [2/6] 금액 관련 특성 생성...")
    df['Amount'] = df['Amount'].replace({r'\$': '', r',': ''}, regex=True).astype(float)
    
    # 음수나 NaN 처리
    df['Amount'] = df['Amount'].fillna(0).clip(lower=0)
    
    df['Amount_log'] = np.log1p(df['Amount'])  # log(1+x) 변환
    
    # 금액 구간 (NaN 처리)
    df['Amount_bin'] = pd.cut(df['Amount'], 
                               bins=[0, 10, 50, 100, 500, float('inf')],
                               labels=[0, 1, 2, 3, 4],
                               include_lowest=True)
    df['Amount_bin'] = df['Amount_bin'].fillna(0).astype(int)
    
    # 3. 사용자별 통계 특성
    print("  [3/6] 사용자별 통계 특성 생성...")
    user_stats = df.groupby('User')['Amount'].agg([
        ('User_TotalTransactions', 'count'),
        ('User_AvgAmount', 'mean'),
        ('User_StdAmount', 'std'),
        ('User_MaxAmount', 'max'),
        ('User_MinAmount', 'min')
    ]).reset_index()
    
    # 표준편차가 NaN인 경우 0으로 대체
    user_stats['User_StdAmount'] = user_stats['User_StdAmount'].fillna(0)
    
    df = df.merge(user_stats, on='User', how='left')
    
    # 4. 카드별 통계 특성
    print("  [4/6] 카드별 통계 특성 생성...")
    card_stats = df.groupby('Card')['Amount'].agg([
        ('Card_TotalTransactions', 'count'),
        ('Card_AvgAmount', 'mean')
    ]).reset_index()
    
    df = df.merge(card_stats, on='Card', how='left')
    
    # MCC 통계는 데이터 누출이므로 제거!
    # MCC는 타겟 카테고리와 직접 매핑되므로 사용하면 안됨
    
    # 5. 상대적 특성 (현재 거래 vs 사용자 평균)
    print("  [5/5] 상대적 특성 생성...")
    df['Amount_vs_UserAvg'] = df['Amount'] / (df['User_AvgAmount'] + 1e-10)
    df['Amount_vs_CardAvg'] = df['Amount'] / (df['Card_AvgAmount'] + 1e-10)
    
    print(f"피처 엔지니어링 완료. 총 컬럼 수: {len(df.columns)}")
    
    return df


def preprocess_full_pipeline(data_file, output_dir='02_data/01_processed'):
    """
    전체 데이터 전처리 파이프라인
    """
    print("="*70)
    print("전체 데이터셋 전처리 파이프라인 시작")
    print("="*70)
    
    start_time = datetime.now()
    
    # 1. 데이터 로드
    print(f"\n[단계 1/5] 데이터 로드: {data_file}")
    df = pd.read_csv(data_file)
    print(f"  - 전체 데이터: {len(df):,}건")
    print(f"  - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. 카테고리 매핑 및 필터링
    print("\n[단계 2/5] 카테고리 매핑 및 데이터 필터링")
    mcc_map = get_category_mapping()
    df['Category'] = df['MCC'].map(mcc_map)
    
    original_count = len(df)
    df.dropna(subset=['Category'], inplace=True)
    filtered_count = len(df)
    print(f"  - 유효 카테고리 데이터: {filtered_count:,}건")
    print(f"  - 제외된 데이터: {original_count - filtered_count:,}건")
    
    # 3. 피처 엔지니어링
    print("\n[단계 3/5] 피처 엔지니어링")
    df = advanced_feature_engineering(df)
    
    # 4. 특성 선택 및 스케일링
    print("\n[단계 4/5] 특성 선택 및 스케일링")
    
    # 사용할 특성 정의 (MCC 특성 제거 - 데이터 누출 방지)
    numeric_features = [
        'Amount', 'Amount_log', 'Amount_bin',
        'Hour', 'DayOfWeek', 'DayOfMonth',
        'IsWeekend', 'IsNight', 'IsBusinessHour',
        'User_TotalTransactions', 'User_AvgAmount', 'User_StdAmount',
        'User_MaxAmount', 'User_MinAmount',
        'Card_TotalTransactions', 'Card_AvgAmount',
        'Amount_vs_UserAvg', 'Amount_vs_CardAvg'
    ]
    
    # 실제 존재하는 특성만 선택
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    print(f"  - 선택된 특성 수: {len(numeric_features)}")
    print(f"  - 특성 목록: {numeric_features}")
    
    # 타겟 인코딩
    print("\n  타겟 변수 인코딩...")
    label_encoder = LabelEncoder()
    df['Category_encoded'] = label_encoder.fit_transform(df['Category'])
    
    # 특성 스케일링
    print("  특성 스케일링...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features])
    
    # 5. 최종 데이터 저장
    print("\n[단계 5/5] 전처리 데이터 저장")
    
    # 스케일된 특성 DataFrame 생성
    scaled_feature_names = [f"{f}_scaled" for f in numeric_features]
    processed_df = pd.DataFrame(X_scaled, columns=scaled_feature_names)
    
    # 타겟과 식별자 추가
    processed_df['Category_encoded'] = df['Category_encoded'].values
    processed_df['Category'] = df['Category'].values
    
    if 'User' in df.columns:
        processed_df['User'] = df['User'].values
    if 'Card' in df.columns:
        processed_df['Card'] = df['Card'].values
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'preprocessed_full_featured.csv')
    
    print(f"  - 저장 경로: {output_file}")
    processed_df.to_csv(output_file, index=False)
    
    # 메타정보 저장
    metadata = {
        'total_samples': len(processed_df),
        'num_features': len(numeric_features),
        'feature_names': numeric_features,
        'scaled_feature_names': scaled_feature_names,
        'categories': list(label_encoder.classes_),
        'processing_time': str(datetime.now() - start_time)
    }
    
    metadata_file = os.path.join(output_dir, 'preprocessing_metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"  - 메타데이터: {metadata_file}")
    
    # 요약 정보 출력
    print("\n" + "="*70)
    print("전처리 완료 요약")
    print("="*70)
    print(f"총 샘플 수: {len(processed_df):,}건")
    print(f"총 특성 수: {len(numeric_features)}개")
    print(f"타겟 클래스 수: {len(label_encoder.classes_)}개")
    print(f"클래스별 분포:")
    for i, cat in enumerate(label_encoder.classes_):
        count = (processed_df['Category_encoded'] == i).sum()
        print(f"  - {cat}: {count:,}건 ({count/len(processed_df)*100:.2f}%)")
    print(f"\n처리 시간: {datetime.now() - start_time}")
    print("="*70)
    
    return output_file, metadata


if __name__ == '__main__':
    # 경로 설정
    data_file = '02_data/00_raw/credit_card_transactions-ibm_v2.csv'
    output_dir = '02_data/01_processed'
    
    # 전처리 실행
    output_file, metadata = preprocess_full_pipeline(data_file, output_dir)
    
    print(f"\n✅ 전처리된 데이터가 준비되었습니다: {output_file}")
    print(f"다음 단계: GPU 학습 및 그리드 서치를 실행하세요.")
