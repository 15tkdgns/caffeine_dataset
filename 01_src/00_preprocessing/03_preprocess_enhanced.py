"""
개선된 시퀀스 예측 전처리 (Refer 모델 피처 반영)
목표: Accuracy 49% → 60%+
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os

def get_category_mapping():
    """6개 비즈니스 카테고리 MCC 매핑"""
    mcc_map = {
        5411: '식료품', 5499: '식료품', 5462: '식료품',
        5541: '주유', 5542: '주유',
        4121: '교통', 4111: '교통', 4784: '교통', 4131: '교통',
        5912: '생활', 5200: '생활', 5722: '생활',
        5812: '외식', 5813: '외식', 5814: '외식', 5815: '외식',
        5311: '쇼핑', 5310: '쇼핑', 5300: '쇼핑',
        5651: '쇼핑', 5942: '쇼핑', 5964: '쇼핑', 5945: '쇼핑',
    }
    return mcc_map


def add_time_features_enhanced(df):
    """
    Refer 모델 기반 세밀한 시간 피처
    """
    print("  [시간 피처] 세밀한 시간대 구분...")
    
    # 기본 시간 피처
    df['Hour'] = df['Transaction_DateTime'].dt.hour
    df['DayOfWeek'] = df['Transaction_DateTime'].dt.dayofweek
    df['DayOfMonth'] = df['Transaction_DateTime'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # ⭐ Refer 모델 피처: 세밀한 시간대
    df['IsLunchTime'] = ((df['Hour'] >= 11) & (df['Hour'] <= 14)).astype(int)    # 점심
    df['IsEvening'] = ((df['Hour'] >= 18) & (df['Hour'] <= 21)).astype(int)      # 저녁
    df['IsMorningRush'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)).astype(int)    # 출근
    df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)         # 야간
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 18)).astype(int)  # 업무시간
    
    return df


def add_amount_features_enhanced(df):
    """
    Refer 모델 기반 금액 피처
    """
    print("  [금액 피처] 금액 구간 및 변환...")
    
    # 금액 정제
    df['Amount'] = df['Amount'].replace({r'\$': '', r',': ''}, regex=True).astype(float)
    df['Amount'] = df['Amount'].fillna(0).clip(lower=0)
    
    # 기본 변환
    df['Amount_log'] = np.log1p(df['Amount'])
    
    # ⭐ Refer 모델 피처: 금액 구간 (4단계)
    df['AmountBin'] = pd.cut(
        df['Amount'],
        bins=[-float('inf'), 20, 50, 100, float('inf')],
        labels=['저가', '중가', '고가', '초고가']
    )
    df['AmountBin_encoded'] = df['AmountBin'].cat.codes
    
    return df


def add_user_statistics_enhanced(df, train_only=True):
    """
    Refer 모델 기반 사용자 통계 피처
    train_only=True면 현재 시점까지만 사용 (데이터 누출 방지)
    """
    print("  [사용자 통계] 고급 사용자 프로파일링...")
    
    if train_only:
        # 각 행까지의 누적 통계 (데이터 누출 없음)
        df['User_AvgAmount'] = df.groupby('User')['Amount'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(df['Amount'].mean())
        
        df['User_StdAmount'] = df.groupby('User')['Amount'].transform(
            lambda x: x.expanding().std().shift(1)
        ).fillna(0)
        
        df['User_TxCount'] = df.groupby('User').cumcount()
    else:
        # 전체 통계 (Train에서 계산해서 Test에 적용)
        user_stats = df.groupby('User')['Amount'].agg(['mean', 'std', 'count'])
        user_stats.columns = ['User_AvgAmount', 'User_StdAmount', 'User_TxCount']
        user_stats['User_StdAmount'] = user_stats['User_StdAmount'].fillna(0)
        df = df.merge(user_stats, on='User', how='left')
    
    # ⭐ Refer 모델 피처: 사용자 선호 카테고리
    user_fav = df.groupby('User')['Current_Category'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
    )
    df = df.merge(user_fav.rename('User_FavCategory'), on='User', how='left')
    
    return df


def add_user_category_ratio(df):
    """
    ⭐ Refer 모델 핵심 피처: 사용자별 카테고리 이용 비율
    """
    print("  [카테고리 비율] 사용자별 카테고리 선호도...")
    
    # 카테고리 목록
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    
    # 사용자별 카테고리 빈도
    cat_freq = df.groupby(['User', 'Current_Category']).size().unstack(fill_value=0)
    
    # 비율로 변환
    cat_ratio = cat_freq.div(cat_freq.sum(axis=1), axis=0)
    cat_ratio.columns = [f'User_{c}_Ratio' for c in cat_ratio.columns]
    cat_ratio = cat_ratio.reset_index()
    
    df = df.merge(cat_ratio, on='User', how='left')
    
    # 누락값 처리
    for col in df.columns:
        if col.startswith('User_') and col.endswith('_Ratio'):
            df[col] = df[col].fillna(0)
    
    return df


def create_sequence_features(df):
    """시퀀스 피처 생성"""
    print("  [시퀀스 피처] 시계열 정보...")
    
    # 시간순 정렬
    df = df.sort_values(['User', 'Transaction_DateTime']).reset_index(drop=True)
    
    # 현재/이전/다음 카테고리
    df['Current_Category'] = df['Category']
    df['Next_Category'] = df.groupby('User')['Category'].shift(-1)
    df['Previous_Category'] = df.groupby('User')['Category'].shift(1).fillna('NONE')
    
    # 시간 간격
    df['Time_Since_Last'] = df.groupby('User')['Transaction_DateTime'].diff().dt.total_seconds()
    df['Time_Since_Last'] = df['Time_Since_Last'].fillna(0)
    
    # 거래 순서
    df['Transaction_Sequence'] = df.groupby('User').cumcount() + 1
    
    # 마지막 거래 제거
    original_count = len(df)
    df = df.dropna(subset=['Next_Category'])
    print(f"    - 마지막 거래 제거: {original_count - len(df):,}건")
    
    return df


def preprocess_enhanced_pipeline(data_file, output_dir='02_data/01_processed'):
    """
    개선된 전처리 파이프라인
    Refer 모델 피처 반영
    """
    print("="*70)
    print("개선된 시퀀스 예측 전처리 (Refer 피처 반영)")
    print("="*70)
    
    start_time = datetime.now()
    
    # 1. 데이터 로드
    print(f"\n[1/7] 데이터 로드")
    df = pd.read_csv(data_file)
    print(f"  총 {len(df):,}건")
    
    # 2. 카테고리 매핑
    print("\n[2/7] 카테고리 매핑")
    mcc_map = get_category_mapping()
    df['Category'] = df['MCC'].map(mcc_map)
    df = df.dropna(subset=['Category'])
    print(f"  유효 데이터: {len(df):,}건")
    
    # 3. DateTime 파싱
    print("\n[3/7] 날짜/시간 파싱")
    df['Transaction_DateTime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str) + '-' + 
        df['Day'].astype(str) + ' ' + 
        df['Time'], 
        errors='coerce'
    )
    
    # 4. 시퀀스 피처
    print("\n[4/7] 시퀀스 피처 생성")
    df = create_sequence_features(df)
    
    # 5. 시간 피처 (Refer 반영)
    print("\n[5/7] 고급 피처 엔지니어링")
    df = add_time_features_enhanced(df)
    df = add_amount_features_enhanced(df)
    
    # 6. 사용자 통계 (Refer 반영)
    print("\n[6/7] 사용자 프로파일링")
    df = add_user_statistics_enhanced(df, train_only=True)
    df = add_user_category_ratio(df)
    
    # 카테고리별 사용자 빈도
    df['User_Category_Count'] = df.groupby(['User', 'Current_Category']).cumcount() + 1
    
    # 7. 인코딩 및 저장
    print("\n[7/7] 인코딩 및 저장")
    
    # 레이블 인코딩
    label_encoder = LabelEncoder()
    df['Current_Category_encoded'] = label_encoder.fit_transform(df['Current_Category'])
    df['Next_Category_encoded'] = label_encoder.transform(df['Next_Category'])
    df['Previous_Category_encoded'] = df['Previous_Category'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
    )
    df['User_FavCategory_encoded'] = df['User_FavCategory'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
    )
    
    # 사용할 특성 (총 30개+)
    numeric_features = [
        # 금액 (3개)
        'Amount', 'Amount_log', 'AmountBin_encoded',
        # 시간 기본 (3개)
        'Hour', 'DayOfWeek', 'DayOfMonth',
        # 시간 세밀 (5개) ⭐ Refer
        'IsWeekend', 'IsLunchTime', 'IsEvening', 'IsMorningRush', 'IsNight', 'IsBusinessHour',
        # 사용자 기본 (3개)
        'User_AvgAmount', 'User_StdAmount', 'User_TxCount',
        # 시퀀스 (3개)
        'Time_Since_Last', 'Transaction_Sequence', 'User_Category_Count',
        # 카테고리 인코딩 (3개)
        'Current_Category_encoded', 'Previous_Category_encoded', 'User_FavCategory_encoded',
    ]
    
    # 카테고리 비율 추가 ⭐ Refer
    ratio_features = [col for col in df.columns if col.startswith('User_') and col.endswith('_Ratio')]
    numeric_features.extend(ratio_features)
    
    # NaN 처리
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features])
    scaled_feature_names = [f"{f}_scaled" for f in numeric_features]
    
    # 저장
    processed_df = pd.DataFrame(X_scaled, columns=scaled_feature_names)
    processed_df['Next_Category_encoded'] = df['Next_Category_encoded'].values
    processed_df['Next_Category'] = df['Next_Category'].values
    processed_df['Current_Category'] = df['Current_Category'].values
    processed_df['User'] = df['User'].values
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'preprocessed_enhanced.csv')
    
    print(f"  저장 경로: {output_file}")
    processed_df.to_csv(output_file, index=False)
    
    # 메타정보
    metadata = {
        'total_samples': len(processed_df),
        'num_features': len(numeric_features),
        'feature_names': numeric_features,
        'categories': list(label_encoder.classes_),
        'processing_time': str(datetime.now() - start_time),
        'enhancements': [
            'Refer 피처: IsLunchTime, IsEvening, IsMorningRush',
            'Refer 피처: AmountBin (4단계)',
            'Refer 피처: User_FavCategory',
            'Refer 피처: User_{Cat}_Ratio (카테고리별 이용 비율)',
            f'총 피처: {len(numeric_features)}개 (기존 15개 → {len(numeric_features)}개)'
        ]
    }
    
    metadata_file = os.path.join(output_dir, 'preprocessing_enhanced_metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for v in value:
                    f.write(f"  - {v}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # 요약
    print("\n" + "="*70)
    print("전처리 완료 - 개선 사항")
    print("="*70)
    print(f"총 시퀀스: {len(processed_df):,}개")
    print(f"총 피처: {len(numeric_features)}개 (기존 15개 → {len(numeric_features)}개)")
    print(f"\n추가된 Refer 피처:")
    print(f"  1. 세밀한 시간대: IsLunchTime, IsEvening, IsMorningRush")
    print(f"  2. 금액 구간: AmountBin (4단계)")
    print(f"  3. 사용자 선호: User_FavCategory")
    print(f"  4. 카테고리 비율: User_{{Cat}}_Ratio (6개)")
    print(f"\n예상 성능 향상: 49% → 57%+ (Refer 피처 효과)")
    print(f"처리 시간: {datetime.now() - start_time}")
    print("="*70)
    
    return output_file, metadata


if __name__ == '__main__':
    data_file = '02_data/00_raw/credit_card_transactions-ibm_v2.csv'
    output_dir = '02_data/01_processed'
    
    output_file, metadata = preprocess_enhanced_pipeline(data_file, output_dir)
    
    print(f"\n✅ 개선된 전처리 완료: {output_file}")
    print(f"다음 단계: GPU 학습 (성능 향상 기대!)")
