import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os

def get_new_category_mapping():
    """
    category_mapping.py에 정의된 6개 카테고리를 기반으로 MCC 매핑을 생성합니다.
    '기타'에 해당하는 MCC는 매핑하지 않아 필터링 대상으로 삼습니다.
    """
    print("6개 신규 카테고리에 대한 MCC 매핑을 생성합니다...")
    # 비즈니스 의미가 있는 6개 카테고리 중심
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
    print("MCC 매핑 생성 완료.")
    return mcc_map

def preprocess_data_v2(data_file, mcc_to_label, output_dir='../../02_data/01_processed'):
    """
    신규 카테고리 체계를 적용하여 데이터를 전처리하고 저장합니다.
    매핑되지 않는 '기타' 데이터는 필터링합니다.
    """
    print(f"\n'{data_file}' 파일에서 전체 데이터를 로드하고 신규 전처리를 시작합니다...")
    
    # 1. 데이터 로드
    df = pd.read_csv(data_file)
    print(f"전체 데이터 {len(df)}건 로드 완료.")

    # 2. 신규 카테고리 매핑 적용
    print("  - 신규 카테고리 매핑 적용 중...")
    df['Category'] = df['MCC'].map(mcc_to_label)
    
    # 3. '기타' 데이터 필터링 (매핑되지 않은 데이터 제외)
    original_count = len(df)
    df.dropna(subset=['Category'], inplace=True)
    filtered_count = len(df)
    print(f"  - 비즈니스 의미 없는 데이터 필터링 완료: {original_count - filtered_count}건 제외.")

    # 4. 'Amount' 컬럼 전처리
    print("  - 'Amount' 컬럼을 숫자로 변환 중...")
    df['Amount'] = df['Amount'].replace({r'\$': '', r',': ''}, regex=True).astype(float)
    
    # 5. 날짜/시간 특성 생성
    print("  - 날짜/시간 관련 특성 생성 중 (Transaction_DateTime, Is_Weekend)...")
    df['Transaction_DateTime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str) + '-' + 
        df['Day'].astype(str) + ' ' + 
        df['Time'], errors='coerce'
    )
    df['Is_Weekend'] = df['Transaction_DateTime'].dt.dayofweek.isin([5, 6]).fillna(False)
    
    # 6. 특성 및 타겟 선택
    features = ['Amount', 'Is_Weekend']
    target = 'Category'
    
    X = df[features].astype('float32')
    y = df[target]
    
    # 7. 라벨 인코딩
    print("  - 타겟 변수 'Category'를 숫자로 인코딩 중...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 8. 수치형 특성 스케일링
    print("  - 수치형 특성 'Amount' 스케일링 중...")
    scaler = StandardScaler()
    X_scaled_amount = scaler.fit_transform(X[['Amount']])
    X_scaled = np.hstack((X_scaled_amount, X[['Is_Weekend']].values))
    processed_feature_cols = ['Amount_scaled', 'Is_Weekend']

    # 9. 전처리된 데이터를 DataFrame으로 재구성
    processed_df = pd.DataFrame(X_scaled, columns=processed_feature_cols)
    processed_df['Category_encoded'] = y_encoded
    
    # 필요 시 식별자 컬럼 추가
    for col in ['User', 'Card']:
        if col in df.columns:
            processed_df[col] = df[col].values

    # 10. 전처리된 데이터 저장
    output_file = os.path.join(output_dir, 'preprocessed_transactions_v2.csv')
    print(f"  - 전처리된 데이터를 '{output_file}'에 저장 중...")
    processed_df.to_csv(output_file, index=False)
    
    print("\n신규 데이터 전처리 완료.")
    print(f"  - 최종 데이터 수: {filtered_count}건")
    print(f"  - 저장된 파일: '{output_file}'")
    
    return output_file, label_encoder

def main():
    """
    메인 함수: 신규 카테고리 기반 데이터 전처리 파이프라인을 실행합니다.
    """
    print("="*50)
    print("신규 카테고리 기반 데이터 전처리 스크립트 시작")
    print("="*50)
    
    data_file = '../../02_data/00_raw/credit_card_transactions-ibm_v2.csv'
    output_dir = '../../02_data/01_processed'
    
    mcc_map = get_new_category_mapping()
    processed_file_path, label_encoder = preprocess_data_v2(data_file, mcc_map, output_dir)
    
    print("\n데이터 전처리 파이프라인 완료.")
    print(f"다음 단계에서는 '{processed_file_path}' 파일을 사용하여 머신러닝 모델을 학습할 수 있습니다.")
    print(f"라벨 인코더 클래스 정보: {list(label_encoder.classes_)}")
    print("\n새로운 라벨 분포:")
    print(pd.read_csv(processed_file_path)['Category_encoded'].value_counts())


    print("="*50)
    print("스크립트 실행 완료")
    print("="*50)

if __name__ == '__main__':
    # 스크립트 이름을 train_gpu.py에서 preprocess.py로 변경하는 것을 고려해볼 수 있습니다.
    # 여기서는 기존 파일명을 그대로 사용합니다.
    main()