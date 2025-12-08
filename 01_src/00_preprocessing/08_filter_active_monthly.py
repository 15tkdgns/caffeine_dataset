"""
활동성 기반 사용자 필터링
조건: 월 10건 이상 거래가 5개월 이상 있는 사용자만 선택
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def filter_active_users(input_file, output_dir='02_data/01_processed'):
    """
    월 10건 이상 거래가 5개월 이상인 활동적인 사용자만 필터링
    """
    print("="*70)
    print("활동성 기반 사용자 필터링")
    print("조건: 월 10건 이상 거래 & 5개월 이상 활동")
    print("="*70)
    
    start_time = datetime.now()
    
    # 데이터 로드
    print(f"\n[1/5] 데이터 로드")
    print(f"  파일: {input_file}")
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"  총 샘플: {original_count:,}개")
    
    # Next_Category 컬럼 확인
    if 'Next_Category' not in df.columns:
        print("\n⚠️ Next_Category 컬럼이 없습니다. 원본 데이터부터 전처리 필요")
        return None
    
    # 사용자별 통계 확인
    unique_users = df['User'].nunique()
    print(f"  총 사용자: {unique_users:,}명")
    
    # 날짜 정보 복원 (Year, Month가 있다고 가정)
    # preprocessed_enhanced.csv에는 날짜 정보가 없으므로 원본에서 가져와야 함
    print("\n[2/5] 월별 거래 수 계산")
    
    # 원본 데이터에서 Year, Month 정보 가져오기
    raw_file = '02_data/00_raw/credit_card_transactions-ibm_v2.csv'
    if os.path.exists(raw_file):
        print(f"  원본 데이터 로드: {raw_file}")
        df_raw = pd.read_csv(raw_file, usecols=['User', 'Year', 'Month'])
        
        # User 정보로 매칭 (순서 기반)
        # 주의: 전처리 과정에서 순서가 바뀌었을 수 있으므로 User 기준으로 재매칭 필요
        print("  ⚠️ 경고: 전처리된 데이터와 원본 데이터의 순서가 다를 수 있습니다")
        print("  User 정보만 사용하여 월별 집계")
        
        # User별 YearMonth 생성
        df_raw['YearMonth'] = df_raw['Year'].astype(str) + '-' + df_raw['Month'].astype(str).str.zfill(2)
        
        # 사용자별 월별 거래 수
        user_monthly = df_raw.groupby(['User', 'YearMonth']).size().reset_index(name='MonthlyTxCount')
        
        # 월 10건 이상인 달만 선택
        active_months = user_monthly[user_monthly['MonthlyTxCount'] >= 10]
        
        # 사용자별 활동 월 수
        user_active_months = active_months.groupby('User').size().reset_index(name='ActiveMonths')
        
        print(f"\n[3/5] 활동성 분석")
        print(f"  월 10건 이상 거래한 적이 있는 사용자: {len(user_active_months):,}명")
        
        # 5개월 이상 활동한 사용자 선택
        quality_users = user_active_months[user_active_months['ActiveMonths'] >= 5]['User'].values
        
        print(f"\n[4/5] 필터링 적용")
        print(f"  5개월 이상 활동 사용자: {len(quality_users):,}명")
        print(f"  필터링 비율: {len(quality_users)/unique_users*100:.1f}%")
        
        # 필터링
        df_filtered = df[df['User'].isin(quality_users)].copy()
        filtered_count = len(df_filtered)
        
        print(f"\n필터링 결과:")
        print(f"  원본 샘플: {original_count:,}개")
        print(f"  필터링 후: {filtered_count:,}개")
        print(f"  제거된 샘플: {original_count - filtered_count:,}개 ({(1-filtered_count/original_count)*100:.1f}%)")
        print(f"  유지된 사용자: {len(quality_users):,}명 / {unique_users:,}명")
        
        # 저장
        print(f"\n[5/5] 저장")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'preprocessed_filtered_monthly.csv')
        df_filtered.to_csv(output_file, index=False)
        print(f"  저장 경로: {output_file}")
        
        # 메타데이터 저장
        metadata = {
            'original_samples': original_count,
            'filtered_samples': filtered_count,
            'reduction_rate': f"{(1-filtered_count/original_count)*100:.1f}%",
            'original_users': unique_users,
            'filtered_users': len(quality_users),
            'filter_condition': '월 10건 이상 거래 & 5개월 이상 활동',
            'processing_time': str(datetime.now() - start_time)
        }
        
        metadata_file = os.path.join(output_dir, 'filtering_monthly_metadata.txt')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"  메타데이터: {metadata_file}")
        print(f"\n처리 시간: {datetime.now() - start_time}")
        print("="*70)
        print("✅ 필터링 완료!")
        print("="*70)
        
        return output_file
    else:
        print(f"\n❌ 원본 데이터를 찾을 수 없습니다: {raw_file}")
        print("대안: 전처리 과정에 날짜 정보를 포함시켜야 합니다.")
        return None


if __name__ == '__main__':
    input_file = '02_data/01_processed/preprocessed_enhanced.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ 파일 없음: {input_file}")
        print("먼저 전처리를 실행하세요:")
        print("  python3 01_src/00_preprocessing/03_preprocess_enhanced.py")
    else:
        output_file = filter_active_users(input_file)
        if output_file:
            print(f"\n✅ 다음 단계: GPU 모델 학습")
            print(f"  python3 01_src/01_training/10_compare_models.py --data {output_file}")
