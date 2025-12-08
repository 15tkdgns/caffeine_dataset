"""
SMOTE 데이터 증강 모듈
클래스 불균형 해결을 위한 SMOTE (Synthetic Minority Over-sampling Technique) 적용
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from datetime import datetime
import os


def apply_smote_augmentation(input_file, output_dir='02_data/01_processed', 
                             sampling_strategy='auto', k_neighbors=5):
    """
    SMOTE를 사용하여 클래스 불균형 해결
    
    Parameters:
    -----------
    input_file : str
        입력 데이터 파일 경로
    output_dir : str
        출력 디렉토리
    sampling_strategy : str or dict
        'auto': 소수 클래스를 다수 클래스와 같은 수준으로
        dict: 각 클래스별 샘플 수 지정
    k_neighbors : int
        SMOTE에서 사용할 이웃 수
    """
    print("="*70)
    print("SMOTE 데이터 증강")
    print("="*70)
    
    start_time = datetime.now()
    
    # 데이터 로드
    print(f"\n[1/5] 데이터 로드")
    print(f"  파일: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  총 샘플: {len(df):,}개")
    
    # 피처와 타겟 분리
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    target_col = 'Next_Category_encoded'
    
    if not feature_cols:
        print("❌ 스케일링된 피처가 없습니다")
        return None
    
    if target_col not in df.columns:
        print(f"❌ 타겟 컬럼 '{target_col}'이 없습니다")
        return None
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 클래스 분포 확인
    print(f"\n[2/5] 클래스 분포 분석 (증강 전)")
    print("="*60)
    class_counts = Counter(y)
    total = len(y)
    
    category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / total * 100
        cat_name = category_names[class_id] if class_id < len(category_names) else f'Class_{class_id}'
        print(f"  {cat_name:8s} (ID {class_id}): {count:10,}개 ({percentage:5.2f}%)")
    
    # 가장 많은 클래스와 적은 클래스
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)
    imbalance_ratio = class_counts[max_class] / class_counts[min_class]
    
    print(f"\n  불균형 비율: {imbalance_ratio:.2f}:1")
    print(f"  (가장 많은 클래스 / 가장 적은 클래스)")
    
    # SMOTE 적용
    print(f"\n[3/5] SMOTE 증강 적용")
    print(f"  전략: {sampling_strategy}")
    print(f"  k_neighbors: {k_neighbors}")
    
    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=42,
            n_jobs=-1
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"  ✅ SMOTE 증강 완료")
        
    except Exception as e:
        print(f"  ❌ SMOTE 증강 실패: {e}")
        return None
    
    # 증강 후 클래스 분포
    print(f"\n[4/5] 클래스 분포 분석 (증강 후)")
    print("="*60)
    class_counts_after = Counter(y_resampled)
    total_after = len(y_resampled)
    
    for class_id in sorted(class_counts_after.keys()):
        count_before = class_counts.get(class_id, 0)
        count_after = class_counts_after[class_id]
        increase = count_after - count_before
        percentage = count_after / total_after * 100
        
        cat_name = category_names[class_id] if class_id < len(category_names) else f'Class_{class_id}'
        print(f"  {cat_name:8s} (ID {class_id}): {count_after:10,}개 ({percentage:5.2f}%) [+{increase:,}]")
    
    print(f"\n  총 샘플 증가: {len(df):,} → {len(X_resampled):,} (+{len(X_resampled)-len(df):,})")
    print(f"  증가율: {(len(X_resampled)/len(df)-1)*100:.1f}%")
    
    # 데이터프레임 재구성
    print(f"\n[5/5] 증강 데이터 저장")
    
    # 증강된 데이터를 DataFrame으로 변환
    df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
    df_resampled[target_col] = y_resampled
    
    # 추가 컬럼 (Next_Category, Current_Category 등)은 예측 불가능하므로 타겟만 유지
    # 원본에 있던 메타 컬럼은 증강 데이터에서는 의미 없음
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'preprocessed_smote_augmented.csv')
    df_resampled.to_csv(output_file, index=False)
    
    print(f"  저장 경로: {output_file}")
    
    # 메타데이터 저장
    metadata = {
        'original_samples': len(df),
        'augmented_samples': len(df_resampled),
        'increase': len(df_resampled) - len(df),
        'increase_rate': f"{(len(df_resampled)/len(df)-1)*100:.1f}%",
        'smote_strategy': str(sampling_strategy),
        'k_neighbors': k_neighbors,
        'class_distribution_before': dict(class_counts),
        'class_distribution_after': dict(class_counts_after),
        'imbalance_ratio_before': float(imbalance_ratio),
        'processing_time': str(datetime.now() - start_time)
    }
    
    metadata_file = os.path.join(output_dir, 'smote_metadata.txt')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"SMOTE 데이터 증강 메타데이터\n")
        f.write(f"{'='*60}\n\n")
        
        for key, value in metadata.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    cat_name = category_names[k] if k < len(category_names) else f'Class_{k}'
                    f.write(f"  {cat_name} (ID {k}): {v:,}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"  메타데이터: {metadata_file}")
    print(f"\n처리 시간: {datetime.now() - start_time}")
    print("="*70)
    print("✅ SMOTE 증강 완료!")
    print("="*70)
    
    return output_file, metadata


def main():
    """메인 실행"""
    # 입력 파일 우선순위
    candidates = [
        '02_data/01_processed/preprocessed_filtered_monthly.csv',
        '02_data/01_processed/preprocessed_enhanced.csv'
    ]
    
    input_file = None
    for candidate in candidates:
        if os.path.exists(candidate):
            input_file = candidate
            print(f"✅ 입력 파일: {candidate}")
            break
    
    if not input_file:
        print("❌ 입력 데이터를 찾을 수 없습니다")
        print("다음 중 하나를 실행하세요:")
        print("  1. python3 01_src/00_preprocessing/03_preprocess_enhanced.py")
        print("  2. python3 01_src/00_preprocessing/08_filter_active_monthly.py")
        return
    
    # SMOTE 적용
    output_file, metadata = apply_smote_augmentation(
        input_file,
        sampling_strategy='auto',  # 모든 소수 클래스를 다수 클래스 수준으로
        k_neighbors=5
    )
    
    if output_file:
        print(f"\n✅ 다음 단계: GPU 모델 학습")
        print(f"  python3 01_src/01_training/10_compare_gpu_models.py")


if __name__ == '__main__':
    main()
