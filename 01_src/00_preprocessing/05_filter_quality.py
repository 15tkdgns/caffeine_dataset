"""
방법4: 품질 점수 기반 필터링 (강력)
- 거래 횟수, 카테고리 다양성, 금액 일관성 종합
- 상위 30%만 선택
- 목표: 60%+ Accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import json
import os
from datetime import datetime


def calculate_quality_score(df):
    """사용자별 품질 점수 계산"""
    print("\n품질 점수 계산 중...")
    
    # 사용 가능한 컬럼 확인
    has_amount = 'Amount_scaled' in df.columns
    has_category = 'Current_Category' in df.columns
    has_sequence = 'Transaction_Sequence_scaled' in df.columns
    
    user_metrics = []
    
    for user in df['User'].unique():
        user_data = df[df['User'] == user]
        
        # 1. 거래 횟수 (40%)
        tx_count = len(user_data)
        tx_score = min(tx_count / 10000, 1.0)
        
        # 2. 카테고리 다양성 (40%)
        if has_category:
            unique_cats = user_data['Current_Category'].nunique()
            diversity_score = unique_cats / 6.0
        else:
            # Next_Category로 대체
            unique_cats = user_data['Next_Category'].nunique()
            diversity_score = unique_cats / 6.0
        
        # 3. 금액 일관성 (10%) - scaled 사용
        if has_amount:
            amounts = user_data['Amount_scaled'].values
            if len(amounts) > 1:
                cv = np.std(amounts) / (abs(np.mean(amounts)) + 1e-10)
                consistency_score = 1.0 / (1.0 + abs(cv))
            else:
                consistency_score = 0.5
        else:
            consistency_score = 0.5
        
        # 4. 활동 지속성 (10%)
        persistence_score = 0.5  # 단순화
        
        # 종합 점수 (거래 횟수와 다양성 중심)
        final_score = (
            tx_score * 0.4 +
            diversity_score * 0.4 +
            consistency_score * 0.1 +
            persistence_score * 0.1
        )
        
        user_metrics.append({
            'User': user,
            'tx_count': tx_count,
            'diversity': unique_cats,
            'consistency': consistency_score,
            'persistence': persistence_score,
            'quality_score': final_score
        })
    
    metrics_df = pd.DataFrame(user_metrics)
    
    print(f"\n품질 점수 통계:")
    print(f"  평균: {metrics_df['quality_score'].mean():.3f}")
    print(f"  중앙값: {metrics_df['quality_score'].median():.3f}")
    print(f"  최대: {metrics_df['quality_score'].max():.3f}")
    print(f"  최소: {metrics_df['quality_score'].min():.3f}")
    
    return metrics_df


def filter_by_quality(df, top_percent=30):
    """품질 점수 상위 N% 선택"""
    print("="*70)
    print(f"품질 점수 기반 필터링 (상위 {top_percent}%)")
    print("="*70)
    
    # 품질 점수 계산
    quality_df = calculate_quality_score(df)
    
    # 상위 N% 선택
    threshold = quality_df['quality_score'].quantile(1 - top_percent/100)
    high_quality_users = quality_df[quality_df['quality_score'] >= threshold]['User'].values
    
    print(f"\n필터링 결과:")
    print(f"  전체 사용자: {len(quality_df)}명")
    print(f"  선택된 사용자: {len(high_quality_users)}명")
    print(f"  품질 점수 임계값: {threshold:.3f}")
    
    # 필터링
    filtered_df = df[df['User'].isin(high_quality_users)].copy()
    
    print(f"\n데이터 크기:")
    print(f"  원본: {len(df):,}건")
    print(f"  필터링 후: {len(filtered_df):,}건")
    print(f"  감소율: {(1 - len(filtered_df)/len(df))*100:.1f}%")
    
    # 선택된 사용자 통계
    selected_quality = quality_df[quality_df['User'].isin(high_quality_users)]
    print(f"\n선택된 사용자 품질:")
    print(f"  평균 거래 수: {selected_quality['tx_count'].mean():.0f}건")
    print(f"  평균 카테고리 수: {selected_quality['diversity'].mean():.1f}개")
    print(f"  평균 품질 점수: {selected_quality['quality_score'].mean():.3f}")
    
    return filtered_df


def train_with_quality_filtered(df):
    """품질 필터링된 데이터로 학습"""
    print("\n" + "="*70)
    print("품질 필터링 데이터로 모델 학습")
    print("="*70)
    
    # 피처 로드
    feature_file = '02_data/01_processed/selected_features_enhanced.json'
    with open(feature_file, 'r', encoding='utf-8') as f:
        feature_info = json.load(f)
    
    selected_features = [f"{f}_scaled" for f in feature_info['selected_features']]
    
    # 데이터 준비
    X = df[selected_features].values.astype('float32')
    y = df['Next_Category_encoded'].values.astype('int32')
    
    print(f"\n피처: {len(selected_features)}개")
    print(f"샘플: {len(X):,}개")
    
    # 불균형 보정
    sample_weights = compute_sample_weight('balanced', y)
    
    # 분할
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # 학습
    model = xgb.XGBClassifier(
        device='cuda',
        tree_method='hist',
        max_depth=10,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("\n학습 시작...")
    start_time = datetime.now()
    model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"학습 완료: {training_time:.1f}초")
    
    # 평가
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n성능:")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1:      {f1_macro:.4f}")
    print(f"  Weighted F1:   {f1_weighted:.4f}")
    
    print(f"\n비교:")
    print(f"  기존:       45.90% Acc")
    print(f"  품질 필터:  {acc*100:.2f}% Acc")
    print(f"  개선:       {(acc-0.4590)*100:+.2f}%p")
    
    print(f"\nRefer 대비:")
    print(f"  Refer:      63.09% Acc")
    print(f"  품질 필터:  {acc*100:.2f}% Acc")
    print(f"  갭:         {(acc-0.6309)*100:+.2f}%p")
    
    # 카테고리별
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    print(f"\n카테고리별 성능:")
    print(classification_report(y_test, y_pred, target_names=categories, digits=4))
    
    # 저장
    output_dir = '03_models/12_quality_filtered'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = os.path.join(output_dir, f'quality_filtered_top30_{timestamp}.joblib')
    joblib.dump(model, model_file)
    
    print(f"\n✅ 모델 저장: {model_file}")
    
    return acc, f1_macro


def main():
    print("="*70)
    print("방법4: 품질 점수 기반 필터링")
    print("="*70)
    
    # 원본 로드
    df = pd.read_csv('02_data/01_processed/preprocessed_enhanced.csv')
    print(f"\n원본: {len(df):,}건")
    
    # 품질 필터링 (상위 30%)
    filtered_df = filter_by_quality(df, top_percent=30)
    
    # 학습
    acc, f1 = train_with_quality_filtered(filtered_df)
    
    # 결론
    print("\n" + "="*70)
    print("최종 결론")
    print("="*70)
    
    if acc >= 0.60:
        print(f"\n🎉 성공! {acc*100:.2f}% Accuracy")
        print(f"   Refer 수준 달성!")
    elif acc >= 0.55:
        print(f"\n✅ 개선! {acc*100:.2f}% Accuracy")
        print(f"   기존 대비 {(acc-0.4590)*100:+.2f}%p 향상")
    else:
        print(f"\n⚠️  {acc*100:.2f}% Accuracy")
print(f"   추가 개선 필요")


if __name__ == '__main__':
    main()
