"""
원본 없이 활동성 필터링
Transaction_Sequence 범위를 활용한 월별 활동 추정
조건: 사용자당 거래 수 기반 (월 10건 × 5개월 = 최소 50건 이상, 실제로는 더 엄격하게)
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


def filter_active_users_strict(df, min_transactions=200):
    """
    엄격한 활동성 필터링
    - 사용자당 최소 거래 수
    - 카테고리 다양성 (최소 4개 카테고리)
    """
    print("="*70)
    print("엄격한 활동성 필터링")
    print("="*70)
    
    print(f"\n조건:")
    print(f"  1. 사용자당 최소 {min_transactions}건 거래")
    print(f"  2. 최소 4개 이상 카테고리 사용")
    
    # 사용자별 통계
    user_stats = df.groupby('User').agg({
        'Next_Category': ['count', 'nunique']
    })
    user_stats.columns = ['tx_count', 'cat_count']
    
    # 조건 적용
    active_users = user_stats[
        (user_stats['tx_count'] >= min_transactions) &
        (user_stats['cat_count'] >= 4)
    ].index
    
    print(f"\n필터링 결과:")
    print(f"  전체 사용자: {len(user_stats)}명")
    print(f"  활동적 사용자: {len(active_users)}명 ({len(active_users)/len(user_stats)*100:.1f}%)")
    
    # 선택된 사용자 통계
    selected_stats = user_stats.loc[active_users]
    print(f"\n선택된 사용자 특징:")
    print(f"  평균 거래 수: {selected_stats['tx_count'].mean():.0f}건")
    print(f"  중앙값 거래 수: {selected_stats['tx_count'].median():.0f}건")
    print(f"  평균 카테고리 수: {selected_stats['cat_count'].mean():.1f}개")
    
    # 필터링
    filtered_df = df[df['User'].isin(active_users)].copy()
    
    print(f"\n데이터 변화:")
    print(f"  원본: {len(df):,}건")
    print(f"  필터링: {len(filtered_df):,}건")
    print(f"  감소율: {(1 - len(filtered_df)/len(df))*100:.1f}%")
    
    return filtered_df


def train_with_filtered(df):
    """필터링된 데이터로 학습"""
    print("\n" + "="*70)
    print("필터링된 데이터로 XGBoost 학습")
    print("="*70)
    
    # 피처 로드
    feature_file = '02_data/01_processed/selected_features_enhanced.json'
    with open(feature_file, 'r') as f:
        feature_info = json.load(f)
    
    selected_features = [f"{f}_scaled" for f in feature_info['selected_features']]
    
    X = df[selected_features].values.astype('float32')
    y = df['Next_Category_encoded'].values.astype('int32')
    
    print(f"\n데이터:")
    print(f"  샘플: {len(X):,}개")
    print(f"  피처: {len(selected_features)}개")
    print(f"  메모리: {X.nbytes / 1024**2:.1f} MB")
    
    # 불균형 보정
    sample_weights = compute_sample_weight('balanced', y)
    
    # 분할
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n분할:")
    print(f"  학습: {len(X_train):,}개")
    print(f"  테스트: {len(X_test):,}개")
    
    # 학습
    print("\nXGBoost 학습 (GPU)...")
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
    
    start_time = datetime.now()
    model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"학습 완료: {training_time:.1f}초")
    
    # 평가
    print("\n" + "="*70)
    print("성능 평가")
    print("="*70)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n성능:")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1:      {f1_macro:.4f}")
    print(f"  Weighted F1:   {f1_weighted:.4f}")
    
    # 비교
    print(f"\n비교:")
    print(f"  기존 (전체):      45.90% Acc, 44.93% F1")
    print(f"  품질 필터 (30%):  48.00% Acc, 45.96% F1")
    print(f"  활동성 필터:      {acc*100:.2f}% Acc, {f1_macro*100:.2f}% F1")
    
    print(f"\nRefer 대비:")
    print(f"  Refer:         63.09% Acc, 54.86% F1")
    print(f"  활동성 필터:   {acc*100:.2f}% Acc, {f1_macro*100:.2f}% F1")
    print(f"  갭:            {(acc-0.6309)*100:+.2f}%p Acc, {(f1_macro-0.5486)*100:+.2f}%p F1")
    
    # 카테고리별
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    print(f"\n카테고리별 성능:")
    report = classification_report(y_test, y_pred, target_names=categories, digits=4, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=categories, digits=4))
    
    # 생활 카테고리 강조
    life_f1 = report['생활']['f1-score']
    print(f"\n🔍 생활 카테고리 F1: {life_f1:.4f}")
    print(f"   이전 대비: {(life_f1 - 0.2654)*100:+.2f}%p")
    
    # 저장
    output_dir = '03_models/13_strict_active'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = os.path.join(output_dir, f'strict_active_{timestamp}.joblib')
    joblib.dump(model, model_file)
    
    metadata = {
        'model_name': 'xgboost_strict_active',
        'filtering': 'min_200_tx_and_4_categories',
        'original_samples': 6443429,
        'filtered_samples': len(df),
        'reduction_rate': f"{(1 - len(df)/6443429)*100:.1f}%",
        'performance': {
            'accuracy': float(acc),
            'macro_f1': float(f1_macro),
            'weighted_f1': float(f1_weighted),
            'life_category_f1': float(life_f1)
        },
        'comparison': {
            'vs_baseline_acc': float(acc - 0.4590),
            'vs_quality_filter_acc': float(acc - 0.4800),
            'vs_refer_acc': float(acc - 0.6309)
        },
        'training_time': training_time,
        'created_at': datetime.now().isoformat()
    }
    
    metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 모델 저장: {model_file}")
    print(f"✅ 메타데이터: {metadata_file}")
    
    return acc, f1_macro


def main():
    print("="*70)
    print("엄격한 활동성 필터링 + 학습")
    print("="*70)
    print("\n목표: 고품질 활동 사용자만 선별")
    
    # 데이터 로드
    print("\n데이터 로드...")
    df = pd.read_csv('02_data/01_processed/preprocessed_enhanced.csv')
    print(f"원본: {len(df):,}건")
    
    # 필터링 (최소 200건)
    filtered_df = filter_active_users_strict(df, min_transactions=200)
    
    # 학습
    acc, f1 = train_with_filtered(filtered_df)
    
    # 결론
    print("\n" + "="*70)
    print("최종 결론")
    print("="*70)
    
    if acc >= 0.60:
        print(f"\n🎉 대성공! {acc*100:.2f}% Accuracy")
        print(f"   Refer 수준 달성!")
    elif acc >= 0.55:
        print(f"\n✅ 성공! {acc*100:.2f}% Accuracy")
        print(f"   Refer까지 {(0.6309-acc)*100:.2f}%p")
    elif acc >= 0.50:
        print(f"\n✨ 개선! {acc*100:.2f}% Accuracy")
        print(f"   기존 대비 {(acc-0.4590)*100:+.2f}%p")
    else:
        print(f"\n⚠️  {acc*100:.2f}% Accuracy")
        print(f"   추가 개선 필요")


if __name__ == '__main__':
    main()
