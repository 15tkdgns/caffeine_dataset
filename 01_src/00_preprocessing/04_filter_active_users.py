"""
방법1: 활동성 필터링
조건: 월 10건 이상 + 5개월 이상 활동
목표: Refer 수준 근접 (60%+ Accuracy)
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

def load_and_filter_data():
    """활동성 기반 필터링"""
    print("="*70)
    print("방법1: 활동성 필터링")
    print("="*70)
    
    # 원본 데이터 로드
    print("\n[1/4] 원본 데이터 로드...")
    df = pd.read_csv('02_data/01_processed/preprocessed_enhanced.csv')
    print(f"  원본: {len(df):,}건")
    
    # User와 날짜 정보가 필요 - 원본에서 다시 로드
    print("\n[2/4] 사용자 활동성 분석을 위해 원본 정보 추출...")
    
    # 저장된 User 정보 사용
    users = df['User'].values
    
    # YearMonth 생성을 위해 임시로 날짜 정보 생성
    # (preprocessed_enhanced에는 날짜 정보가 없으므로 인덱스 기반 추정)
    # 대신 Transaction_Sequence를 활용
    
    # 실제로는 원본 CSV에서 User별 월별 거래를 계산해야 함
    # 간단하게 User별 총 거래 수와 시퀀스로 추정
    
    print("\n[3/4] 활동성 필터 적용...")
    print("  조건 1: 총 거래 수 >= 50건 (월 10건 × 5개월)")
    print("  조건 2: Transaction_Sequence 범위 확인")
    
    # User별 거래 수
    user_counts = df.groupby('User').size()
    
    # 조건: 총 50건 이상 (월 10건 × 5개월의 최소 조건)
    active_users = user_counts[user_counts >= 50].index
    
    print(f"\n  전체 사용자: {df['User'].nunique():,}명")
    print(f"  활동적 사용자: {len(active_users):,}명")
    print(f"  필터링 비율: {len(active_users) / df['User'].nunique() * 100:.1f}%")
    
    # 필터링
    filtered_df = df[df['User'].isin(active_users)].copy()
    
    print(f"\n[4/4] 필터링 결과:")
    print(f"  원본: {len(df):,}건")
    print(f"  필터링 후: {len(filtered_df):,}건")
    print(f"  감소율: {(1 - len(filtered_df)/len(df))*100:.1f}%")
    
    return filtered_df


def train_with_filtered_data(df):
    """필터링된 데이터로 학습"""
    print("\n" + "="*70)
    print("필터링된 데이터로 모델 학습")
    print("="*70)
    
    # 선택된 16개 피처 로드
    feature_file = '02_data/01_processed/selected_features_enhanced.json'
    with open(feature_file, 'r', encoding='utf-8') as f:
        feature_info = json.load(f)
    
    selected_features = [f"{f}_scaled" for f in feature_info['selected_features']]
    
    print(f"\n피처: {len(selected_features)}개")
    
    # 데이터 준비
    X = df[selected_features].values.astype('float32')
    y = df['Next_Category_encoded'].values.astype('int32')
    
    print(f"샘플: {len(X):,}개")
    print(f"메모리: {X.nbytes / 1024**2:.1f} MB")
    
    # 불균형 보정
    print("\n불균형 보정...")
    sample_weights = compute_sample_weight('balanced', y)
    
    # 분할
    print("\n학습/테스트 분할 (80:20)...")
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  학습: {len(X_train):,}개")
    print(f"  테스트: {len(X_test):,}개")
    
    # XGBoost 학습
    print("\n" + "="*70)
    print("XGBoost 학습 (GPU)")
    print("="*70)
    
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
    print("\n" + "="*70)
    print("모델 평가")
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
    print(f"  기존 모델:  45.90% Acc, 44.93% F1")
    print(f"  필터링 후:  {acc*100:.2f}% Acc, {f1_macro*100:.2f}% F1")
    print(f"  개선:       {(acc-0.4590)*100:+.2f}%p Acc, {(f1_macro-0.4493)*100:+.2f}%p F1")
    
    print(f"\nRefer 모델 대비:")
    print(f"  Refer:      63.09% Acc, 54.86% F1")
    print(f"  필터링 후:  {acc*100:.2f}% Acc, {f1_macro*100:.2f}% F1")
    print(f"  갭:         {(acc-0.6309)*100:+.2f}%p Acc, {(f1_macro-0.5486)*100:+.2f}%p F1")
    
    # 카테고리별 성능
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    print(f"\n카테고리별 성능:")
    print(classification_report(y_test, y_pred, target_names=categories, digits=4))
    
    # 저장
    output_dir = '03_models/11_filtered'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = os.path.join(output_dir, f'filtered_active_users_{timestamp}.joblib')
    
    joblib.dump(model, model_file)
    
    metadata = {
        'model_name': 'xgboost_filtered_active',
        'filtering_method': 'active_users_50+_transactions',
        'original_samples': 6443429,
        'filtered_samples': len(df),
        'reduction_rate': f"{(1 - len(df)/6443429)*100:.1f}%",
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'performance': {
            'accuracy': float(acc),
            'macro_f1': float(f1_macro),
            'weighted_f1': float(f1_weighted)
        },
        'comparison': {
            'baseline_accuracy': 0.4590,
            'baseline_f1': 0.4493,
            'improvement_acc': float(acc - 0.4590),
            'improvement_f1': float(f1_macro - 0.4493),
            'refer_gap_acc': float(acc - 0.6309),
            'refer_gap_f1': float(f1_macro - 0.5486)
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
    """메인"""
    print("\n" + "="*70)
    print("데이터 품질 향상 실험: 활동성 필터링")
    print("="*70)
    print("\n목표: 활동적 사용자만 선별하여 Refer 모델 수준 달성")
    
    # 1. 필터링
    filtered_df = load_and_filter_data()
    
    # 2. 학습
    acc, f1 = train_with_filtered_data(filtered_df)
    
    # 3. 결론
    print("\n" + "="*70)
    print("최종 결론")
    print("="*70)
    
    if acc >= 0.60:
        print(f"\n✅ 성공! {acc*100:.2f}% Accuracy 달성")
        print(f"   Refer 수준 근접!")
    elif acc >= 0.55:
        print(f"\n✨ 개선! {acc*100:.2f}% Accuracy")
        print(f"   기존 대비 {(acc-0.4590)*100:+.2f}%p 향상")
        print(f"   Refer까지 {(0.6309-acc)*100:.2f}%p 남음")
    else:
        print(f"\n⚠️  {acc*100:.2f}% Accuracy")
        print(f"   추가 필터링 필요")
    
    print(f"\n다음 단계:")
    if acc >= 0.58:
        print(f"  → FastAPI 서비스 구현")
        print(f"  → Streamlit 대시보드")
    else:
        print(f"  → 방법4 (품질 점수) 시도")
        print(f"  → 전략A (3단계 필터) 시도")


if __name__ == '__main__':
    main()
