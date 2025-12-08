"""
선택된 16개 핵심 피처로 최종 모델 학습
- Refer 피처 포함
- 불균형 보정
- GPU 가속
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

# 선택된 피처 로드
def load_selected_features():
    """피처 셀렉션 결과 로드"""
    feature_file = '02_data/01_processed/selected_features_enhanced.json'
    with open(feature_file, 'r', encoding='utf-8') as f:
        feature_info = json.load(f)
    
    selected_features = [f"{f}_scaled" for f in feature_info['selected_features']]
    
    print(f"선택된 피처: {len(selected_features)}개")
    for i, feat in enumerate(feature_info['selected_features'], 1):
        print(f"  {i:2d}. {feat}")
    
    return selected_features


def load_data(file_path, selected_features):
    """데이터 로드 (선택된 피처만)"""
    print(f"\n데이터 로드: {file_path}")
    df = pd.read_csv(file_path)
    
    # 선택된 피처만 추출
    X = df[selected_features].values.astype('float32')
    y = df['Next_Category_encoded'].values
    
    print(f"  샘플: {len(X):,}개")
    print(f"  피처: {len(selected_features)}개")
    print(f"  메모리: {X.nbytes / 1024**2:.1f} MB")
    
    return X, y


def train_final_model(X_train, y_train, X_test, y_test, sample_weights_train):
    """최종 모델 학습"""
    print("\n" + "="*70)
    print("최종 모델 학습 (XGBoost GPU)")
    print("="*70)
    
    # 최적 파라미터 (그리드 서치 + 추가 튜닝)
    best_params = {
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
    }
    
    print(f"하이퍼파라미터:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    model = xgb.XGBClassifier(
        device='cuda',
        tree_method='hist',
        random_state=42,
        eval_metric='mlogloss',
        **best_params
    )
    
    print(f"\n학습 시작...")
    start_time = datetime.now()
    
    # 불균형 보정 적용
    model.fit(X_train, y_train, sample_weight=sample_weights_train, verbose=False)
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"학습 완료: {training_time:.1f}초 ({training_time/60:.1f}분)")
    
    return model, training_time


def evaluate_model(model, X_test, y_test):
    """모델 평가"""
    print("\n" + "="*70)
    print("모델 평가")
    print("="*70)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n전체 성능:")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1:      {f1_macro:.4f}")
    print(f"  Weighted F1:   {f1_weighted:.4f}")
    
    # Refer 모델과 비교
    refer_acc = 0.6309
    refer_f1 = 0.5486
    
    print(f"\nRefer 모델 대비:")
    print(f"  Accuracy:  {acc:.4f} vs {refer_acc:.4f} ({(acc-refer_acc)*100:+.2f}%p)")
    print(f"  Macro F1:  {f1_macro:.4f} vs {refer_f1:.4f} ({(f1_macro-refer_f1)*100:+.2f}%p)")
    
    # 카테고리별 성능
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    print(f"\n카테고리별 성능:")
    print(classification_report(y_test, y_pred, target_names=categories, digits=4))
    
    return {
        'accuracy': acc,
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted
    }


def save_model(model, metadata, output_dir='03_models/08_final'):
    """최종 모델 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 모델 저장
    model_file = os.path.join(output_dir, f'final_model_{timestamp}.joblib')
    joblib.dump(model, model_file)
    print(f"\n✅ 모델 저장: {model_file}")
    
    # 메타데이터 저장
    metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ 메타데이터: {metadata_file}")
    
    return model_file, metadata_file


def main():
    """메인"""
    print("="*70)
    print("최종 모델 학습 (선택된 16개 피처)")
    print("="*70)
    
    # 1. 선택된 피처 로드
    selected_features = load_selected_features()
    
    # 2. 데이터 로드
    data_file = '02_data/01_processed/preprocessed_enhanced.csv'
    X, y = load_data(data_file, selected_features)
    
    # 3. 불균형 보정
    print("\n불균형 보정 적용...")
    sample_weights = compute_sample_weight('balanced', y)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"클래스별 가중치:")
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    for cls, count, cat in zip(unique_classes, class_counts, categories):
        weight = sample_weights[y == cls][0]
        print(f"  {cat:6s}: {count:,}건 → 가중치 {weight:.3f}")
    
    # 4. 학습/테스트 분할
    print(f"\n학습/테스트 분할 (80:20)...")
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  학습: {len(X_train):,}개")
    print(f"  테스트: {len(X_test):,}개")
    
    # 5. 모델 학습
    model, training_time = train_final_model(X_train, y_train, X_test, y_test, sw_train)
    
    # 6. 모델 평가
    performance = evaluate_model(model, X_test, y_test)
    
    # 7. 모델 저장
    metadata = {
        'model_name': 'xgboost_final_16features',
        'model_version': 'v1.0',
        'num_features': len(selected_features),
        'selected_features': [f.replace('_scaled', '') for f in selected_features],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'training_time_seconds': training_time,
        'performance': performance,
        'refer_comparison': {
            'refer_accuracy': 0.6309,
            'refer_macro_f1': 0.5486,
            'accuracy_gap': performance['accuracy'] - 0.6309,
            'f1_gap': performance['macro_f1'] - 0.5486
        },
        'enhancements': [
            'Refer 피처 6개 (User_*_Ratio)',
            '불균형 보정 (sample_weight)',
            'GPU 가속 (CUDA)',
            '피처 셀렉션 (27개 → 16개)',
            'Macro F1 최적화'
        ],
        'created_at': datetime.now().isoformat()
    }
    
    save_model(model, metadata)
    
    # 8. 최종 요약
    print("\n" + "="*70)
    print("최종 결과 요약")
    print("="*70)
    print(f"\n📊 성능:")
    print(f"  Accuracy:    {performance['accuracy']:.4f} ({performance['accuracy']*100:.2f}%)")
    print(f"  Macro F1:    {performance['macro_f1']:.4f}")
    print(f"  Weighted F1: {performance['weighted_f1']:.4f}")
    
    print(f"\n📈 Refer 모델 대비:")
    print(f"  Accuracy 갭: {(performance['accuracy']-0.6309)*100:+.2f}%p")
    print(f"  Macro F1 갭: {(performance['macro_f1']-0.5486)*100:+.2f}%p")
    
    print(f"\n⚡ 효율:")
    print(f"  피처 수: 27개 → 16개 (40% 절감)")
    print(f"  학습 시간: {training_time:.1f}초")
    print(f"  메모리: {X.nbytes / 1024**2:.1f} MB")
    
    print(f"\n🎯 다음 단계:")
    print(f"  1. 이 모델로 FastAPI 서비스 구현")
    print(f"  2. Streamlit 대시보드 연동")
    print(f"  3. 추가 성능 개선 (앙상블, 파인튜닝)")
    
    print("\n" + "="*70)
    print("학습 완료!")
    print("="*70)


if __name__ == '__main__':
    main()
