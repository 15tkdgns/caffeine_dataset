"""
피처 셀렉션 (Feature Selection)
목표: 30개 → 15-20개 핵심 피처만 선택
방법: XGBoost Feature Importance 기반
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import json

def load_data(file_path):
    """데이터 로드"""
    print(f"\n데이터 로드: {file_path}")
    df = pd.read_csv(file_path)
    
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    target_col = 'Next_Category_encoded'
    
    X = df[feature_cols].values.astype('float32')
    y = df[target_col].values
    
    print(f"  전체 피처: {len(feature_cols)}개")
    print(f"  샘플 수: {len(X):,}개")
    
    return X, y, feature_cols


def train_baseline_model(X_train, y_train, X_test, y_test):
    """베이스라인 모델 학습 (피처 중요도 계산용)"""
    print("\n베이스라인 모델 학습 (전체 피처, CPU)...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,  # 빠른 계산용
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1  # CPU 병렬
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    
    return model


def analyze_feature_importance(model, feature_names, top_k=20):
    """피처 중요도 분석"""
    print(f"\n피처 중요도 분석 (Top {top_k})...")
    
    # 중요도 추출
    importances = model.feature_importances_
    
    # DataFrame 생성
    importance_df = pd.DataFrame({
        'Feature': [f.replace('_scaled', '') for f in feature_names],
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Top K
    top_features = importance_df.head(top_k)
    
    print(f"\n상위 {top_k}개 피처:")
    print("="*60)
    for idx, row in top_features.iterrows():
        print(f"{row.name+1:2d}. {row['Feature']:35s} {row['Importance']*100:6.2f}%")
    
    # 누적 중요도
    cumsum = top_features['Importance'].cumsum()
    print(f"\n상위 {top_k}개 누적 중요도: {cumsum.iloc[-1]*100:.2f}%")
    
    return importance_df, top_features


def select_features_by_importance(importance_df, threshold=0.01, min_features=15, max_features=20):
    """
    피처 선택 전략
    1. 중요도 threshold 이상
    2. 최소 min_features개
    3. 최대 max_features개
    """
    print(f"\n피처 선택 (threshold={threshold*100}%)...")
    
    # threshold 이상 피처
    selected = importance_df[importance_df['Importance'] >= threshold]
    
    # 최소/최대 제약
    if len(selected) < min_features:
        print(f"  threshold 미달 → 상위 {min_features}개 강제 선택")
        selected = importance_df.head(min_features)
    elif len(selected) > max_features:
        print(f"  threshold 초과 → 상위 {max_features}개로 제한")
        selected = importance_df.head(max_features)
    
    selected_features = selected['Feature'].tolist()
    selected_features_scaled = [f"{f}_scaled" for f in selected_features]
    
    print(f"\n✅ 선택된 피처: {len(selected_features)}개")
    print("="*60)
    for idx, feat in enumerate(selected_features, 1):
        imp = selected.loc[selected['Feature'] == feat, 'Importance'].values[0]
        print(f"{idx:2d}. {feat:35s} {imp*100:6.2f}%")
    
    return selected_features_scaled


def evaluate_selected_features(X_train, y_train, X_test, y_test, 
                                 feature_cols, selected_feature_cols):
    """선택된 피처로 성능 평가"""
    print("\n선택된 피처로 성능 평가 (CPU)...")
    
    # 인덱스 추출
    selected_indices = [feature_cols.index(f) for f in selected_feature_cols]
    
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    # 모델 학습
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1  # CPU 병렬
    )
    
    model.fit(X_train_selected, y_train, verbose=False)
    
    y_pred = model.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    
    return acc, f1


def visualize_importance(importance_df, top_k=20, save_path='feature_importance.png'):
    """피처 중요도 시각화 (matplotlib 필요)"""
    print(f"\n⚠️ matplotlib 미설치로 시각화 스킵")
    return


def main():
    """메인"""
    print("="*70)
    print("피처 셀렉션 (Feature Selection)")
    print("="*70)
    
    # 데이터 로드
    data_file = '02_data/01_processed/preprocessed_sequence.csv'
    print(f"⚠️ preprocessed_enhanced.csv 없음 → preprocessed_sequence.csv 사용")
    X, y, feature_cols = load_data(data_file)
    
    # 샘플링 (GPU 메모리 고려)
    sample_size = min(200000, len(X))  # 100만 → 20만
    print(f"\n⚡ 빠른 분석을 위해 {sample_size:,}개 샘플 사용 (GPU 메모리 고려)")
    
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    # 1. 베이스라인 (전체 피처)
    print("\n" + "="*70)
    print("1단계: 베이스라인 모델 (전체 피처)")
    print("="*70)
    baseline_model = train_baseline_model(X_train, y_train, X_test, y_test)
    
    # 2. 피처 중요도 분석
    print("\n" + "="*70)
    print("2단계: 피처 중요도 분석")
    print("="*70)
    importance_df, top_features = analyze_feature_importance(
        baseline_model, feature_cols, top_k=20
    )
    
    # 3. 피처 선택
    print("\n" + "="*70)
    print("3단계: 피처 선택")
    print("="*70)
    selected_features = select_features_by_importance(
        importance_df, threshold=0.01, min_features=15, max_features=18
    )
    
    # 4. 선택된 피처로 성능 평가
    print("\n" + "="*70)
    print("4단계: 선택된 피처 성능 평가")
    print("="*70)
    selected_acc, selected_f1 = evaluate_selected_features(
        X_train, y_train, X_test, y_test, feature_cols, selected_features
    )
    
    # 5. 결과 요약
    print("\n" + "="*70)
    print("결과 요약")
    print("="*70)
    print(f"\n전체 피처 ({len(feature_cols)}개):")
    print(f"  - 사용한 메모리: 100%")
    print(f"  - 학습 시간: 기준")
    
    print(f"\n선택된 피처 ({len(selected_features)}개):")
    print(f"  - 사용한 메모리: {len(selected_features)/len(feature_cols)*100:.1f}%")
    print(f"  - 학습 시간: 예상 {len(selected_features)/len(feature_cols)*100:.0f}%")
    print(f"  - Accuracy: {selected_acc:.4f}")
    print(f"  - Macro F1: {selected_f1:.4f}")
    
    # 6. 선택된 피처 저장
    selected_features_info = {
        'selected_features': [f.replace('_scaled', '') for f in selected_features],
        'num_features': len(selected_features),
        'total_features': len(feature_cols),
        'reduction_ratio': len(selected_features) / len(feature_cols),
        'performance': {
            'accuracy': float(selected_acc),
            'macro_f1': float(selected_f1)
        }
    }
    
    output_file = '02_data/01_processed/selected_features.json'
    with open(output_file, 'w') as f:
        json.dump(selected_features_info, f, indent=2)
    
    print(f"\n✅ 선택된 피처 정보 저장: {output_file}")
    
    # 7. 시각화
    visualize_importance(importance_df, top_k=20, save_path='05_docs/feature_importance.png')
    
    print("\n" + "="*70)
    print("피처 셀렉션 완료!")
    print("="*70)
    print(f"\n💡 권장: {len(selected_features)}개 피처 사용")
    print(f"   ({len(feature_cols) - len(selected_features)}개 제거 → 메모리 {(1-len(selected_features)/len(feature_cols))*100:.0f}% 절감)")


if __name__ == '__main__':
    main()
