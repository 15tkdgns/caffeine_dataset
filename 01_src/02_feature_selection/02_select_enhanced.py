"""
개선된 피처(23개)로 피처 셀렉션
목표: 23개 → 15-18개 핵심 피처만 선택
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import json
import os

def main():
    """메인"""
    print("="*70)
    print("피처 셀렉션 (Enhanced 23개 피처)")
    print("="*70)
    
    # 데이터 로드
    data_file = '02_data/01_processed/preprocessed_enhanced.csv'
    
    if not os.path.exists(data_file):
        print(f"\n❌ 파일 없음: {data_file}")
        print("먼저 개선된 전처리를 실행하세요:")
        print("  python3 01_src/00_preprocessing/03_preprocess_enhanced.py")
        return
    
    print(f"\n데이터 로드: {data_file}")
    df = pd.read_csv(data_file)
    
    feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    target_col = 'Next_Category_encoded'
    
    X = df[feature_cols].values.astype('float32')
    y = df[target_col].values
    
    print(f"  전체 피처: {len(feature_cols)}개")
    print(f"  샘플 수: {len(X):,}개")
    
    # 샘플링 (CPU 메모리 고려)
    sample_size = min(500000, len(X))
    print(f"\n⚡ 분석용 샘플: {sample_size:,}개")
    
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    # 베이스라인 (전체 피처)
    print("\n" + "="*70)
    print("1단계: 베이스라인 (전체 피처)")
    print("="*70)
    print(f"\n{len(feature_cols)}개 피처로 학습 중...")
    
    model_full = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model_full.fit(X_train, y_train, verbose=False)
    y_pred = model_full.predict(X_test)
    
    acc_full = accuracy_score(y_test, y_pred)
    f1_full = f1_score(y_test, y_pred, average='macro')
    
    print(f"  Accuracy: {acc_full:.4f}")
    print(f"  Macro F1: {f1_full:.4f}")
    
    # 피처 중요도
    print("\n" + "="*70)
    print("2단계: 피처 중요도 분석")
    print("="*70)
    
    importances = model_full.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': [f.replace('_scaled', '') for f in feature_cols],
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n상위 20개 피처:")
    print("="*60)
    for idx, row in importance_df.head(20).iterrows():
        print(f"{row.name+1:2d}. {row['Feature']:40s} {row['Importance']*100:6.2f}%")
    
    # 누적 중요도
    importance_df['Cumsum'] = importance_df['Importance'].cumsum()
    
    # 피처 선택 (누적 95%)
    print("\n" + "="*70)
    print("3단계: 피처 선택 (누적 95% 기준)")
    print("="*70)
    
    selected_95 = importance_df[importance_df['Cumsum'] <= 0.95]
    
    # 최소 12개, 최대 18개
    if len(selected_95) < 12:
        selected_df = importance_df.head(12)
        print(f"  누적 95% 미만 → 상위 12개 강제 선택")
    elif len(selected_95) > 18:
        selected_df = importance_df.head(18)
        print(f"  누적 95% 초과 → 상위 18개로 제한")
    else:
        selected_df = selected_95
        print(f"  누적 95% 기준: {len(selected_df)}개 선택")
    
    selected_features = selected_df['Feature'].tolist()
    selected_features_scaled = [f"{f}_scaled" for f in selected_features]
    
    print(f"\n✅ 선택된 피처: {len(selected_features)}개")
    print("="*60)
    for idx, feat in enumerate(selected_features, 1):
        imp = selected_df[selected_df['Feature'] == feat]['Importance'].values[0]
        print(f"{idx:2d}. {feat:40s} {imp*100:6.2f}%")
    
    # 선택된 피처로 평가
    print("\n" + "="*70)
    print("4단계: 선택된 피처 성능 평가")
    print("="*70)
    
    selected_indices = [feature_cols.index(f) for f in selected_features_scaled]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    model_selected = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model_selected.fit(X_train_selected, y_train, verbose=False)
    y_pred_selected = model_selected.predict(X_test_selected)
    
    acc_selected = accuracy_score(y_test, y_pred_selected)
    f1_selected = f1_score(y_test, y_pred_selected, average='macro')
    
    print(f"\n전체 피처 ({len(feature_cols)}개):")
    print(f"  Accuracy: {acc_full:.4f}")
    print(f"  Macro F1: {f1_full:.4f}")
    
    print(f"\n선택된 피처 ({len(selected_features)}개):")
    print(f"  Accuracy: {acc_selected:.4f} ({acc_selected-acc_full:+.4f})")
    print(f"  Macro F1: {f1_selected:.4f} ({f1_selected-f1_full:+.4f})")
    print(f"  메모리 절감: {(1-len(selected_features)/len(feature_cols))*100:.1f}%")
    print(f"  속도 향상: 예상 ~{(1-len(selected_features)/len(feature_cols))*100:.0f}%")
    
    # 저장
    selected_info = {
        'selected_features': selected_features,
        'num_features': len(selected_features),
        'total_features': len(feature_cols),
        'reduction': f"{(1-len(selected_features)/len(feature_cols))*100:.1f}%",
        'performance': {
            'baseline_accuracy': float(acc_full),
            'baseline_f1': float(f1_full),
            'selected_accuracy': float(acc_selected),
            'selected_f1': float(f1_selected),
            'accuracy_diff': float(acc_selected - acc_full),
            'f1_diff': float(f1_selected - f1_full)
        }
    }
    
    output_file = '02_data/01_processed/selected_features_enhanced.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 저장: {output_file}")
    
    # 제거된 피처 확인
    removed_features = set([f.replace('_scaled', '') for f in feature_cols]) - set(selected_features)
    if removed_features:
        print(f"\n제거된 피처 ({len(removed_features)}개):")
        for feat in sorted(removed_features):
            imp = importance_df[importance_df['Feature'] == feat]['Importance'].values
            if len(imp) > 0:
                print(f"  - {feat:40s} {imp[0]*100:6.2f}%")
    
    print("\n" + "="*70)
    print("피처 셀렉션 완료!")
    print("="*70)
    print(f"\n💡 권장: {len(selected_features)}개 피처 사용")
    print(f"   ({len(feature_cols)} → {len(selected_features)}개)")
    print(f"   메모리: -{(1-len(selected_features)/len(feature_cols))*100:.0f}%, 성능 손실: {(f1_selected-f1_full)*100:+.2f}%p")


if __name__ == '__main__':
    main()
