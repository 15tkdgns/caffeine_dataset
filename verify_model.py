#!/usr/bin/env python3
"""
최고 성능 모델 검증 스크립트
생성일: 2025-12-10
"""

import joblib
import json
import os
import sys

def verify_model():
    """모델 파일 검증 및 정보 출력"""
    
    model_path = "best_model_xgboost_acc_73.47.joblib"
    metadata_path = "best_model_metadata.json"
    
    print("=" * 70)
    print("🏆 최고 성능 모델 검증")
    print("=" * 70)
    
    # 1. 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(metadata_path):
        print(f"⚠️  메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
    
    # 2. 파일 크기 확인
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n📦 모델 파일 정보:")
    print(f"   - 경로: {os.path.abspath(model_path)}")
    print(f"   - 크기: {model_size_mb:.2f} MB")
    
    # 3. 모델 로드 테스트
    try:
        print(f"\n🔄 모델 로딩 중...")
        model = joblib.load(model_path)
        print(f"✅ 모델 로드 성공!")
        print(f"   - 타입: {type(model).__name__}")
        
        # 모델 속성 확인
        if hasattr(model, 'n_features_in_'):
            print(f"   - 입력 피처 개수: {model.n_features_in_}")
        if hasattr(model, 'n_classes_'):
            print(f"   - 출력 클래스 개수: {model.n_classes_}")
        if hasattr(model, 'classes_'):
            print(f"   - 클래스 레이블: {model.classes_}")
            
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)
    
    # 4. 메타데이터 출력
    if os.path.exists(metadata_path):
        print(f"\n📊 모델 성능 지표:")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        best_model_name = metadata.get('best_model', 'Unknown')
        best_model_info = metadata.get('models', {}).get(best_model_name, {})
        
        print(f"   - 최고 모델: {best_model_name}")
        print(f"   - 정확도: {best_model_info.get('accuracy', 0) * 100:.2f}%")
        print(f"   - Macro F1: {best_model_info.get('macro_f1', 0) * 100:.2f}%")
        print(f"   - Weighted F1: {best_model_info.get('weighted_f1', 0) * 100:.2f}%")
        print(f"   - 학습 시간: {best_model_info.get('train_time', 0):.2f}초")
        
        # 카테고리별 F1 Score
        category_f1 = best_model_info.get('category_f1', [])
        if category_f1:
            categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
            print(f"\n📈 카테고리별 F1 Score:")
            for i, (cat, f1) in enumerate(zip(categories, category_f1)):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"   {medal} {cat}: {f1 * 100:.2f}%")
        
        # 피처 정보
        features = metadata.get('features', [])
        print(f"\n🔧 피처 정보:")
        print(f"   - 총 {len(features)}개 피처")
        print(f"   - 피처 목록: {', '.join(features[:5])}...")
    
    # 5. 사용 예시
    print(f"\n💻 사용 예시:")
    print(f"""
    import joblib
    import numpy as np
    
    # 모델 로드
    model = joblib.load('{model_path}')
    
    # 예측 (24개 피처 필요)
    X_sample = np.random.randn(1, {model.n_features_in_ if hasattr(model, 'n_features_in_') else 24})
    prediction = model.predict(X_sample)
    probability = model.predict_proba(X_sample)
    
    print(f"예측 카테고리: {{prediction[0]}}")
    print(f"예측 확률: {{probability[0]}}")
    """)
    
    print("=" * 70)
    print("✅ 모델 검증 완료!")
    print("=" * 70)

if __name__ == "__main__":
    verify_model()
