"""
GPU 사용 가능 여부 체크 유틸리티
"""

import sys

def check_tensorflow_gpu():
    """TensorFlow GPU 체크"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU: {len(gpus)}개 감지")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("❌ TensorFlow GPU: 감지 안됨")
            return False
    except Exception as e:
        print(f"❌ TensorFlow GPU 체크 실패: {e}")
        return False


def check_xgboost_gpu():
    """XGBoost GPU 체크"""
    try:
        import xgboost as xgb
        # GPU 장치 테스트
        dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
        params = {'device': 'cuda', 'tree_method': 'hist'}
        model = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        print("✅ XGBoost GPU: 사용 가능")
        return True
    except Exception as e:
        print(f"❌ XGBoost GPU 체크 실패: {e}")
        return False


def check_cuml_gpu():
    """cuML GPU 체크"""
    try:
        from cuml.ensemble import RandomForestClassifier
        import cupy as cp
        
        # 간단한 테스트
        X = cp.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=cp.float32)
        y = cp.array([0, 1, 0, 1], dtype=cp.int32)
        
        model = RandomForestClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        print("✅ cuML GPU (RandomForest): 사용 가능")
        return True
    except Exception as e:
        print(f"❌ cuML GPU 체크 실패: {e}")
        return False


def check_lightgbm_gpu():
    """LightGBM GPU 체크"""
    try:
        import lightgbm as lgb
        
        # GPU 파라미터 테스트
        train_data = lgb.Dataset([[1, 2], [3, 4]], label=[0, 1])
        params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
        model = lgb.train(params, train_data, num_boost_round=1)
        
        print("✅ LightGBM GPU: 사용 가능")
        return True
    except Exception as e:
        print(f"❌ LightGBM GPU 체크 실패: {e}")
        return False


def check_catboost_gpu():
    """CatBoost GPU 체크"""
    try:
        from catboost import CatBoostClassifier
        import numpy as np
        
        # 간단한 테스트
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        model = CatBoostClassifier(
            task_type='GPU',
            devices='0',
            iterations=10,
            verbose=False
        )
        model.fit(X, y)
        
        print("✅ CatBoost GPU: 사용 가능")
        return True
    except Exception as e:
        print(f"❌ CatBoost GPU 체크 실패: {e}")
        return False


def check_all_gpus():
    """모든 GPU 프레임워크 체크"""
    print("="*70)
    print("GPU 사용 가능 여부 체크")
    print("="*70)
    
    results = {}
    
    print("\n[1/4] TensorFlow GPU")
    results['tensorflow'] = check_tensorflow_gpu()
    
    print("\n[2/4] XGBoost GPU")
    results['xgboost'] = check_xgboost_gpu()
    
    print("\n[3/4] cuML GPU (RandomForest)")
    results['cuml'] = check_cuml_gpu()
    
    print("\n[4/5] LightGBM GPU")
    results['lightgbm'] = check_lightgbm_gpu()
    
    print("\n[5/5] CatBoost GPU")
    results['catboost'] = check_catboost_gpu()
    
    # 요약
    print("\n" + "="*70)
    print("GPU 사용 가능 모델 요약")
    print("="*70)
    
    available_models = []
    for framework, available in results.items():
        status = "✅ 사용 가능" if available else "❌ 사용 불가"
        print(f"{framework:15s}: {status}")
        if available:
            available_models.append(framework)
    
    print("\n" + "="*70)
    if available_models:
        print(f"✅ GPU 사용 가능 모델: {', '.join(available_models)}")
        print(f"   총 {len(available_models)}개 프레임워크")
    else:
        print("❌ GPU 사용 가능한 모델이 없습니다")
        print("   CPU 모드로 학습하거나 GPU 설정을 확인하세요")
    
    return results


def get_available_gpu_models():
    """GPU 사용 가능한 모델 목록 반환"""
    results = check_all_gpus()
    available = [k for k, v in results.items() if v]
    return available


if __name__ == '__main__':
    results = check_all_gpus()
    
    # 시스템 종료 코드
    # GPU 사용 가능한 모델이 하나라도 있으면 0, 없으면 1
    if any(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
