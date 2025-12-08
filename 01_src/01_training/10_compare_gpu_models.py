"""
GPU 전용 모델 비교 프레임워크
GPU를 사용하는 모델만 학습 및 비교
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time

# GPU 체크
import sys
sys.path.append('01_src/utils')
from gpu_check import get_available_gpu_models


class GPUModelComparison:
    """GPU 모델 비교 클래스"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self, test_size=0.2, sample_frac=1.0):
        """데이터 로드 및 분할"""
        print("="*70)
        print("데이터 로드")
        print("="*70)
        
        df = pd.read_csv(self.data_file)
        print(f"총 샘플: {len(df):,}개")
        
        # 샘플링 (필요시)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"샘플링 후: {len(df):,}개 ({sample_frac*100:.0f}%)")
        
        # 피처와 타겟 분리
        feature_cols = [col for col in df.columns if col.endswith('_scaled')]
        target_col = 'Next_Category_encoded'
        
        if target_col not in df.columns:
            raise ValueError(f"타겟 컬럼 '{target_col}'이 없습니다")
        
        X = df[feature_cols].values.astype('float32')
        y = df[target_col].values.astype('int32')
        
        self.feature_names = [col.replace('_scaled', '') for col in feature_cols]
        
        print(f"피처 수: {len(feature_cols)}개")
        print(f"클래스 수: {len(np.unique(y))}개")
        
        # 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n학습 데이터: {len(self.X_train):,}개")
        print(f"테스트 데이터: {len(self.X_test):,}개")
        
        return self
    
    def add_xgboost_model(self):
        """XGBoost GPU 모델 추가"""
        try:
            import xgboost as xgb
            
            model = xgb.XGBClassifier(
                device='cuda',
                tree_method='hist',
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['XGBoost (GPU)'] = {
                'model': model,
                'framework': 'xgboost',
                'device': 'GPU'
            }
            print("✅ XGBoost (GPU) 추가")
        except Exception as e:
            print(f"❌ XGBoost (GPU) 추가 실패: {e}")
    
    def add_randomforest_cuml(self):
        """cuML RandomForest GPU 모델 추가"""
        try:
            from cuml.ensemble import RandomForestClassifier
            import cupy as cp
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                max_features=0.8,
                random_state=42,
                n_streams=4
            )
            
            self.models['RandomForest (cuML GPU)'] = {
                'model': model,
                'framework': 'cuml',
                'device': 'GPU',
                'requires_cupy': True
            }
            print("✅ RandomForest (cuML GPU) 추가")
        except Exception as e:
            print(f"❌ RandomForest (cuML GPU) 추가 실패: {e}")
    
    def add_lightgbm_gpu(self):
        """LightGBM GPU 모델 추가"""
        try:
            import lightgbm as lgb
            
            model = lgb.LGBMClassifier(
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=128,
                random_state=42
            )
            
            self.models['LightGBM (GPU)'] = {
                'model': model,
                'framework': 'lightgbm',
                'device': 'GPU'
            }
            print("✅ LightGBM (GPU) 추가")
        except Exception as e:
            print(f"❌ LightGBM (GPU) 추가 실패: {e}")
    
    def add_tensorflow_model(self):
        """TensorFlow Neural Network GPU 모델 추가"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # GPU 확인
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                print("❌ TensorFlow GPU 사용 불가")
                return
            
            n_features = self.X_train.shape[1]
            n_classes = len(np.unique(self.y_train))
            
            model = keras.Sequential([
                keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(n_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.models['Neural Network (TF GPU)'] = {
                'model': model,
                'framework': 'tensorflow',
                'device': 'GPU',
                'is_keras': True
            }
            print("✅ Neural Network (TensorFlow GPU) 추가")
        except Exception as e:
            print(f"❌ Neural Network (TensorFlow GPU) 추가 실패: {e}")
    
    def add_catboost_gpu(self):
        """CatBoost GPU 모델 추가"""
        try:
            from catboost import CatBoostClassifier
            
            model = CatBoostClassifier(
                task_type='GPU',
                devices='0',
                iterations=300,
                depth=10,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
            
            self.models['CatBoost (GPU)'] = {
                'model': model,
                'framework': 'catboost',
                'device': 'GPU'
            }
            print("✅ CatBoost (GPU) 추가")
        except Exception as e:
            print(f"❌ CatBoost (GPU) 추가 실패: {e}")
    
    def add_extratrees_cpu(self):
        """ExtraTrees CPU 모델 추가 (참고용)"""
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.utils.class_weight import compute_sample_weight
            
            model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.models['ExtraTrees (CPU)'] = {
                'model': model,
                'framework': 'sklearn',
                'device': 'CPU'
            }
            print("✅ ExtraTrees (CPU) 추가")
        except Exception as e:
            print(f"❌ ExtraTrees (CPU) 추가 실패: {e}")
    
    def add_randomforest_cpu(self):
        """RandomForest CPU 모델 추가 (참고용)"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.models['RandomForest (CPU)'] = {
                'model': model,
                'framework': 'sklearn',
                'device': 'CPU'
            }
            print("✅ RandomForest (CPU) 추가")
        except Exception as e:
            print(f"❌ RandomForest (CPU) 추가 실패: {e}")
    
    def train_all_models(self):
        """모든 모델 학습"""
        print("\n" + "="*70)
        print("모델 학습")
        print("="*70)
        
        for model_name, model_info in self.models.items():
            print(f"\n[{model_name}] 학습 시작...")
            
            try:
                start_time = time.time()
                
                # cuML 모델은 CuPy 배열 필요
                if model_info.get('requires_cupy', False):
                    import cupy as cp
                    X_train_gpu = cp.array(self.X_train, dtype=cp.float32)
                    y_train_gpu = cp.array(self.y_train, dtype=cp.int32)
                    model_info['model'].fit(X_train_gpu, y_train_gpu)
                
                # TensorFlow 모델
                elif model_info.get('is_keras', False):
                    model_info['model'].fit(
                        self.X_train, self.y_train,
                        epochs=10,
                        batch_size=1024,
                        validation_split=0.1,
                        verbose=0
                    )
                
                # 일반 모델 (XGBoost, LightGBM)
                else:
                    model_info['model'].fit(self.X_train, self.y_train)
                
                train_time = time.time() - start_time
                
                print(f"  ✅ 학습 완료 ({train_time:.2f}초)")
                model_info['train_time'] = train_time
                
            except Exception as e:
                print(f"  ❌ 학습 실패: {e}")
                model_info['error'] = str(e)
    
    def evaluate_all_models(self):
        """모든 모델 평가"""
        print("\n" + "="*70)
        print("모델 평가")
        print("="*70)
        
        for model_name, model_info in self.models.items():
            if 'error' in model_info:
                print(f"\n[{model_name}] 건너뛰기 (학습 실패)")
                continue
            
            print(f"\n[{model_name}] 평가 중...")
            
            try:
                start_time = time.time()
                
                # 예측
                if model_info.get('requires_cupy', False):
                    import cupy as cp
                    X_test_gpu = cp.array(self.X_test, dtype=cp.float32)
                    y_pred_gpu = model_info['model'].predict(X_test_gpu)
                    y_pred = cp.asnumpy(y_pred_gpu)
                
                elif model_info.get('is_keras', False):
                    y_pred_proba = model_info['model'].predict(self.X_test, verbose=0)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                
                else:
                    y_pred = model_info['model'].predict(self.X_test)
                
                predict_time = time.time() - start_time
                
                # 평가 지표
                accuracy = accuracy_score(self.y_test, y_pred)
                macro_f1 = f1_score(self.y_test, y_pred, average='macro')
                weighted_f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # 카테고리별 F1
                category_f1 = f1_score(self.y_test, y_pred, average=None)
                
                self.results[model_name] = {
                    'accuracy': float(accuracy),
                    'macro_f1': float(macro_f1),
                    'weighted_f1': float(weighted_f1),
                    'category_f1': category_f1.tolist(),
                    'train_time': model_info['train_time'],
                    'predict_time': float(predict_time),
                    'framework': model_info['framework'],
                    'device': model_info['device']
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Macro F1: {macro_f1:.4f}")
                print(f"  Weighted F1: {weighted_f1:.4f}")
                print(f"  학습 시간: {model_info['train_time']:.2f}초")
                print(f"  예측 시간: {predict_time:.2f}초")
                
            except Exception as e:
                print(f"  ❌ 평가 실패: {e}")
    
    def save_results(self, output_dir='03_models/comparison'):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON 저장
        results_file = os.path.join(output_dir, 'gpu_models_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # CSV 저장
        results_df = pd.DataFrame(self.results).T
        results_csv = os.path.join(output_dir, 'gpu_models_comparison.csv')
        results_df.to_csv(results_csv)
        
        print(f"\n✅ 결과 저장:")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {results_csv}")
        
        return results_df


def main():
    """메인 실행"""
    print("="*70)
    print("GPU 모델 비교 프레임워크")
    print("="*70)
    
    # GPU 체크
    print("\n[단계 1] GPU 사용 가능 여부 확인")
    available_gpus = get_available_gpu_models()
    
    if not available_gpus:
        print("\n❌ GPU 사용 가능한 모델이 없습니다")
        print("GPU 설정을 확인하거나 CPU 모드를 사용하세요")
        return
    
    # 데이터 파일 확인
    data_file = '02_data/01_processed/preprocessed_filtered_monthly.csv'
    if not os.path.exists(data_file):
        print(f"\n⚠️ 필터링된 데이터가 없습니다: {data_file}")
        print("대체 파일 사용: preprocessed_enhanced.csv")
        data_file = '02_data/01_processed/preprocessed_enhanced.csv'
    
    if not os.path.exists(data_file):
        print(f"\n❌ 데이터 파일 없음: {data_file}")
        return
    
    # 모델 비교 실행
    comparison = GPUModelComparison(data_file)
    
    # 데이터 로드
    print("\n[단계 2] 데이터 로드")
    comparison.load_data(test_size=0.2, sample_frac=1.0)
    
    # 모델 추가
    print("\n[단계 3] 모델 추가 (GPU 우선)")
    
    # GPU 모델
    print("\n[3-1] GPU 모델:")
    if 'xgboost' in available_gpus:
        comparison.add_xgboost_model()
    if 'cuml' in available_gpus:
        comparison.add_randomforest_cuml()
    if 'lightgbm' in available_gpus:
        comparison.add_lightgbm_gpu()
    if 'tensorflow' in available_gpus:
        comparison.add_tensorflow_model()
    
    # CatBoost GPU 시도 (GPU 목록에 없어도 시도)
    comparison.add_catboost_gpu()
    
    # CPU 모델 (비교용)
    print("\n[3-2] CPU 모델 (성능 비교용):")
    comparison.add_extratrees_cpu()
    comparison.add_randomforest_cpu()
    
    if not comparison.models:
        print("\n❌ 추가된 모델이 없습니다")
        return
    
    print(f"\n총 {len(comparison.models)}개 모델 추가됨")
    
    # 학습
    print("\n[단계 4] 모델 학습")
    comparison.train_all_models()
    
    # 평가
    print("\n[단계 5] 모델 평가")
    comparison.evaluate_all_models()
    
    # 결과 저장
    print("\n[단계 6] 결과 저장")
    results_df = comparison.save_results()
    
    # 요약
    print("\n" + "="*70)
    print("GPU 모델 성능 비교 요약")
    print("="*70)
    print(results_df[['accuracy', 'macro_f1', 'weighted_f1', 'train_time']].to_string())
    
    # Top 3 선정
    top3 = results_df.nlargest(3, 'accuracy')
    print("\n" + "="*70)
    print("Top 3 모델 (Accuracy 기준)")
    print("="*70)
    for idx, (model_name, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{idx}. {model_name}")
        print(f"   Accuracy: {row['accuracy']:.4f}")
        print(f"   Macro F1: {row['macro_f1']:.4f}")
        print(f"   학습 시간: {row['train_time']:.2f}초")


if __name__ == '__main__':
    main()
