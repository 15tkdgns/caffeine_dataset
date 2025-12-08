"""
LightGBM (CUDA) 프로덕션 모델 학습
최고 성능 모델 (Accuracy 49.11%) 재학습 및 저장
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

print("="*80)
print("🚀 LightGBM (CUDA) 프로덕션 모델 학습")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1/6] SMOTE 증강 데이터 로드")

X_train = np.load('02_data/02_augmented/X_train_smote.npy')
y_train = np.load('02_data/02_augmented/y_train_smote.npy')
X_test = np.load('02_data/02_augmented/X_test.npy')
y_test = np.load('02_data/02_augmented/y_test.npy')

print(f"  학습 데이터: {len(X_train):,}건")
print(f"  테스트 데이터: {len(X_test):,}건")
print(f"  피처 개수: {X_train.shape[1]}개")
print(f"  클래스 개수: {len(np.unique(y_train))}개")

# 메타데이터 로드
with open('02_data/02_augmented/metadata.json', 'r', encoding='utf-8') as f:
    data_metadata = json.load(f)

feature_names = data_metadata['feature_names']
print(f"  피처 목록: {len(feature_names)}개")

# ============================================================
# 2. 모델 정의
# ============================================================
print("\n[2/6] LightGBM (CUDA) 모델 정의")

model = lgb.LGBMClassifier(
    # device='cpu',  # GPU 미지원 환경에서 CPU 사용
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,  # CPU 멀티코어 활용
    verbose=-1
)

print("  ✅ 모델 파라미터:")
print(f"     - Device: CPU (Multi-core)")
print(f"     - N Estimators: 300")
print(f"     - Max Depth: 10")
print(f"     - Learning Rate: 0.1")
print(f"     - Num Leaves: 128")

# ============================================================
# 3. 모델 학습
# ============================================================
print("\n[3/6] 모델 학습 시작...")

start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"  ✅ 학습 완료: {train_time:.2f}초 ({train_time/60:.2f}분)")

# ============================================================
# 4. 모델 평가
# ============================================================
print("\n[4/6] 모델 평가")

# 예측
y_pred = model.predict(X_test)

# 성능 지표
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
category_f1 = f1_score(y_test, y_pred, average=None)

print(f"\n  📊 성능 지표:")
print(f"     Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"     Macro F1:    {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"     Weighted F1: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")

# 카테고리별 F1
category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
print(f"\n  📈 카테고리별 F1 Score:")
for cat_name, f1 in zip(category_names, category_f1):
    print(f"     {cat_name:6s}: {f1:.4f}")

# ============================================================
# 5. 모델 저장
# ============================================================
print("\n[5/6] 모델 및 메타데이터 저장")

# 저장 디렉토리
output_dir = '03_models/production_models'
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 파일 저장
model_filename = f'lightgbm_cuda_production_{timestamp}.joblib'
model_path = os.path.join(output_dir, model_filename)
joblib.dump(model, model_path)
print(f"  ✅ 모델 저장: {model_path}")

# SIZE 확인
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"     모델 크기: {model_size_mb:.2f} MB")

# ============================================================
# 6. 메타데이터 저장
# ============================================================
print("\n[6/6] 입력 스펙 메타데이터 저장")

# 입력 데이터 통계
input_stats = {}
for i, feat_name in enumerate(feature_names):
    feat_data = X_train[:, i]
    input_stats[feat_name] = {
        'index': i,
        'mean': float(np.mean(feat_data)),
        'std': float(np.std(feat_data)),
        'min': float(np.min(feat_data)),
        'max': float(np.max(feat_data)),
        'dtype': 'float32'
    }

# 카테고리 매핑
category_mapping = {
    0: '교통',
    1: '생활',
    2: '쇼핑',
    3: '식료품',
    4: '외식',
    5: '주유'
}

# 전체 메타데이터
metadata = {
    'model_info': {
        'model_name': 'LightGBM (CUDA) Production',
        'model_file': model_filename,
        'model_version': 'v1.0',
        'model_type': 'LightGBM Classifier',
        'framework': 'lightgbm',
        'device': 'GPU (CUDA)',
        'created_at': datetime.now().isoformat(),
        'model_size_mb': round(model_size_mb, 2)
    },
    
    'performance': {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'category_f1': {cat: round(f1, 4) for cat, f1 in zip(category_names, category_f1)},
        'train_time_seconds': round(train_time, 2),
        'train_time_minutes': round(train_time / 60, 2)
    },
    
    'training_data': {
        'description': 'SMOTE 증강 데이터',
        'filter_condition': data_metadata['filter_condition'],
        'original_samples': data_metadata['original_samples'],
        'filtered_samples': data_metadata['filtered_samples'],
        'active_users': data_metadata['active_users'],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'smote_ratio': round(len(X_train) / data_metadata['train_original'], 2)
    },
    
    'model_parameters': {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'n_estimators': 300,
        'max_depth': 10,
        'learning_rate': 0.1,
        'num_leaves': 128,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    },
    
    'input_spec': {
        'description': '모델 입력 데이터 스펙',
        'input_shape': [None, X_train.shape[1]],
        'input_dtype': 'float32',
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
        'feature_statistics': input_stats,
        'expected_format': 'numpy.ndarray or pandas.DataFrame',
        'scaling_method': 'StandardScaler applied during preprocessing',
        'missing_values': 'Not allowed - please impute before prediction'
    },
    
    'output_spec': {
        'description': '모델 출력 데이터 스펙',
        'output_type': 'class_label',
        'n_classes': len(category_mapping),
        'class_mapping': category_mapping,
        'output_dtype': 'int32',
        'prediction_methods': {
            'predict': 'Returns class labels (0-5)',
            'predict_proba': 'Returns probability distribution over classes'
        }
    },
    
    'usage_example': {
        'python_code': '''
# 모델 로드
import joblib
import numpy as np

model = joblib.load('lightgbm_cuda_production_TIMESTAMP.joblib')

# 입력 데이터 준비 (예시)
# 반드시 27개 피처가 올바른 순서로 정렬되어야 함
X_input = np.array([[
    # Amount_scaled, Amount_log_scaled, AmountBin_encoded_scaled,
    # Hour_scaled, DayOfWeek_scaled, DayOfMonth_scaled,
    # IsWeekend_scaled, IsLunchTime_scaled, IsEvening_scaled,
    # IsMorningRush_scaled, IsNight_scaled, IsBusinessHour_scaled,
    # User_AvgAmount_scaled, User_StdAmount_scaled, User_TxCount_scaled,
    # Time_Since_Last_scaled, Transaction_Sequence_scaled,
    # User_Category_Count_scaled, Current_Category_encoded_scaled,
    # Previous_Category_encoded_scaled, User_FavCategory_encoded_scaled,
    # User_교통_Ratio_scaled, User_생활_Ratio_scaled,
    # User_쇼핑_Ratio_scaled, User_식료품_Ratio_scaled,
    # User_외식_Ratio_scaled, User_주유_Ratio_scaled
    0.5, 0.3, 0.2, 0.6, 0.4, 0.5, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.4, 0.3, 0.5, 0.2, 0.7,
    0.8, 2.0, 1.0, 3.0, 0.1, 0.2, 0.3, 0.15, 0.2, 0.05
]], dtype=np.float32)

# 예측 (클래스 레이블)
y_pred = model.predict(X_input)
print(f"예측 카테고리: {y_pred[0]}")  # 0-5

# 예측 (확률 분포)
y_proba = model.predict_proba(X_input)
print(f"예측 확률: {y_proba[0]}")  # [교통, 생활, 쇼핑, 식료품, 외식, 주유]
''',
        'curl_example': '''
# FastAPI 서버에 REST API 요청 (예시)
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": [0.5, 0.3, 0.2, 0.6, 0.4, 0.5, 0.0, 0.0, 1.0,
                 0.0, 0.0, 1.0, 0.4, 0.3, 0.5, 0.2, 0.7, 0.8,
                 2.0, 1.0, 3.0, 0.1, 0.2, 0.3, 0.15, 0.2, 0.05]
  }'
'''
    },
    
    'important_notes': [
        '⚠️ 입력 데이터는 반드시 27개 피처를 정확한 순서로 포함해야 합니다',
        '⚠️ 모든 피처는 StandardScaler로 정규화되어야 합니다 (feature_statistics 참고)',
        '⚠️ GPU(CUDA) 환경에서 학습되었지만, CPU 환경에서도 예측 가능합니다',
        '⚠️ 결측값은 허용되지 않습니다. 예측 전에 반드시 처리하세요',
        '✅ SMOTE 증강 데이터로 학습되어 클래스 불균형이 해소되었습니다',
        '✅ 실시간 예측에 최적화되어 있습니다'
    ]
}

# 메타데이터 저장
metadata_filename = f'lightgbm_cuda_metadata_{timestamp}.json'
metadata_path = os.path.join(output_dir, metadata_filename)

with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"  ✅ 메타데이터 저장: {metadata_path}")

# ============================================================
# 7. 완료 요약
# ============================================================
print("\n" + "="*80)
print("✅ LightGBM (CUDA) 프로덕션 모델 학습 완료!")
print("="*80)

print(f"\n📦 생성된 파일:")
print(f"   1. 모델 파일:      {model_path}")
print(f"   2. 메타데이터:      {metadata_path}")

print(f"\n📊 최종 성능:")
print(f"   - Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   - Macro F1:    {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"   - 학습 시간:    {train_time:.2f}초")

print(f"\n🎯 사용 방법:")
print(f"   모델 로드: model = joblib.load('{model_path}')")
print(f"   예측: y_pred = model.predict(X_input)  # X_input shape: (n_samples, 27)")
print(f"   확률: y_proba = model.predict_proba(X_input)")

print(f"\n📚 자세한 사용법은 메타데이터 파일을 참고하세요:")
print(f"   {metadata_path}")

print("\n" + "="*80)
