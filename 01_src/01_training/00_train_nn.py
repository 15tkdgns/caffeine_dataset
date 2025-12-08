

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import numpy as np # For numerical operations

import argparse
import sys

def check_gpu(device_arg):
    """GPU 장치를 확인하고 상세 정보를 출력합니다. device_arg가 'gpu'일 경우 강제합니다."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # GPU 메모리를 동적으로 할당하도록 설정합니다.
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"성공: {len(gpus)}개의 GPU를 찾았습니다:")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"GPU 설정 중 런타임 오류 발생: {e}")
            if device_arg == 'gpu':
                raise RuntimeError("GPU 초기화 실패. --device=gpu가 지정되어 종료합니다.")
            return False
    else:
        if device_arg == 'gpu':
            raise RuntimeError("GPU를 찾을 수 없습니다. --device=gpu가 지정되어 종료합니다.")
        print("경고: GPU를 찾을 수 없습니다. CPU로 학습을 진행합니다.")
        return False

def load_processed_data(file_path):
    """
    전처리된 데이터를 CSV 파일에서 로드합니다.
    빠른 테스트를 위해 100,000건만 로드합니다.
    """
    print(f"'{file_path}'에서 전처리된 데이터를 로드합니다 (최대 100,000건)...")
    df = pd.read_csv(file_path, nrows=100000)
    print(f"총 {len(df)}건의 전처리된 데이터 로드 완료.")
    return df

def build_model(input_shape, num_classes):
    """
    간단한 피드포워드 신경망 모델을 구축합니다.
    """
    print("\n신경망 모델을 구축합니다...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("모델 컴파일 완료.")
    model.summary()
    return model

def main():
    """
    메인 함수: 전처리된 데이터를 사용하여 머신러닝 모델을 학습합니다.
    """
    parser = argparse.ArgumentParser(description="신경망 모델 학습 스크립트")
    parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto', help="실행 디바이스 선택")
    args = parser.parse_args()

    print("="*50)
    print("머신러닝 모델 학습 스크립트 시작")
    print("="*50)
    
    # 1. GPU 확인
    use_gpu = check_gpu(args.device)
    
    if args.device == 'gpu' and not use_gpu:
         # check_gpu에서 이미 raise하지만 안전장치
         raise RuntimeError("GPU를 사용할 수 없습니다.")

    device = '/GPU:0' if use_gpu else '/CPU:0'
    print(f"학습 디바이스: {device}")

    # 2. 전처리된 데이터 파일 경로 설정
    # 스크립트가 실행되는 현재 작업 디렉토리(프로젝트 루트)를 기준으로 경로 설정
    processed_data_file = '02_data/01_processed/preprocessed_transactions_v2.csv'
    
    # 3. 전처리된 데이터 로드
    df_processed = load_processed_data(processed_data_file)
    
    # 4. 특성(X)과 타겟(y) 분리
    # 'Amount_scaled', 'Is_Weekend'가 특성으로, 'Category_encoded'가 타겟으로 가정합니다.
    # 'User', 'Card', 'Transaction_ID' 등은 학습에 사용되지 않는 식별자로 간주합니다.
    features_cols = [col for col in df_processed.columns if col.endswith('_scaled') or col == 'Is_Weekend']
    target_col = 'Category_encoded'
    
    X = df_processed[features_cols].values
    y = df_processed[target_col].values
    
    # 5. 학습/테스트 데이터 분리
    print("\n학습/테스트 데이터 분리 중...")
    # 타겟 변수의 클래스 분포를 유지하면서 분리 (stratify)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  - 학습 데이터셋 크기: {len(X_train)}건")
    print(f"  - 테스트 데이터셋 크기: {len(X_test)}건")
    
    # 클래스 수 결정 (타겟 변수의 고유 값 개수)
    num_classes = len(np.unique(y))
    
    # 6. 모델 구축
    model = build_model(input_shape=(X_train.shape[1],), num_classes=num_classes)
    
    # 7. TensorBoard 콜백 설정
    log_dir = '04_logs/00_tensorflow/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"\nTensorBoard 로그는 '{log_dir}' 디렉토리에 저장됩니다.")
    print("TensorBoard를 실행하려면, 새로운 터미널을 열고 가상 환경을 활성화한 후 다음 명령어를 입력하세요:")
    print(f"source mlcu/bin/activate && tensorboard --logdir {os.path.dirname(log_dir)}") # 상위 디렉토리 지정
    
    # 8. 모델 학습
    print("\n모델 학습을 시작합니다...")
    with tf.device(device):
        history = model.fit(X_train, y_train,
                          epochs=1,  # 빠른 테스트를 위해 에포크를 1로 설정
                          batch_size=1024, # 배치 크기
                          validation_data=(X_test, y_test),
                          callbacks=[tensorboard_callback], # TensorBoard 콜백 추가
                          verbose=1) # 학습 진행 상황 출력
    
    print("\n모델 학습 완료.")
    
    # 9. 모델 평가
    print("\n모델 평가 중...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n최종 테스트 정확도: {accuracy:.4f}")
    
    # 10. (선택 사항) 모델 저장
    model_save_path = '03_models/00_nn/' + f"my_transaction_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.keras"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\n학습된 모델이 '{model_save_path}'에 저장되었습니다.")

    print("="*50)
    print("스크립트 실행 완료")
    print("="*50)

if __name__ == '__main__':
    main()
