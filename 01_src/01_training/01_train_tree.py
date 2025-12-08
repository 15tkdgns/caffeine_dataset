


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score

import joblib

import os
import shutil
import subprocess

import argparse

import time

from datetime import datetime



# 모델 라이브러리 임포트

import lightgbm as lgb

import xgboost as xgb

from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTreesClassifier



# RAPIDS cuML ExtraTreesClassifier 임포트 시도

try:

    from cuml.ensemble import ExtraTreesClassifier as CumlExtraTreesClassifier

    _has_cuml = True

    print("cuML ExtraTreesClassifier를 성공적으로 로드했습니다.")

except ImportError as e:

    _has_cuml = False

    print(f"경고: cuML ExtraTreesClassifier를 로드할 수 없습니다. 오류: {e}")
    print("ExtraTrees는 CPU 버전으로 실행됩니다.")

    print("GPU 가속 ExtraTrees를 사용하려면 RAPIDS cuML을 설치해야 합니다.")

    print("설치 방법: `conda install -c rapidsai -c conda-forge -c nvidia cuml` (CUDA 버전과 호환되는지 확인하세요)")





def is_gpu_available():
    """
    NVIDIA GPU 드라이버가 확인되면 True를 반환합니다.
    torch가 설치되어 있다면 CUDA 사용 여부도 함께 점검합니다.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return bool(result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return False


def resolve_gpu_mode(device_option, gpu_available):
    """
    --device 옵션(auto/gpu/cpu)을 해석하여 GPU를 사용할지 결정합니다.
    """
    if device_option == "gpu":
        if not gpu_available:
            raise RuntimeError(
                "GPU 모드가 요청되었지만 GPU가 감지되지 않았습니다. "
                "CUDA 드라이버/라이브러리를 점검해 주세요."
            )
        return True

    if device_option == "auto":
        if gpu_available:
            return True
        else:
            # User requested strict GPU usage
            raise RuntimeError("GPU를 찾을 수 없습니다. (Auto 모드에서도 GPU 필수)")

    return False


def load_processed_data(file_path, nrows=None):

    """

    전처리된 데이터를 CSV 파일에서 로드합니다.

    nrows를 통해 테스트 시 로드할 데이터 양을 조절할 수 있습니다.

    """

    print(f"'{file_path}'에서 전처리된 데이터를 로드합니다 (nrows={nrows if nrows else '모두'})...")

    df = pd.read_csv(file_path, nrows=nrows)

    print(f"총 {len(df)}건의 전처리된 데이터 로드 완료.")

    return df



def train_model(model_name, X_train, y_train, X_test, y_test, use_gpu=True):

    """지정된 모델을 학습하고 평가합니다.

    use_gpu=True일 경우 GPU 파라미터를 설정합니다.

    """

    device_label = "GPU" if use_gpu else "CPU"
    print(f"\n{'='*50}")
    print(f"모델 학습 시작: {model_name} ({device_label})")
    print(f"{ '='*50}")

    model = None
    if model_name == 'lightgbm':
        if use_gpu:
            print("LightGBM을 GPU로 실행합니다.")
            model = lgb.LGBMClassifier(random_state=42, n_estimators=500, device='gpu')
        else:
            print("LightGBM을 CPU로 실행합니다.")
            model = lgb.LGBMClassifier(random_state=42, n_estimators=500, n_jobs=-1)
    elif model_name == "xgboost":
        # XGBoost 2.0+ / 3.0+ settings
        tree_method = "hist" 
        device = "cuda" if use_gpu else "cpu"
        
        model = xgb.XGBClassifier(
            random_state=42,
            tree_method=tree_method,
            device=device,
            n_estimators=500,
        )
        print(f"XGBoost를 {device_label}로 실행합니다.")
    elif model_name == "randomforest":
        if use_gpu:
            # Try to use cuml RandomForest first
            try:
                from cuml.ensemble import RandomForestClassifier as CumlRandomForestClassifier
                model = CumlRandomForestClassifier(
                    random_state=42,
                    n_estimators=500,
                    max_depth=16,
                    n_streams=1,
                )
                print("cuML (GPU) RandomForestClassifier를 사용합니다.")
            except (ImportError, Exception) as e:
                print(f"경고: cuML RandomForestClassifier 로딩 실패 ({e}).")
                print("대안으로 XGBoost Random Forest (GPU)를 사용합니다.")
                
                # XGBoost Random Forest on GPU
                model = xgb.XGBRFClassifier(
                    random_state=42,
                    n_estimators=500,
                    device="cuda",
                    tree_method="hist",
                    subsample=0.8,
                    colsample_bynode=0.8,
                )
                print("XGBoost Random Forest (GPU)를 사용합니다.")
        else:
            raise RuntimeError("GPU 사용이 강제되었으나 GPU 모드가 아닙니다.")
    elif model_name == "extratrees":
        if use_gpu:
            # Try to use cuml ExtraTrees
            try:
                from cuml.ensemble import ExtraTreesClassifier as CumlExtraTreesClassifier
                model = CumlExtraTreesClassifier(
                    random_state=42,
                    n_estimators=500,
                    max_depth=16,
                    n_streams=1,
                )
                print("cuML (GPU) ExtraTreesClassifier를 사용합니다.")
            except ImportError:
                print("경고: cuML에서 ExtraTreesClassifier를 찾을 수 없습니다.")
                print("대안으로 XGBoost Random Forest (GPU)를 사용하여 ExtraTrees를 대체합니다.")
                
                # XGBoost Random Forest on GPU
                # ExtraTrees와 유사하게 동작하도록 파라미터 설정 (랜덤 포레스트 모드)
                model = xgb.XGBRFClassifier(
                    random_state=42,
                    n_estimators=500,
                    device="cuda",
                    tree_method="hist",
                    subsample=0.8, # 약간의 무작위성 추가
                    colsample_bynode=0.8,
                )
                print("XGBoost Random Forest (GPU)를 ExtraTrees 대용으로 실행합니다.")
        else:
            # GPU 모드가 아닌 경우 (여기까지 도달하면 안됨, 강제 종료 예정)
            raise RuntimeError("GPU 사용이 강제되었으나 GPU 모드가 아닙니다.")
    else:
        raise ValueError("지원하지 않는 모델 이름입니다: lightgbm, xgboost, randomforest, extratrees 중 선택하세요.")

    # 모델 학습
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"학습 완료! (소요 시간: {training_time:.2f}초)")

    # 모델 평가
    print("\n모델 평가 중...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # zero_division=0 추가

    print(f"\n--- {model_name} 모델 성능 ---")
    print(f"정확도: {accuracy:.4f}")
    print("\n분류 리포트:")
    print(report)
    print("---------------------------\n")

    return model, accuracy, training_time, report





def save_model(model, model_name, accuracy):

    """

    학습된 모델을 저장합니다.

    """

    model_dirs = {

        'lightgbm': '03_models/01_lgbm',

        'xgboost': '03_models/02_xgb',

        'randomforest': '03_models/04_rf',

        'extratrees': '03_models/03_et'

    }

    save_dir = model_dirs.get(model_name)

    os.makedirs(save_dir, exist_ok=True)

    

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # joblib.dump는 .joblib 확장자를 자동으로 붙이지 않으므로 명시적으로 지정

    filename = f"{model_name}_model_acc_{accuracy:.4f}_{timestamp}.joblib"

    save_path = os.path.join(save_dir, filename)



    joblib.dump(model, save_path)

    print(f"학습된 모델이 '{save_path}'에 저장되었습니다.")

    return save_path



def main(args, use_gpu):

    """

    메인 함수: 데이터 로드, 모델 학습, 평가 및 저장을 수행합니다.

    """

    print(f"\n{'='*60}")

    print(f"트리 기반 모델 학습 스크립트 시작 - 모델: {args.model}")

    print(f"{'='*60}")

    print(f"학습 디바이스: {'GPU' if use_gpu else 'CPU'}")

    

    # 데이터 로드 (전체 데이터셋 사용)

    processed_data_file = '02_data/01_processed/preprocessed_transactions_v2.csv'

    # 빠른 테스트를 위해 nrows를 50만으로 제한. 실제 학습 시 None으로 변경하여 전체 데이터 사용

    df_processed = load_processed_data(processed_data_file, nrows=500000)



    # 특성(X)과 타겟(y) 분리

    features_cols = [col for col in df_processed.columns if col.endswith('_scaled') or col == 'Is_Weekend']

    target_col = 'Category_encoded'

    X = df_processed[features_cols]

    y = df_processed[target_col]



    # 학습/테스트 데이터 분리

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    

    # 모델 학습, 평가, 저장

    model, accuracy, training_time, report = train_model(
        args.model, X_train, y_train, X_test, y_test, use_gpu
    )

    saved_model_path = save_model(model, args.model, accuracy)



    print(f"\n{'='*60}")

    print(f"모델 학습 스크립트 완료 - 모델: {args.model}")

    print(f"{'='*60}")

    

    return {

        'model_name': args.model,

        'accuracy': accuracy,

        'training_time': training_time,

        'saved_model_path': saved_model_path

    }



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="트리 기반 모델 학습 스크립트")

    parser.add_argument(

        '--model', 

        type=str, 

        required=True, 

        choices=['lightgbm', 'xgboost', 'randomforest', 'extratrees'],

        help="학습할 모델을 선택합니다 (lightgbm, xgboost, randomforest, extratrees)."

    )

    parser.add_argument(
        '--device',
        choices=['auto', 'gpu', 'cpu'],
        default='auto',
        help="실행 디바이스를 선택합니다. (auto: GPU가 있으면 사용, gpu: 무조건 GPU, cpu: CPU 강제).",
    )

    args = parser.parse_args()

    gpu_available = is_gpu_available()
    print(f"GPU 감지 결과: {'사용 가능' if gpu_available else '감지되지 않음'} (--device={args.device})")
    try:
        use_gpu = resolve_gpu_mode(args.device, gpu_available)
    except RuntimeError as exc:
        parser.error(str(exc))

    main(args, use_gpu)
