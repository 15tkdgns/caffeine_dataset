"""
Top 3 모델 상세 분석 스크립트
최고 성능 모델 3개에 대한 심층 분석
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def load_comparison_results(results_file='03_models/comparison/gpu_models_results.json'):
    """비교 결과 로드"""
    if not os.path.exists(results_file):
        print(f"❌ 결과 파일을 찾을 수 없습니다: {results_file}")
        print("먼저 모델 비교를 실행하세요:")
        print("  python3 01_src/01_training/10_compare_gpu_models.py")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def select_top3_models(results):
    """Top 3 모델 선정 (Accuracy 기준)"""
    # DataFrame으로 변환
    df = pd.DataFrame(results).T
    
    # Accuracy 기준 정렬
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    # Top 3
    top3 = df_sorted.head(3)
    
    print("="*70)
    print("Top 3 모델 (Accuracy 기준)")
    print("="*70)
    
    for idx, (model_name, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{idx}. {model_name}")
        print(f"   Accuracy: {row['accuracy']:.4f}")
        print(f"   Macro F1: {row['macro_f1']:.4f}")
        print(f"   Weighted F1: {row['weighted_f1']:.4f}")
        print(f"   Device: {row['device']}")
        print(f"   Framework: {row['framework']}")
        print(f"   학습 시간: {row['train_time']:.2f}초")
        print(f"   예측 시간: {row['predict_time']:.2f}초")
    
    return top3


def analyze_category_performance(results, top3_models):
    """카테고리별 성능 분석"""
    print("\n" + "="*70)
    print("카테고리별 F1 Score 비교")
    print("="*70)
    
    category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    
    # 카테고리별 F1 Score 데이터 수집
    category_df = pd.DataFrame()
    
    for model_name in top3_models.index:
        f1_scores = results[model_name]['category_f1']
        category_df[model_name] = f1_scores[:len(category_names)]
    
    category_df.index = category_names
    
    print("\n", category_df.to_string())
    
    # 각 카테고리에서 최고 성능 모델
    print("\n" + "-"*70)
    print("각 카테고리별 최고 성능 모델")
    print("-"*70)
    
    for category in category_names:
        best_model = category_df.loc[category].idxmax()
        best_f1 = category_df.loc[category].max()
        print(f"{category:8s}: {best_model:30s} (F1={best_f1:.4f})")
    
    return category_df


def create_detailed_report(top3_models, category_df, output_dir='05_docs'):
    """상세 분석 리포트 생성"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'TOP3_MODELS_ANALYSIS.md')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 🏆 Top 3 모델 상세 분석\n\n")
        f.write("## 📊 종합 성능 비교\n\n")
        
        # 종합 성능 표
        f.write("| 순위 | 모델 | Accuracy | Macro F1 | Weighted F1 | Device | 학습 시간 |\n")
        f.write("|------|------|----------|----------|-------------|--------|----------|\n")
        
        for idx, (model_name, row) in enumerate(top3_models.iterrows(), 1):
            f.write(f"| {idx} | **{model_name}** | ")
            f.write(f"{row['accuracy']:.4f} | ")
            f.write(f"{row['macro_f1']:.4f} | ")
            f.write(f"{row['weighted_f1']:.4f} | ")
            f.write(f"{row['device']} | ")
            f.write(f"{row['train_time']:.1f}초 |\n")
        
        # 각 모델 상세 분석
        f.write("\n---\n\n")
        f.write("## 🔍 모델별 상세 분석\n\n")
        
        for idx, (model_name, row) in enumerate(top3_models.iterrows(), 1):
            f.write(f"### {idx}. {model_name}\n\n")
            
            # 기본 정보
            f.write("**기본 정보:**\n")
            f.write(f"- Framework: `{row['framework']}`\n")
            f.write(f"- Device: `{row['device']}`\n")
            f.write(f"- 학습 시간: {row['train_time']:.2f}초\n")
            f.write(f"- 예측 시간: {row['predict_time']:.2f}초\n\n")
            
            # 성능 지표
            f.write("**성능 지표:**\n")
            f.write(f"- Accuracy: **{row['accuracy']:.4f}** ({row['accuracy']*100:.2f}%)\n")
            f.write(f"- Macro F1: **{row['macro_f1']:.4f}**\n")
            f.write(f"- Weighted F1: **{row['weighted_f1']:.4f}**\n\n")
            
            # 카테고리별 F1
            f.write("**카테고리별 F1 Score:**\n\n")
            f.write("| 카테고리 | F1 Score | 비고 |\n")
            f.write("|---------|----------|------|\n")
            
            category_names = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
            for cat_idx, category in enumerate(category_names):
                f1_score = category_df.loc[category, model_name]
                is_best = (f1_score == category_df.loc[category].max())
                marker = " 🏆" if is_best else ""
                f.write(f"| {category} | {f1_score:.4f} | {marker} |\n")
            
            # 장단점 (모델별 특징)
            f.write("\n**장점:**\n")
            if 'XGBoost' in model_name:
                f.write("- GPU 가속으로 빠른 학습 속도\n")
                f.write("- 그래디언트 부스팅의 강력한 성능\n")
                f.write("- 하이퍼파라미터 튜닝 용이\n")
            elif 'CatBoost' in model_name:
                f.write("- 범주형 변수 처리 우수\n")
                f.write("- 과적합 방지 기능 내장\n")
                f.write("- 디폴트 파라미터로도 좋은 성능\n")
            elif 'RandomForest' in model_name or 'ExtraTrees' in model_name:
                f.write("- 랜덤성으로 과적합 방지\n")
                f.write("- 클래스 불균형 처리 (class_weight)\n")
                f.write("- 해석 가능한 Feature Importance\n")
            elif 'Neural Network' in model_name:
                f.write("- 복잡한 비선형 패턴 학습\n")
                f.write("- GPU 가속으로 대규모 데이터 처리\n")
                f.write("- 유연한 아키텍처 설계\n")
            
            f.write("\n**단점:**\n")
            if 'XGBoost' in model_name:
                f.write("- 클래스 불균형에 추가 설정 필요\n")
                f.write("- 메모리 사용량 높음\n")
            elif 'CatBoost' in model_name:
                f.write("- 학습 속도가 다소 느림\n")
                f.write("- GPU 메모리 사용량 많음\n")
            elif 'CPU' in model_name:
                f.write("- CPU 기반으로 학습 속도 느림\n")
                f.write("- 대규모 데이터 처리 제한적\n")
            elif 'Neural Network' in model_name:
                f.write("- 하이퍼파라미터 튜닝 복잡\n")
                f.write("- 해석 가능성 낮음\n")
            
            f.write("\n---\n\n")
        
        # 추천 사항
        f.write("## 💡 추천 사항\n\n")
        best_model = top3_models.index[0]
        best_acc = top3_models.iloc[0]['accuracy']
        
        f.write(f"### 프로덕션 배포 추천: **{best_model}**\n\n")
        f.write(f"- **이유**: 가장 높은 Accuracy ({best_acc:.4f})\n")
        f.write(f"- **장점**: {top3_models.iloc[0]['device']} 사용으로 빠른 예측\n")
        f.write(f"- **고려사항**: 학습 시간 {top3_models.iloc[0]['train_time']:.1f}초\n\n")
        
        f.write("### 상황별 추천\n\n")
        f.write("1. **속도 우선**: 학습 시간이 가장 짧은 모델 선택\n")
        f.write("2. **정확도 우선**: Accuracy가 가장 높은 모델 (현재 1위 모델)\n")
        f.write("3. **균형 중시**: Macro F1이 높은 모델 (소수 클래스 성능 고려)\n\n")
    
    print(f"\n✅ 상세 리포트 저장: {output_file}")
    return output_file


def main():
    """메인 실행"""
    print("="*70)
    print("Top 3 모델 상세 분석")
    print("="*70)
    
    # 결과 로드
    print("\n[1/3] 모델 비교 결과 로드")
    results = load_comparison_results()
    
    if not results:
        return
    
    print(f"총 {len(results)}개 모델 결과 로드됨")
    
    # Top 3 선정
    print("\n[2/3] Top 3 모델 선정")
    top3_models = select_top3_models(results)
    
    # 카테고리별 성능 분석
    print("\n[3/3] 카테고리별 성능 분석")
    category_df = analyze_category_performance(results, top3_models)
    
    # 상세 리포트 생성
    print("\n[4/4] 상세 리포트 생성")
    report_file = create_detailed_report(top3_models, category_df)
    
    print("\n" + "="*70)
    print("분석 완료!")
    print("="*70)
    print(f"\n상세 리포트: {report_file}")
    print("\n다음 단계: 리포트 확인 후 최적 모델 선택")


if __name__ == '__main__':
    main()
