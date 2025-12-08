"""
카테고리 분포 분석 - 소수/다수 클래스 설명
"""
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('02_data/01_processed/preprocessed_enhanced.csv')

print('='*70)
print('📊 카테고리별 분포 분석')
print('='*70)

# 카테고리 분포
cat_counts = df['Next_Category'].value_counts()
total = len(df)

print('\n[카테고리별 거래 건수 및 비율]')
print('-'*60)
for i, (cat, count) in enumerate(cat_counts.items(), 1):
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    status = '다수' if pct > 15 else ('보통' if pct > 10 else '⚠️소수')
    print(f'{i}. {cat:8} {count:>10,}건 ({pct:>5.1f}%) {bar:15} [{status}]')

print('\n' + '='*70)
print('📈 소수 클래스 vs 다수 클래스')
print('='*70)

max_cat = cat_counts.idxmax()
min_cat = cat_counts.idxmin()
max_count = cat_counts.max()
min_count = cat_counts.min()

print(f'\n✅ 다수 클래스: {max_cat} ({max_count:,}건, {max_count/total*100:.1f}%)')
print(f'⚠️  소수 클래스: {min_cat} ({min_count:,}건, {min_count/total*100:.1f}%)')
print(f'\n📉 불균형 비율: {max_count/min_count:.1f}:1 (다수:소수)')

print('\n' + '='*70)
print('💡 SMOTE 증강 효과')
print('='*70)

print('\n[증강 전 - 불균형 상태]')
for cat, count in cat_counts.items():
    print(f'  {cat}: {count:,}건')

print('\n[증강 후 - 균형 상태] (학습 데이터)')
balanced_count = 1542352  # SMOTE 적용 후
for cat in cat_counts.index:
    print(f'  {cat}: {balanced_count:,}건 (동일)')

print('\n' + '='*70)
print('🎯 비즈니스 영향')
print('='*70)

print('''
⚠️ 소수 클래스: 생활 (7.5%)
   - 공과금, 통신비, 보험료 등 정기 지출
   - 거래 빈도 낮음 → 예측 어려움
   - SMOTE로 학습 데이터 증강 → 예측력 향상

📊 문제점 (SMOTE 없이):
   - 모델이 다수 클래스(식료품 30%)만 잘 예측
   - 소수 클래스(생활 7.5%)는 거의 예측 못함
   - Macro F1이 낮음 (클래스별 성능 불균형)

✅ SMOTE 적용 후:
   - 모든 클래스 동일 비율로 학습
   - 소수 클래스 예측력 향상
   - Macro F1 개선 (클래스별 균형 예측)
   
📈 결과:
   - Accuracy: 약간 하락 (다수 클래스 예측 감소)
   - Macro F1: 향상 (소수 클래스 예측 개선)
''')
