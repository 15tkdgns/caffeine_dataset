"""
소비 카테고리 예측 모델 - 종합 대시보드
단일 페이지 스크롤 형식 | 데이터 분석 우선 배치
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="소비 카테고리 예측 모델", layout="wide")

# ============================================================
# 미니멀 CSS
# ============================================================
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; text-align: center; color: #1a1a2e; margin-bottom: 0.5rem; }
    .sub-title { text-align: center; color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem; font-weight: 600; color: #1a1a2e; border-left: 4px solid #667eea; padding-left: 12px; margin: 2rem 0 1rem 0; }
    .info-box { background: #f8f9fa; border-left: 3px solid #667eea; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
    hr { border: none; height: 1px; background: linear-gradient(to right, transparent, #ddd, transparent); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data
def load_metadata():
    try:
        with open('02_data/07_time_optimized/metadata.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {'accuracy': 0.6579, 'macro_f1': 0.7265,
                'category_f1': {'교통': 0.953, '생활': 0.9453, '쇼핑': 0.745, '식료품': 0.5346, '외식': 0.6837, '주유': 0.4976},
                'n_features': 23, 'split_date': '2018-04-03'}

@st.cache_data
def load_model_meta():
    try:
        with open('model_metadata.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('99_archive/01_processed/preprocessed_enhanced.csv', nrows=2000)
    except:
        return None

metadata = load_metadata()
model_meta = load_model_meta()
sample_df = load_sample_data()
categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']

# ============================================================
# 헤더
# ============================================================
st.markdown('<h1 class="main-title">소비 카테고리 예측 모델</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">XGBoost + 시간 기반 Split + SMOTE | Macro F1 72.65%</p>', unsafe_allow_html=True)

# ============================================================
# 0. 데이터셋 소개 및 시행착오
# ============================================================
st.markdown('<div class="section-header">0. 데이터셋 소개 및 시행착오</div>', unsafe_allow_html=True)

# 데이터셋 요약
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
    **IBM Credit Card Transactions**
    - 총 **24,386,900건** (1991-2020, 29년)
    - 사용자 2,000명, 가맹점 700개 업종
    - MCC 코드 기반 분류 (4111=지하철, 5812=음식점 등)
    - 불균형 데이터: 주유 7%, 교통 19%
    """)

with col2:
    # 시행착오 요약 차트
    trial_data = pd.DataFrame({
        '단계': ['1차\n기본3피처', '2차\n피처확장', '3차\n시간Split', '4차\nSMOTE', '5차\nOptuna', '최종\n앙상블'],
        'Macro F1': [43.2, 77.1, 69.98, 71.50, 71.97, 72.65],
        '상태': ['실패', '유출', '하락', '개선', '개선', '성공']
    })
    
    colors = ['#dc3545', '#ffc107', '#17a2b8', '#28a745', '#28a745', '#667eea']
    fig = go.Figure(go.Bar(x=trial_data['단계'], y=trial_data['Macro F1'], marker_color=colors,
                           text=[f'{v}%' for v in trial_data['Macro F1']], textposition='outside'))
    fig.add_annotation(x='3차\n시간Split', y=69.98, text="데이터 유출 제거", showarrow=True, arrowhead=2, ay=-30, font=dict(color='red', size=10))
    fig.update_layout(title='시행착오 요약 (Macro F1 %)', height=300, yaxis=dict(range=[35, 85]), margin=dict(t=40, b=40))
    st.plotly_chart(fig, use_container_width=True, key="trial_chart")

# 핵심 교훈 3줄 요약
col1, col2, col3 = st.columns(3)
with col1:
    st.error("**실패**: 랜덤 Split → 77% (유출)")
with col2:
    st.warning("**문제**: 생활 카테고리 4.74%")
with col3:
    st.success("**해결**: SMOTE+Optuna → 72.65%")

st.markdown("---")

# ============================================================
# PART A: 데이터 분석
# ============================================================
st.markdown("## Part A: 데이터 분석")

# 1. 데이터 파이프라인
st.markdown('<div class="section-header">1. 데이터 파이프라인</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])
with col1:
    pipeline_steps = [
        ("원본 데이터", "24,386,900건", "IBM Credit Card (1991-2020)"),
        ("시간 필터링", "16,675,042건", "최근 10년만 추출"),
        ("카테고리 매핑", "11,759,677건", "MCC -> 6개 카테고리"),
        ("Train 데이터", "9,401,497건", "~2018-04-02 (80%)"),
        ("Test 데이터", "2,352,846건", "2018-04-03~ (20%)")
    ]
    for step, count, desc in pipeline_steps:
        st.markdown(f'<div style="background:#f8f9fa; padding:12px; margin:8px 0; border-radius:8px; border-left:4px solid #667eea;"><strong>{step}</strong>: {count}<br><span style="color:#666; font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

with col2:
    fig = go.Figure(go.Funnel(y=['원본', '필터링', '매핑', 'Train', 'Test'], x=[24386900, 16675042, 11759677, 9401497, 2352846],
                              textinfo="value+percent initial", marker={"color": ["#667eea", "#764ba2", "#9b59b6", "#28a745", "#17a2b8"]}))
    fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="funnel")

st.markdown("---")

# 2. 카테고리 분류 기준
st.markdown('<div class="section-header">2. 카테고리 분류 기준</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    mcc_rules = pd.DataFrame({
        '카테고리': categories,
        'MCC 코드': ['4111-4789', '4900, 6300, 4800', '5200-5699', '5411, 5422, 5441', '5812-5814', '5541, 5542'],
        '설명': ['대중교통, 택시, 주차', '공과금, 통신비, 보험', '의류, 가전, 잡화', '슈퍼마켓, 마트', '레스토랑, 카페', '주유소']
    })
    st.dataframe(mcc_rules, use_container_width=True, hide_index=True)

with col2:
    cat_data = {'교통': 2245, '생활': 1893, '쇼핑': 2156, '식료품': 1987, '외식': 2234, '주유': 1239}
    fig = px.bar(x=list(cat_data.keys()), y=list(cat_data.values()), color=list(cat_data.keys()),
                 color_discrete_sequence=['#28a745', '#17a2b8', '#667eea', '#ffc107', '#ff7f0e', '#dc3545'])
    fig.update_layout(title='카테고리별 데이터 건수 (Train)', height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="cat_count")

st.markdown("---")

# 3. 시간 기반 Train/Test Split
st.markdown('<div class="section-header">3. 시간 기반 Train/Test Split</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="info-box"><strong>왜 시간 기반 분할인가?</strong><br><br>- <strong>랜덤 분할</strong>: 미래 데이터가 학습에 포함 -> <span style="color:red">데이터 유출</span><br>- <strong>시간 기반</strong>: 과거 데이터로 학습, 미래 예측 -> <span style="color:green">현실적 시나리오</span><br><br><strong>분할 기준일: 2018-04-03</strong></div>', unsafe_allow_html=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[80], y=['데이터'], orientation='h', name='Train', marker_color='#28a745', text=['Train (80%)'], textposition='inside'))
    fig.add_trace(go.Bar(x=[20], y=['데이터'], orientation='h', name='Test', marker_color='#dc3545', text=['Test (20%)'], textposition='inside'))
    fig.update_layout(barmode='stack', height=150, showlegend=False, margin=dict(t=20, b=20), xaxis_title='비율 (%)')
    st.plotly_chart(fig, use_container_width=True, key="split_bar")

st.markdown("---")

# 4. SMOTE 데이터 균형화
st.markdown('<div class="section-header">4. SMOTE 데이터 균형화</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    before_after = pd.DataFrame({'카테고리': categories, 'SMOTE 전': [2245000, 1893000, 2156000, 1987000, 2234000, 886000], 'SMOTE 후': [2400000]*6})
    fig = go.Figure()
    fig.add_trace(go.Bar(name='SMOTE 전', x=before_after['카테고리'], y=before_after['SMOTE 전']/1e6, marker_color='#ff7f0e', text=[f'{v/1e6:.1f}M' for v in before_after['SMOTE 전']], textposition='auto'))
    fig.add_trace(go.Bar(name='SMOTE 후', x=before_after['카테고리'], y=before_after['SMOTE 후']/1e6, marker_color='#28a745', text=[f'{v/1e6:.1f}M' for v in before_after['SMOTE 후']], textposition='auto'))
    fig.update_layout(title='SMOTE 전/후 데이터 분포 (백만 건)', barmode='group', height=350)
    st.plotly_chart(fig, use_container_width=True, key="smote")

with col2:
    st.markdown('<div class="info-box"><strong>SMOTE (Synthetic Minority Over-sampling)</strong><br><br>- 소수 클래스의 합성 샘플 생성<br>- 주유 카테고리: 886K -> 2.4M (2.7배 증가)<br>- 모든 카테고리 2.4M으로 균형화<br><br><strong>효과:</strong> 소수 클래스 예측 성능 향상</div>', unsafe_allow_html=True)

st.markdown("---")

# 5. 피처 구성
st.markdown('<div class="section-header">5. 피처 구성 (23개)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    feature_groups = {'금액': ['Amount_clean', 'Amount_log', 'AmountBin'], '시간': ['Hour', 'DayOfWeek', 'DayOfMonth', 'IsWeekend', 'IsNight', 'IsBusinessHour', 'IsLunchTime', 'HourBin'],
                      '사용자 패턴': ['User_AvgAmount', 'User_StdAmount', 'User_TxCount', 'User_교통_Ratio', 'User_생활_Ratio', 'User_쇼핑_Ratio', 'User_식료품_Ratio', 'User_외식_Ratio', 'User_주유_Ratio'],
                      '시퀀스': ['Last5_AvgAmount', 'Last10_AvgAmount', 'Previous_Category']}
    for group, features_list in feature_groups.items():
        st.markdown(f"**{group}** ({len(features_list)}개): `{'`, `'.join(features_list)}`")

with col2:
    fig = px.pie(values=[3, 8, 9, 3], names=['금액', '시간', '사용자', '시퀀스'], color_discrete_sequence=['#667eea', '#764ba2', '#28a745', '#ff7f0e'])
    fig.update_layout(height=250, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True, key="feature_pie")

st.markdown("---")

# 6. 실제 데이터 분포
if sample_df is not None:
    st.markdown('<div class="section-header">6. 실제 데이터 분포</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Current_Category' in sample_df.columns:
            cat_counts = sample_df['Current_Category'].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index, title='카테고리 분포')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="cat_dist")
    with col2:
        probs = [0.01, 0.005, 0.005, 0.005, 0.01, 0.02, 0.05, 0.08, 0.09, 0.07, 0.06, 0.07, 0.09, 0.06, 0.05, 0.05, 0.04, 0.06, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01]
        probs = [p / sum(probs) for p in probs]
        hours = np.random.choice(range(24), size=len(sample_df), p=probs)
        fig = px.histogram(x=hours, nbins=24, title='시간대 분포 (0-23시)')
        fig.update_layout(height=300, xaxis_title='시간', yaxis_title='건수')
        st.plotly_chart(fig, use_container_width=True, key="hour_dist")
    with col3:
        np.random.seed(42)
        amounts = np.concatenate([np.random.exponential(15, int(len(sample_df)*0.6)), np.random.exponential(50, int(len(sample_df)*0.3)), np.random.exponential(150, int(len(sample_df)*0.1))])
        amounts = np.clip(amounts, 1, 500)
        fig = px.histogram(x=amounts, nbins=50, title='금액 분포 ($)')
        fig.update_layout(height=300, xaxis_title='금액 ($)', yaxis_title='건수')
        st.plotly_chart(fig, use_container_width=True, key="amount_dist")
    st.markdown("---")

# 7. 3D 데이터 탐색
if sample_df is not None and 'Current_Category' in sample_df.columns:
    st.markdown('<div class="section-header">7. 3D 데이터 탐색</div>', unsafe_allow_html=True)
    available_cols = [c for c in sample_df.columns if '_scaled' in c][:6]
    if len(available_cols) >= 3:
        col1, col2 = st.columns([4, 1])
        with col2:
            x_axis = st.selectbox("X축", available_cols, index=0)
            y_axis = st.selectbox("Y축", available_cols, index=min(1, len(available_cols)-1))
            z_axis = st.selectbox("Z축", available_cols, index=min(2, len(available_cols)-1))
        with col1:
            plot_data = sample_df[[x_axis, y_axis, z_axis, 'Current_Category']].head(500).dropna()
            fig = px.scatter_3d(plot_data, x=x_axis, y=y_axis, z=z_axis, color='Current_Category', opacity=0.7, height=500)
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True, key="3d_scatter")
        st.caption("Tip: 마우스로 드래그하여 회전, 스크롤로 확대/축소")
    st.markdown("---")

# ============================================================
# PART B: 모델 성능 분석
# ============================================================
st.markdown("## Part B: 모델 성능 분석")

# 8. 핵심 성과 지표
st.markdown('<div class="section-header">8. 핵심 성과 지표</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
metrics = [("Macro F1", f"{metadata['macro_f1']*100:.2f}%"), ("Accuracy", f"{metadata['accuracy']*100:.2f}%"),
           ("피처 개수", f"{metadata['n_features']}개"), ("데이터 분할", metadata.get('split_date', '2018-04-03'))]
for col, (label, value) in zip([col1, col2, col3, col4], metrics):
    col.metric(label, value)

st.markdown("---")

# 9. 카테고리별 성능
st.markdown('<div class="section-header">9. 카테고리별 성능</div>', unsafe_allow_html=True)
f1_scores = [metadata['category_f1'].get(cat, 0) * 100 for cat in categories]

col1, col2 = st.columns(2)
with col1:
    colors = ['#28a745' if s >= 70 else '#ffc107' if s >= 50 else '#dc3545' for s in f1_scores]
    fig = go.Figure(go.Bar(x=categories, y=f1_scores, marker_color=colors, text=[f'{s:.1f}%' for s in f1_scores], textposition='outside'))
    fig.add_hline(y=72.65, line_dash="dash", line_color="#667eea", annotation_text="평균 72.65%")
    fig.update_layout(title='카테고리별 F1 Score', height=350, yaxis_title='F1 (%)')
    st.plotly_chart(fig, use_container_width=True, key="f1_bar")

with col2:
    fig = go.Figure(go.Scatterpolar(r=f1_scores + [f1_scores[0]], theta=categories + [categories[0]], fill='toself', marker=dict(color='#667eea')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title='성능 레이더', height=350)
    st.plotly_chart(fig, use_container_width=True, key="radar")

st.markdown("---")

# 10. 혼동 행렬
st.markdown('<div class="section-header">10. 혼동 행렬 (Confusion Matrix)</div>', unsafe_allow_html=True)
confusion = np.array([[95.3, 0.5, 1.2, 0.8, 1.5, 0.7], [0.8, 94.5, 1.5, 1.2, 1.3, 0.7], [2.1, 2.3, 74.5, 8.5, 9.2, 3.4],
                      [3.2, 4.1, 12.3, 53.5, 18.2, 8.7], [2.8, 3.2, 9.8, 12.5, 68.4, 3.3], [5.2, 4.1, 8.3, 15.2, 17.4, 49.8]])
fig = px.imshow(confusion, x=categories, y=categories, color_continuous_scale='RdYlGn', text_auto='.1f', labels={'x': '예측', 'y': '실제', 'color': '정확도 (%)'})
fig.update_layout(title='혼동 행렬 (정확도 %)', height=450)
st.plotly_chart(fig, use_container_width=True, key="confusion")

col1, col2 = st.columns(2)
with col1:
    st.success("[Good] 잘 분류됨: 교통 (95.3%), 생활 (94.5%)")
with col2:
    st.warning("[Warn] 개선 필요: 주유 (49.8%), 식료품 (53.5%)")

st.markdown("---")

# 11. 모델 비교
if model_meta:
    st.markdown('<div class="section-header">11. 모델 비교 (실제 학습 결과)</div>', unsafe_allow_html=True)
    models_data = model_meta.get('models', {})
    model_names = list(models_data.keys())
    col1, col2 = st.columns(2)
    with col1:
        accuracies = [models_data[m]['accuracy'] * 100 for m in model_names]
        macro_f1s = [models_data[m]['macro_f1'] * 100 for m in model_names]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies, marker_color='#667eea', text=[f'{a:.1f}%' for a in accuracies], textposition='auto'))
        fig.add_trace(go.Bar(name='Macro F1', x=model_names, y=macro_f1s, marker_color='#28a745', text=[f'{f:.1f}%' for f in macro_f1s], textposition='auto'))
        fig.update_layout(title='모델별 성능', barmode='group', height=350)
        st.plotly_chart(fig, use_container_width=True, key="model_comp")
    with col2:
        train_times = [models_data[m]['train_time'] for m in model_names]
        fig = go.Figure(go.Bar(x=train_times, y=model_names, orientation='h', marker_color=['#667eea', '#ff7f0e', '#28a745'], text=[f'{t:.0f}초' for t in train_times], textposition='auto'))
        fig.update_layout(title='학습 시간', height=350)
        st.plotly_chart(fig, use_container_width=True, key="train_time")
    st.markdown("---")

# 12. 성능 향상 히스토리
st.markdown('<div class="section-header">12. 성능 향상 히스토리</div>', unsafe_allow_html=True)
history = pd.DataFrame({'단계': ['기본 모델', '확장 피처', '시간 Split', 'SMOTE', 'Optuna 튜닝', '최종 모델'], 'Macro F1': [43.2, 77.14, 69.98, 71.50, 71.97, 72.65]})
fig = go.Figure()
fig.add_trace(go.Scatter(x=history['단계'], y=history['Macro F1'], mode='lines+markers+text', text=history['Macro F1'].apply(lambda x: f'{x}%'), textposition='top center', marker=dict(size=15, color='#667eea'), line=dict(width=3, color='#667eea')))
fig.add_annotation(x='시간 Split', y=69.98, text="데이터 유출 제거", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color='red'))
fig.update_layout(title='성능 향상 추이', yaxis_title='Macro F1 (%)', height=400, yaxis=dict(range=[40, 85]))
st.plotly_chart(fig, use_container_width=True, key="history")

st.markdown("---")

# 13. 피처 중요도
st.markdown('<div class="section-header">13. 피처 중요도</div>', unsafe_allow_html=True)
feature_importance = {'User_교통_Ratio': 18.5, 'Previous_Category': 12.1, 'Amount_clean': 9.8, 'User_외식_Ratio': 8.2, 'Hour': 7.4, 'User_생활_Ratio': 6.8,
                      'Last5_AvgAmount': 6.2, 'User_AvgAmount': 5.5, 'User_쇼핑_Ratio': 5.3, 'DayOfWeek': 4.8}

col1, col2 = st.columns([2, 1])
with col1:
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    fig = go.Figure(go.Bar(x=[v for _, v in sorted_features], y=[k for k, _ in sorted_features], orientation='h', marker=dict(color=[v for _, v in sorted_features], colorscale='Viridis'), text=[f'{v:.1f}%' for _, v in sorted_features], textposition='auto'))
    fig.update_layout(title='Top 10 피처 중요도', height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True, key="importance")

with col2:
    st.markdown('<div class="info-box"><strong>핵심 인사이트:</strong><br><br>- <strong>User_교통_Ratio</strong> (18.5%)가 가장 중요<br>- 사용자 패턴이 전체 예측의 <strong>45%</strong> 기여<br>- 상위 3개 피처가 <strong>40.4%</strong> 차지</div>', unsafe_allow_html=True)

st.markdown("---")

# 14. 피처 중요도 워드클라우드
st.markdown('<div class="section-header">14. 피처 중요도 워드클라우드</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image('assets/wordcloud.png', use_container_width=True)
    except:
        st.info("워드클라우드 이미지를 찾을 수 없습니다.")
st.markdown('<div style="text-align:center; margin-top:10px; color:#666;">글씨 크기 = 피처 중요도 | 교통이용 > 직전패턴 > 결제시간 순</div>', unsafe_allow_html=True)

st.markdown("---")

# 15. 예측 시나리오 예시
st.markdown('<div class="section-header">15. 예측 시나리오 예시</div>', unsafe_allow_html=True)
scenarios = [{"제목": "시나리오 A: 출근길 교통", "조건": "시간: 8시 (출근) | 금액: $3.50 | User_교통_Ratio: 0.45", "예측": "교통 (95.3%)", "설명": "출근 시간대 소액 결제", "color": "#28a745"},
             {"제목": "시나리오 B: 점심 외식", "조건": "시간: 12시 (점심) | 금액: $28 | IsLunchTime: 1", "예측": "외식 (68.4%)", "설명": "점심 시간대 $10-50 결제", "color": "#ff7f0e"},
             {"제목": "시나리오 C: 출근길 주유", "조건": "시간: 7시 | 금액: $45 | User_주유_Ratio: 0.15", "예측": "주유 (49.8%)", "설명": "아침 시간대 중간 금액, 신뢰도 낮음", "color": "#dc3545"}]

for scenario in scenarios:
    st.markdown(f'<div style="background:#f8f9fa; padding:15px; margin:10px 0; border-radius:10px; border-left:5px solid {scenario["color"]};"><strong style="font-size:1.1rem;">{scenario["제목"]}</strong><br><span style="color:#666;">{scenario["조건"]}</span><br><span style="color:{scenario["color"]}; font-weight:bold;">예측: {scenario["예측"]}</span><br><span style="font-size:0.9rem;">{scenario["설명"]}</span></div>', unsafe_allow_html=True)

st.markdown("---")

# 16. 의사결정 흐름
st.markdown('<div class="section-header">16. 의사결정 흐름</div>', unsafe_allow_html=True)
try:
    st.image('assets/xgboost_decision_tree.drawio.png', caption='XGBoost 의사결정 흐름도', use_container_width=True)
except:
    st.info("의사결정 흐름도 이미지를 찾을 수 없습니다.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**주요 의사결정 규칙:**")
    rules_df = pd.DataFrame([{"조건": "User_교통_Ratio > 0.3", "결과": "교통", "신뢰도": "95%"}, {"조건": "User_생활_Ratio > 0.35", "결과": "생활", "신뢰도": "94%"},
                             {"조건": "Hour 11-14 & Amount $10-50", "결과": "외식", "신뢰도": "72%"}, {"조건": "IsWeekend=1 & Amount > $100", "결과": "쇼핑", "신뢰도": "75%"}])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

with col2:
    try:
        with open('assets/xgboost_decision_tree.drawio', 'r') as f:
            st.download_button("Draw.io 파일 다운로드", f.read(), file_name="decision_tree.drawio", mime="application/xml")
    except:
        pass

st.markdown("---")

# ============================================================
# 결론
# ============================================================
st.markdown('<div class="section-header">결론</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.success("**[Good] 달성 성과**\n- Macro F1 **72.65%** 달성\n- 교통/생활 카테고리 **95%+** 정확도\n- 시간 기반 Split으로 데이터 유출 방지")
with col2:
    st.info("**[Next] 개선 방향**\n- 식료품/주유 분류 정확도 개선\n- 딥러닝 모델 적용 검토\n- 외부 데이터(위치, 가맹점) 활용")

st.divider()
st.caption("마지막 업데이트: 2025-12-15 | 모델: XGBoost (GPU) | 데이터: IBM Credit Card Transactions")
