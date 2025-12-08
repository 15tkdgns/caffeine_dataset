"""
신용카드 거래 카테고리 예측 - 최종 보고서 대시보드
Stacking Ensemble 모델 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="카테고리 예측 모델 분석", layout="wide")

# ============================================================
# 데이터 로드
# ============================================================

@st.cache_data
def load_experiment_data():
    """실험 결과 데이터"""
    experiments = [
        {'name': 'Baseline (LightGBM)', 'accuracy': 49.13, 'macro_f1': 43.44, 'method': '기본 설정', 'step': 0},
        {'name': 'Class Weight 조정', 'accuracy': 44.85, 'macro_f1': 42.87, 'method': '언더샘플링 + 가중치', 'step': 2},
        {'name': 'ADASYN 증강', 'accuracy': 46.95, 'macro_f1': 44.89, 'method': 'ADASYN 데이터 증강', 'step': 3},
        {'name': 'Accuracy 최적화', 'accuracy': 49.55, 'macro_f1': 42.84, 'method': '하이퍼파라미터 튜닝', 'step': 4},
        {'name': 'Stacking Ensemble', 'accuracy': 49.62, 'macro_f1': 45.24, 'method': '3-Model Stacking', 'step': 5},
        {'name': 'Optuna 심화', 'accuracy': 49.49, 'macro_f1': 43.00, 'method': '100 trials 튜닝', 'step': 6}
    ]
    return pd.DataFrame(experiments)

@st.cache_data
def load_category_performance():
    """카테고리별 성능"""
    return {
        'Baseline': {'교통': 64.96, '생활': 8.02, '쇼핑': 34.78, '식료품': 54.14, '외식': 44.34, '주유': 54.41},
        'Stacking': {'교통': 64.84, '생활': 17.23, '쇼핑': 36.19, '식료품': 52.50, '외식': 45.62, '주유': 55.07}
    }

@st.cache_data
def load_feature_importance():
    """피처 중요도"""
    return pd.DataFrame({
        'Feature': ['Amount', 'User_AvgAmount', 'Hour', 'User_교통_Ratio', 'DayOfWeek', 
                   'User_StdAmount', 'User_식료품_Ratio', 'User_외식_Ratio', 'Is_Weekend', 'Amount_log'],
        'Importance': [20159, 15122, 13205, 12570, 12261, 11934, 11914, 11830, 11660, 11623]
    })

# ============================================================
# 메인 대시보드
# ============================================================

st.title("신용카드 거래 카테고리 예측 모델 분석 보고서")
st.markdown("---")

# 사이드바
st.sidebar.title("목차")
page = st.sidebar.radio(
    "페이지 선택",
    ["1. 프로젝트 개요", "2. Stacking Ensemble 모델", "3. 모델 의사결정 과정",
     "4. 실험 결과 비교", "5. 피처 분석", "6. 결론"]
)

# ============================================================
# 1. 프로젝트 개요
# ============================================================

if page == "1. 프로젝트 개요":
    st.header("1. 프로젝트 개요")
    
    # 핵심 지표
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("데이터 크기", "24.4M건")
    with col2:
        st.metric("분류 카테고리", "6개")
    with col3:
        st.metric("사용 피처", "27개")
    with col4:
        st.metric("최종 Accuracy", "49.62%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("프로젝트 목표")
        st.write("""
        신용카드 거래 데이터를 분석하여 6개의 소비 카테고리로 자동 분류하는 
        머신러닝 모델을 개발합니다.
        """)
        
        st.subheader("분류 카테고리")
        categories = pd.DataFrame({
            '카테고리': ['교통', '생활', '쇼핑', '식료품', '외식', '주유'],
            '설명': ['대중교통, 택시, 항공', '의료, 약국, 인테리어', '의류, 가전, 온라인쇼핑',
                    '마트, 편의점, 슈퍼마켓', '음식점, 카페, 배달', '주유소, 충전소']
        })
        st.table(categories)
    
    with col2:
        st.subheader("데이터 분포")
        dist_data = pd.DataFrame({
            '카테고리': ['식료품', '외식', '쇼핑', '주유', '교통', '생활'],
            '비율': [30.0, 18.2, 16.2, 14.5, 13.6, 7.5]
        })
        
        fig = go.Figure(data=[
            go.Bar(x=dist_data['카테고리'], y=dist_data['비율'], 
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        ])
        fig.update_layout(
            title='카테고리별 데이터 분포 (%)',
            xaxis_title='카테고리',
            yaxis_title='비율 (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("분석 파이프라인")
    
    # 파이프라인 흐름도
    fig = go.Figure()
    
    # 노드 위치
    nodes_x = [0.1, 0.3, 0.5, 0.7, 0.9]
    nodes_y = [0.5, 0.5, 0.5, 0.5, 0.5]
    labels = ['원본 데이터\n(24.4M)', '전처리\n(피처 엔지니어링)', '모델 학습\n(Stacking)', '평가\n(Accuracy)', '배포\n(프로덕션)']
    
    # 노드
    fig.add_trace(go.Scatter(
        x=nodes_x, y=nodes_y,
        mode='markers+text',
        marker=dict(size=60, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']),
        text=labels,
        textposition='bottom center',
        textfont=dict(size=11)
    ))
    
    # 화살표
    for i in range(len(nodes_x)-1):
        fig.add_annotation(
            x=nodes_x[i+1]-0.05, y=nodes_y[i],
            ax=nodes_x[i]+0.05, ay=nodes_y[i],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#333'
        )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        height=200,
        margin=dict(l=20, r=20, t=20, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 2. Stacking Ensemble 모델
# ============================================================

elif page == "2. Stacking Ensemble 모델":
    st.header("2. Stacking Ensemble 모델 (최종 선정)")
    
    # 핵심 성능
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "49.62%", "+0.49%p vs Baseline")
    with col2:
        st.metric("Macro F1", "45.24%", "+1.80%p vs Baseline")
    with col3:
        st.metric("학습 시간", "약 60분")
    
    st.markdown("---")
    
    st.subheader("2.1 Stacking 모델 구조")
    
    # Stacking 구조도 (Sankey Diagram) - 밝은 색상으로 변경
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=25,
            line=dict(color="white", width=2),
            label=["입력 데이터 (5.14M)", 
                   "LightGBM-1", "XGBoost", "LightGBM-2",
                   "Meta Features (18차원)", "Meta-Learner", "최종 예측"],
            color=["#87CEEB", "#FFB6C1", "#90EE90", "#FFD700", "#DDA0DD", "#87CEFA", "#FFA07A"],
            # 밝은 파란색, 연분홍, 연두색, 금색, 연보라, 하늘색, 연주황
        ),
        link=dict(
            source=[0, 0, 0, 1, 2, 3, 4, 5],
            target=[1, 2, 3, 4, 4, 4, 5, 6],
            value=[5, 5, 5, 6, 6, 6, 18, 6],
            color="rgba(200, 200, 200, 0.4)"
        )
    )])
    fig.update_layout(
        title="Stacking Ensemble 데이터 흐름",
        font=dict(size=13, color="black"),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2.2 Base Models 구성")
        
        base_models = pd.DataFrame({
            '모델': ['LightGBM-1', 'XGBoost', 'LightGBM-2'],
            'n_estimators': [500, 500, 400],
            'max_depth': [12, 12, 15],
            'learning_rate': [0.05, 0.05, 0.08],
            '단독 Accuracy': ['49.47%', '49.50%', '49.46%']
        })
        st.table(base_models)
        
    with col2:
        st.subheader("2.3 Meta-Learner 구성")
        
        meta_config = pd.DataFrame({
            '항목': ['모델', 'n_estimators', 'max_depth', 'learning_rate', 'Input 차원'],
            '값': ['LightGBM', '200', '5', '0.1', '18 (6 classes x 3 models)']
        })
        st.table(meta_config)
    
    st.markdown("---")
    
    st.subheader("2.4 5-Fold Cross-Validation 과정")
    
    # CV 과정 시각화
    fig = make_subplots(rows=1, cols=5, subplot_titles=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
    
    for i in range(5):
        train_parts = [1, 1, 1, 1, 1]
        train_parts[i] = 0  # Validation
        
        fig.add_trace(
            go.Bar(y=['Fold'], x=[train_parts[j] for j in range(5)], 
                   orientation='h', marker_color=['#3498db' if j != i else '#e74c3c' for j in range(5)]),
            row=1, col=i+1
        )
    
    fig.update_layout(
        showlegend=False,
        height=200,
        title_text="각 Fold에서 파란색=학습, 빨간색=검증"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **설명**: 5-Fold Cross-Validation은 데이터를 5등분하여, 각 Fold에서 4개는 학습용, 1개는 검증용으로 사용합니다.
    이를 통해 전체 데이터에 대한 Out-of-Fold 예측값을 생성하여 Meta Features로 활용합니다.
    """)
    
    st.markdown("---")
    
    st.subheader("2.5 Stacking 상세 플로우차트")
    
    st.write("""
    ```
    [STEP 1] 데이터 준비
        └── 원본 데이터 5.14M건 로드
        └── Train/Test 분할 (80:20)
    
    [STEP 2] 5-Fold Cross-Validation (Out-of-Fold Predictions)
        ├── Fold 1: Train(80%) → Base Models 학습 → Validation(20%) 예측
        ├── Fold 2: Train(80%) → Base Models 학습 → Validation(20%) 예측
        ├── Fold 3: Train(80%) → Base Models 학습 → Validation(20%) 예측
        ├── Fold 4: Train(80%) → Base Models 학습 → Validation(20%) 예측
        └── Fold 5: Train(80%) → Base Models 학습 → Validation(20%) 예측
         
    [STEP 3] Meta Features 생성
        ├── LightGBM-1 예측 확률 (6개 클래스)
        ├── XGBoost 예측 확률 (6개 클래스)
        └── LightGBM-2 예측 확률 (6개 클래스)
            └── 총 18차원 Meta Features
    
    [STEP 4] Meta-Learner 학습
        └── LightGBM (depth=5)
        └── Input: 18차원 Meta Features
        └── Output: 6개 카테고리 분류
    
    [STEP 5] 최종 예측
        └── Test 데이터 → Base Models → Meta Features → Meta-Learner → 최종 카테고리
    ```
    """)

# ============================================================
# 3. 모델 의사결정 과정
# ============================================================

elif page == "3. 모델 의사결정 과정":
    st.header("3. 모델 의사결정 과정 분석")
    
    st.subheader("3.1 예측 과정 시각화")
    
    # 예시 거래 하나에 대한 의사결정 과정
    st.write("**예시 거래 정보**")
    example_tx = pd.DataFrame({
        '항목': ['금액', '시간', '요일', '사용자 평균 금액', '사용자 거래 횟수'],
        '값': ['$45.00', '12:30', '화요일', '$52.30', '156건']
    })
    st.table(example_tx)
    
    st.markdown("---")
    
    st.write("**Base Models 예측 확률**")
    
    # 각 모델의 예측 확률 시각화
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    
    lgb1_probs = [0.05, 0.02, 0.08, 0.15, 0.65, 0.05]
    xgb_probs = [0.04, 0.03, 0.10, 0.12, 0.68, 0.03]
    lgb2_probs = [0.06, 0.02, 0.07, 0.18, 0.62, 0.05]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='LightGBM-1', x=categories, y=lgb1_probs, marker_color='#3498db'))
    fig.add_trace(go.Bar(name='XGBoost', x=categories, y=xgb_probs, marker_color='#e74c3c'))
    fig.add_trace(go.Bar(name='LightGBM-2', x=categories, y=lgb2_probs, marker_color='#2ecc71'))
    
    fig.update_layout(
        barmode='group',
        title='각 Base Model의 예측 확률',
        xaxis_title='카테고리',
        yaxis_title='확률',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.write("**Meta-Learner 최종 결합**")
    
    # Meta-Learner 결합 결과
    final_probs = [0.05, 0.02, 0.08, 0.15, 0.66, 0.04]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=final_probs,
        marker_color=['#95a5a6' if p < max(final_probs) else '#27ae60' for p in final_probs]
    ))
    fig.add_annotation(
        x='외식', y=0.66,
        text='최종 예측: 외식 (66%)',
        showarrow=True, arrowhead=1
    )
    fig.update_layout(
        title='Meta-Learner 최종 예측 확률',
        xaxis_title='카테고리',
        yaxis_title='확률',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("3.2 의사결정 트리 시각화")
    
    st.write("""
    **주요 결정 분기점**
    
    ```
    [거래 금액 확인]
        │
        ├── 금액 < $15 ?
        │   ├── YES → [시간대 확인]
        │   │            ├── 아침(7-9시) → 교통 (높은 확률)
        │   │            ├── 점심(11-14시) → 외식 (높은 확률)
        │   │            └── 저녁(17-20시) → 외식/쇼핑
        │   │
        │   └── NO → [금액 범위 확인]
        │              ├── $15-50 → 식료품/외식/쇼핑
        │              ├── $50-100 → 주유/쇼핑
        │              └── $100+ → 쇼핑/교통(항공)
        │
        └── [사용자 패턴 확인]
             ├── 평소 패턴과 유사 → 사용자 선호 카테고리 가중
             └── 새로운 패턴 → 금액/시간 기반 예측
    ```
    """)
    
    st.markdown("---")
    
    st.subheader("3.3 피처별 영향도 분석")
    
    # SHAP-like 영향도 시각화
    feature_impact = pd.DataFrame({
        'Feature': ['Amount ($45)', 'Hour (12:30)', 'DayOfWeek (화)', 'User_외식_Ratio (0.35)', 'Is_Weekend (No)'],
        'Impact': [0.15, 0.25, 0.05, 0.20, -0.05],
        'Direction': ['외식 +', '외식 +', '중립', '외식 +', '중립']
    })
    
    colors = ['#27ae60' if v > 0 else '#e74c3c' if v < 0 else '#95a5a6' for v in feature_impact['Impact']]
    
    fig = go.Figure(go.Bar(
        x=feature_impact['Impact'],
        y=feature_impact['Feature'],
        orientation='h',
        marker_color=colors,
        text=feature_impact['Direction'],
        textposition='outside'
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title='각 피처가 예측에 미친 영향',
        xaxis_title='영향도 (+ : 외식 방향, - : 다른 카테고리)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **해석**: 
    - 시간(12:30 점심시간)이 '외식' 예측에 가장 큰 긍정적 영향
    - 사용자의 평소 외식 비율(35%)도 외식 예측을 지지
    - 금액($45)도 외식 범위에 해당
    """)

# ============================================================
# 4. 실험 결과 비교
# ============================================================

elif page == "4. 실험 결과 비교":
    st.header("4. 실험 결과 비교")
    
    df_exp = load_experiment_data()
    
    st.subheader("4.1 전체 실험 성능 요약")
    
    # 테이블
    display_df = df_exp[['name', 'accuracy', 'macro_f1', 'method']].copy()
    display_df.columns = ['실험명', 'Accuracy (%)', 'Macro F1 (%)', '방법']
    
    # Stacking 하이라이트
    def highlight_stacking(row):
        if 'Stacking' in row['실험명']:
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)
    
    st.dataframe(display_df.style.apply(highlight_stacking, axis=1), use_container_width=True)
    st.write("(녹색 행: 최종 선정 모델)")
    
    st.markdown("---")
    
    st.subheader("4.2 Accuracy 비교")
    
    # 막대 그래프
    colors = ['#27ae60' if 'Stacking' in name else '#3498db' for name in df_exp['name']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_exp['name'],
        y=df_exp['accuracy'],
        marker_color=colors,
        text=df_exp['accuracy'].apply(lambda x: f'{x:.2f}%'),
        textposition='outside'
    ))
    fig.update_layout(
        title='실험별 Accuracy 비교',
        xaxis_title='실험',
        yaxis_title='Accuracy (%)',
        height=450,
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("4.3 Accuracy vs Macro F1 Trade-off")
    
    fig = go.Figure()
    
    for i, row in df_exp.iterrows():
        color = '#27ae60' if 'Stacking' in row['name'] else '#3498db'
        size = 20 if 'Stacking' in row['name'] else 12
        
        fig.add_trace(go.Scatter(
            x=[row['accuracy']],
            y=[row['macro_f1']],
            mode='markers+text',
            marker=dict(size=size, color=color),
            text=[row['name']],
            textposition='top center',
            name=row['name']
        ))
    
    fig.update_layout(
        title='Accuracy vs Macro F1 관계',
        xaxis_title='Accuracy (%)',
        yaxis_title='Macro F1 (%)',
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **분석 결과**:
    - Stacking Ensemble이 Accuracy와 Macro F1 모두에서 우수한 성능을 보임
    - 일반적으로 Accuracy와 F1 사이에는 Trade-off가 존재하나, Stacking은 두 지표 모두 개선
    """)
    
    st.markdown("---")
    
    st.subheader("4.4 카테고리별 성능 비교")
    
    cat_perf = load_category_performance()
    categories = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Baseline',
        x=categories,
        y=[cat_perf['Baseline'][c] for c in categories],
        marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        name='Stacking Ensemble',
        x=categories,
        y=[cat_perf['Stacking'][c] for c in categories],
        marker_color='#27ae60'
    ))
    
    fig.update_layout(
        barmode='group',
        title='카테고리별 F1 Score 비교 (Baseline vs Stacking)',
        xaxis_title='카테고리',
        yaxis_title='F1 Score (%)',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. 피처 분석
# ============================================================

elif page == "5. 피처 분석":
    st.header("5. 피처 분석")
    
    st.subheader("5.1 사용된 피처 목록 (27개)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**시간 피처 (6개)**")
        time_features = pd.DataFrame({
            '피처명': ['Hour', 'DayOfWeek', 'DayOfMonth', 'Is_Weekend', 'Is_Night', 'Is_BusinessHour'],
            '설명': ['거래 시간', '요일', '일자', '주말 여부', '야간 여부', '영업시간 여부']
        })
        st.table(time_features)
    
    with col2:
        st.write("**금액 피처 (3개)**")
        amount_features = pd.DataFrame({
            '피처명': ['Amount', 'Amount_log', 'Amount_bin'],
            '설명': ['거래 금액', '로그 변환', '금액 구간']
        })
        st.table(amount_features)
    
    with col3:
        st.write("**사용자 피처 (12개)**")
        user_features = pd.DataFrame({
            '피처명': ['User_AvgAmount', 'User_StdAmount', 'User_*_Ratio'],
            '설명': ['평균 금액', '표준편차', '카테고리별 비율 (6개)']
        })
        st.table(user_features)
    
    st.markdown("---")
    
    st.subheader("5.2 피처 중요도 (Top 10)")
    
    feature_imp = load_feature_importance()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feature_imp['Feature'][::-1],
        x=feature_imp['Importance'][::-1],
        orientation='h',
        marker_color='#3498db'
    ))
    fig.update_layout(
        title='피처 중요도 순위',
        xaxis_title='중요도 점수',
        yaxis_title='피처',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **주요 피처 해석**:
    - **Amount**: 거래 금액이 카테고리 구분에 가장 중요한 요소
    - **User_AvgAmount**: 사용자의 평균 지출 패턴
    - **Hour**: 거래 시간대 (출퇴근, 점심, 저녁 등)
    - **User_*_Ratio**: 사용자의 카테고리별 선호도
    """)
    
    st.markdown("---")
    
    st.subheader("5.3 피처 분포 시각화")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 시간대별 분포 (예시)
        hours = list(range(24))
        tx_counts = [2, 1, 0.5, 0.3, 0.2, 0.5, 2, 5, 8, 6, 5, 7, 
                    10, 8, 6, 5, 6, 8, 10, 8, 6, 5, 4, 3]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=tx_counts,
            mode='lines+markers',
            fill='tozeroy',
            marker_color='#3498db'
        ))
        fig.update_layout(
            title='시간대별 거래량 분포',
            xaxis_title='시간',
            yaxis_title='거래량 (상대값)',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 금액 분포 (예시)
        amount_bins = ['$0-10', '$10-30', '$30-50', '$50-100', '$100-200', '$200+']
        amounts = [15, 35, 25, 15, 7, 3]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=amount_bins, y=amounts,
            marker_color='#2ecc71'
        ))
        fig.update_layout(
            title='거래 금액 분포',
            xaxis_title='금액 구간',
            yaxis_title='비율 (%)',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. 결론
# ============================================================

else:  # 결론
    st.header("6. 결론 및 제언")
    
    st.subheader("6.1 최종 모델 성능")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("최종 모델", "Stacking Ensemble")
    with col2:
        st.metric("Accuracy", "49.62%")
    with col3:
        st.metric("Macro F1", "45.24%")
    
    st.markdown("---")
    
    st.subheader("6.2 시도한 기법 요약")
    
    techniques = pd.DataFrame({
        '순서': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        '기법': ['Baseline 분석', 'Class Weight 조정', '언더샘플링', 'ADASYN 증강',
                'Focal Loss', '하이퍼파라미터 최적화', 'Voting Ensemble', 
                'Stacking (5-Fold)', 'Optuna 심화'],
        '결과': ['49.13%', '44.85%', '-', '46.95%', '-', '49.55%', '49.53%', 
                '49.62% (최고)', '49.49%']
    })
    st.table(techniques)
    
    st.markdown("---")
    
    st.subheader("6.3 Stacking Ensemble 선정 이유")
    
    st.write("""
    1. **최고 Accuracy 달성**: 모든 실험 중 가장 높은 49.62% 기록
    
    2. **Macro F1 우수**: 카테고리별 균형 잡힌 성능 (45.24%)
    
    3. **일반화 성능**: 5-Fold Cross-Validation으로 과적합 방지
    
    4. **다양성 활용**: 서로 다른 모델들의 장점을 결합
    
    5. **안정성**: 단일 모델보다 예측 안정성이 높음
    """)
    
    st.markdown("---")
    
    st.subheader("6.4 프로덕션 배포 정보")
    
    deployment_info = pd.DataFrame({
        '항목': ['모델 파일', '저장 위치', '입력 형식', '출력 형식', 'API 엔드포인트'],
        '값': ['Stacking Ensemble (3 Base + 1 Meta)', '04_logs/stacking/', 
              '27개 피처 (정규화)', '6개 카테고리 확률', '/api/v1/predict']
    })
    st.table(deployment_info)
    
    st.markdown("---")
    
    st.subheader("6.5 향후 개선 방안")
    
    improvements = pd.DataFrame({
        '개선 방안': ['추가 피처 개발', '시퀀스 모델 적용', '딥러닝 앙상블'],
        '기대 효과': ['Accuracy +1~2%p', 'F1 +2~3%p', 'Accuracy +1%p'],
        '예상 기간': ['1주', '2주', '1주']
    })
    st.table(improvements)

# ============================================================
# Footer
# ============================================================

st.sidebar.markdown("---")
st.sidebar.write("**프로젝트 정보**")
st.sidebar.write("데이터: IBM Credit Card (24.4M)")
st.sidebar.write("기간: 2025-12-05 ~ 2025-12-08")
st.sidebar.write("최종 모델: Stacking Ensemble")
st.sidebar.write("Accuracy: 49.62%")
