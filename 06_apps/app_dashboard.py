"""
다음 소비 카테고리 예측 대시보드
개인 소비 데이터 (CSV) 업로드 → 다음 구매 카테고리 예측
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# 페이지 설정
st.set_page_config(
    page_title="💳 다음 소비 예측기",
    page_icon="💳",
    layout="wide"
)

# 카테고리 매핑 (개인 데이터 → 모델 카테고리)
CATEGORY_MAPPING = {
    '식비': '외식',
    '카페/간식': '외식',
    '술/유흥': '외식',
    '온라인쇼핑': '쇼핑',
    '패션/쇼핑': '쇼핑',
    '뷰티/미용': '쇼핑',
    '생활': '생활',
    '생필품': '생활',
    '의료/건강': '생활',
    '교통': '교통',
    '자동차': '주유',
    '주유': '주유',
    # 기타는 가장 가까운 카테고리로 매핑
    '이체': '생활',
    '금융': '생활',
    '투자': '생활',
    '문화/여가': '쇼핑',
    '여행/숙박': '쇼핑',
}

# 모델 카테고리 (학습 시 사용)
MODEL_CATEGORIES = ['교통', '생활', '쇼핑', '식료품', '외식', '주유']


@st.cache_resource
def load_model():
    """학습된 모델 로드"""
    model_dir = '03_models/06_sequence'
    
    # 가장 최근 모델 찾기
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(model_dir, latest_model)
            model = joblib.load(model_path)
            return model, latest_model
    
    return None, None


def parse_personal_data(df):
    """개인 데이터 전처리"""
    # 날짜/시간 파싱
    df['datetime'] = pd.to_datetime(df['날짜'] + ' ' + df['시간'])
    
    # 금액 처리 (문자열 → 숫자)
    df['금액'] = df['금액'].astype(float)
    
    # 카테고리 매핑
    df['mapped_category'] = df['대분류'].map(CATEGORY_MAPPING).fillna('생활')
    
    # 시간순 정렬
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df


def create_features(df):
    """모델 입력 특성 생성"""
    features_list = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # 기본 특성
        features = {
            'Amount': abs(row['금액']),
            'Amount_log': np.log1p(abs(row['금액'])),
            'Hour': row['datetime'].hour,
            'DayOfWeek': row['datetime'].dayofweek,
            'DayOfMonth': row['datetime'].day,
            'IsWeekend': 1 if row['datetime'].dayofweek >= 5 else 0,
            'IsNight': 1 if row['datetime'].hour >= 22 or row['datetime'].hour <= 6 else 0,
            'IsBusinessHour': 1 if 9 <= row['datetime'].hour <= 18 else 0,
        }
        
        # 사용자 통계 (전체 데이터 기반)
        features['User_AvgAmount'] = df['금액'].abs().mean()
        features['User_StdAmount'] = df['금액'].abs().std()
        
        # 시퀀스 특성
        if i > 0:
            time_diff = (row['datetime'] - df.iloc[i-1]['datetime']).total_seconds()
            features['Time_Since_Last'] = time_diff
            features['Previous_Category_encoded'] = MODEL_CATEGORIES.index(df.iloc[i-1]['mapped_category'])
        else:
            features['Time_Since_Last'] = 0
            features['Previous_Category_encoded'] = -1
        
        features['Transaction_Sequence'] = i + 1
        
        # 현재 카테고리 인코딩
        features['Current_Category_encoded'] = MODEL_CATEGORIES.index(row['mapped_category'])
        
        # 카테고리별 빈도
        features['User_Category_Count'] = (df.iloc[:i+1]['mapped_category'] == row['mapped_category']).sum()
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def main():
    st.title("💳 다음 소비 카테고리 예측기")
    st.markdown("개인 소비 데이터를 업로드하면 다음 구매 카테고리를 예측합니다.")
    
    # 사이드바
    with st.sidebar:
        st.header("📊 모델 정보")
        
        model, model_name = load_model()
        
        if model:
            st.success(f"✅ 모델 로드 완료")
            st.caption(f"모델: {model_name}")
        else:
            st.warning("⚠️ 학습된 모델이 없습니다.")
            st.info("먼저 `./run_sequence_pipeline.sh`를 실행하세요.")
        
        st.markdown("---")
        st.header("📁 파일 형식")
        st.markdown("""
        **필수 컬럼**:
        - `날짜`: YYYY-MM-DD
        - `시간`: HH:MM
        - `대분류`: 카테고리
        - `금액`: 숫자 (음수 가능)
        
        **인코딩**: EUC-KR 또는 UTF-8
        """)
    
    # 메인 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("1️⃣ 데이터 업로드")
        
        uploaded_file = st.file_uploader(
            "CSV 파일 선택",
            type=['csv'],
            help="개인 소비 내역 CSV 파일을 업로드하세요."
        )
        
        if uploaded_file:
            try:
                # EUC-KR 시도
                try:
                    df = pd.read_csv(uploaded_file, encoding='euc-kr')
                except:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                
                st.success(f"✅ {len(df)}건의 거래 데이터 로드 완료")
                
                # 데이터 전처리
                df = parse_personal_data(df)
                
                # 최근 5건 표시
                st.subheader("📋 최근 거래 내역")
                display_df = df[['datetime', '대분류', '금액', 'mapped_category']].tail(10).copy()
                display_df.columns = ['날짜/시간', '원본 분류', '금액', '매핑된 카테고리']
                st.dataframe(display_df, use_container_width=True)
                
                # 특성 생성
                features_df = create_features(df)
                
                # 예측
                if model and st.button("🎯 다음 구매 예측하기", type="primary"):
                    with st.spinner("예측 중..."):
                        # 마지막 거래 특성
                        last_features = features_df.iloc[-1:].values
                        
                        # 예측
                        prediction = model.predict(last_features)[0]
                        predicted_category = MODEL_CATEGORIES[prediction]
                        
                        # 확률 (XGBoost의 경우)
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(last_features)[0]
                        else:
                            probabilities = np.zeros(len(MODEL_CATEGORIES))
                            probabilities[prediction] = 1.0
                        
                        # 결과 저장
                        st.session_state['prediction'] = predicted_category
                        st.session_state['probabilities'] = probabilities
                
            except Exception as e:
                st.error(f"❌ 파일 처리 중 오류: {e}")
    
    with col2:
        st.header("2️⃣ 예측 결과")
        
        if 'prediction' in st.session_state:
            predicted = st.session_state['prediction']
            probs = st.session_state['probabilities']
            
            # 예측 카테고리 강조
            st.markdown(f"""
            <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; text-align: center;'>
                <h2 style='color: white; margin: 0;'>다음 구매 예상 카테고리</h2>
                <h1 style='color: #ffd700; margin: 10px 0; font-size: 3em;'>{predicted}</h1>
                <p style='color: white; font-size: 1.2em;'>확률: {probs[MODEL_CATEGORIES.index(predicted)]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 확률 분포 차트
            st.subheader("📊 카테고리별 예측 확률")
            
            prob_df = pd.DataFrame({
                '카테고리': MODEL_CATEGORIES,
                '확률': probs * 100
            }).sort_values('확률', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=prob_df['확률'],
                y=prob_df['카테고리'],
                orientation='h',
                marker=dict(
                    color=prob_df['확률'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=prob_df['확률'].round(1),
                texttemplate='%{text}%',
                textposition='outside'
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="확률 (%)",
                yaxis_title="",
                showlegend=False,
                margin=dict(l=0, r=50, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 정보
            st.subheader("📈 데이터 통계")
            col_a, col_b, col_c = st.columns(3)
            
            if uploaded_file:
                with col_a:
                    st.metric("총 거래", f"{len(df):,}건")
                with col_b:
                    st.metric("평균 금액", f"{abs(df['금액'].mean()):,.0f}원")
                with col_c:
                    recent_category = df.iloc[-1]['mapped_category']
                    st.metric("마지막 거래", recent_category)
        
        else:
            st.info("👈 왼쪽에서 파일을 업로드하고 예측 버튼을 눌러주세요.")
    
    # 하단 정보
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>💡 <b>팁</b>: 더 정확한 예측을 위해서는 최소 50개 이상의 거래 데이터가 필요합니다.</p>
        <p>🔒 데이터는 서버에 저장되지 않으며, 브라우저에서만 처리됩니다.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
