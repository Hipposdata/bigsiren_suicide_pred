import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from streamlit_shap import st_shap
import plotly.graph_objects as go
import bz2

# =========================
# 매핑 딕셔너리 정의 (코드 ↔ 한글)
# =========================
STRESS_MAP = {0: '전혀 없음', 1: '약간', 2: '보통', 3: '많음'}
HEALTH_MAP = {0: '나쁨', 1: '보통', 2: '좋음'}
SLEEP_MAP = {0: '아니오', 1: '예'}
ACHIEVE_MAP = {0: '매우 불만족', 1: '불만족', 2: '보통', 3: '만족', 4: '매우 만족', 5: '최고 만족'}
DAILY_STRESS_MAP = STRESS_MAP
CANCER_FEAR_MAP = {0: '전혀 없음', 1: '약간', 2: '보통', 3: '많음', 4: '매우 많음'}
REGION_MAP = {i: v for i, v in enumerate(['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
                                         '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'])}
MARRIAGE_MAP = {0: '전혀 그렇지 않다', 1: '그렇지 않다', 2: '그렇다', 3: '매우 그렇다'}
FAMILY_MAP = ACHIEVE_MAP
DISEASE_MAP = SLEEP_MAP

# =========================
# Streamlit 설정 및 스타일
# =========================
st.set_page_config(page_title="빅 사이렌", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main {background-color: #f8f9fa; font-family: 'Malgun Gothic', sans-serif;}
    h1 {color: #2E7D32; font-weight: 700; padding-bottom: 20px; border-bottom: 2px solid #2E7D32; margin-bottom: 30px;}
    h2, h3 {color: #1B5E20; margin-top: 30px; margin-bottom: 15px;}
    .stCard {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;}
    .prediction-low-risk {background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 18px;}
    .prediction-high-risk {background-color: #F44336; color: white; padding: 15px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 18px;}
    .probability-box {background-color: #E3F2FD; border: 2px solid #2196F3; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0;}
    .probability-value {font-size: 32px; font-weight: bold; color: #1976D2; margin: 10px 0;}
    .warning-box {background-color: #FFF3E0; border-left: 4px solid #FF9800; padding: 15px; margin: 10px 0; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.facecolor'] = '#f9f9f9'

# =========================
# 헤더 및 안내
# =========================
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/?size=100&id=72xTAy8tXrTD&format=png&color=000000", width=80)
with col2:
    st.title(':green[빅 사이렌 (자살예측 모니터링 시스템)]')
st.markdown("""
<div class="stCard">
    <p>이 시스템은 다양한 심리적, 사회적 요인을 기반으로 자살 위험도를 예측합니다. 
    입력된 정보를 바탕으로 위험도를 분석하고, SHAP 값을 통한 예측 근거를 제공합니다.</p>
    <div class="warning-box">
        <strong>⚠️ 중요 안내:</strong> 이 시스템은 참고용으로만 사용하시고, 실제 위험 상황에서는 반드시 전문가의 도움을 받으시기 바랍니다.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# 모델 및 데이터 로딩 (캐싱)
# =========================
@st.cache_resource

def load_model():
    try:
        with bz2.BZ2File("./optuna_model/lgbm.pbz2", 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.error("모델 파일을 찾을 수 없습니다. ./lgbm.pkl.pbz2 파일을 확인해주세요.")
        return None

model = load_model() 
@st.cache_data
def load_full_train_data():
    try:
        return pd.read_csv('./test_web.csv')
    except:
        st.warning("학습 데이터를 찾을 수 없습니다.")
        return None

model = load_model()
train_data = load_full_train_data()

# =========================
# 입력 폼 함수화 (반복 최소화)
# =========================
def user_input_features():
    questions = [
        {'label': '가정생활에서 느끼는 스트레스 정도는 어떻습니까?', 'options': list(STRESS_MAP.values()), 'map': STRESS_MAP},
        {'label': '본인의 전반적인 건강상태를 어떻게 평가하십니까?', 'options': list(HEALTH_MAP.values()), 'map': HEALTH_MAP},
        {'label': '평소 적정한 수면시간(7-8시간)을 유지하고 계십니까?', 'options': list(SLEEP_MAP.values()), 'map': SLEEP_MAP},
        {'label': '현재 본인의 성취에 대한 만족도는 어느 정도입니까?', 'options': list(ACHIEVE_MAP.values()), 'map': ACHIEVE_MAP},
        {'label': '전반적인 일상생활에서 느끼는 스트레스 정도는 어떻습니까?', 'options': list(DAILY_STRESS_MAP.values()), 'map': DAILY_STRESS_MAP},
        {'label': '암에 대한 두려움 정도는 어떻습니까?', 'options': list(CANCER_FEAR_MAP.values()), 'map': CANCER_FEAR_MAP},
        {'label': '거주하고 계신 시도는 어디입니까?', 'options': list(REGION_MAP.values()), 'map': REGION_MAP},
        {'label': '결혼과 출산이 필수라고 생각하십니까?', 'options': list(MARRIAGE_MAP.values()), 'map': MARRIAGE_MAP},
        {'label': '전반적인 가족관계에 대한 만족도는 어떻습니까?', 'options': list(FAMILY_MAP.values()), 'map': FAMILY_MAP},
        {'label': '현재 만성질환을 앓고 계십니까?', 'options': list(DISEASE_MAP.values()), 'map': DISEASE_MAP}
    ]
    codes = []
    display = {}
    for q, code_name in zip(questions, [
        '스트레스정도_가정생활코드', '분류코드_건강평가코드', '건강관리_적정수면여부', '성취만족도코드',
        '스트레스정도_전반적일상생활코드', '암에대한두려움정도코드', '행정구역시도코드', '결혼문화_결혼출산필수코드',
        '가족관계만족도_전반적가족코드', '유병기간_유병여부'
    ]):
        val = st.sidebar.selectbox(q['label'], q['options'])
        code = q['options'].index(val)
        codes.append(code)
        display[q['label'].split('는')[0].strip()] = val
    data = dict(zip(
        ['스트레스정도_가정생활코드', '분류코드_건강평가코드', '건강관리_적정수면여부', '성취만족도코드',
         '스트레스정도_전반적일상생활코드', '암에대한두려움정도코드', '행정구역시도코드', '결혼문화_결혼출산필수코드',
         '가족관계만족도_전반적가족코드', '유병기간_유병여부'],
        codes
    ))
    return pd.DataFrame([data]), display

# =========================
# 사이드바: 입력 방식
# =========================
st.sidebar.image("https://img.icons8.com/color/96/000000/mental-health.png", width=50)
st.sidebar.title('정보 입력 방식 선택')
st.sidebar.markdown("---")
input_method = st.sidebar.radio("데이터 입력 방식", ["직접 입력", "기존 데이터 조회"])

if input_method == "직접 입력":
    st.sidebar.subheader('개인 정보 입력')
    input_df, display_data = user_input_features()
    if st.sidebar.button('입력값 초기화'):
        st.rerun()
else:
    st.sidebar.subheader('기존 데이터 조회')
    if train_data is not None:
        id_col = 'id' if 'id' in train_data.columns else None
        available_ids = sorted(train_data[id_col].dropna().astype(int).unique()) if id_col else list(range(len(train_data)))
        if available_ids:
            selected_id = st.sidebar.selectbox("데이터 ID 선택", available_ids)
            selected_data = train_data[train_data[id_col] == selected_id].copy() if id_col else train_data.iloc[[selected_id]].copy()
            input_columns = [
                '스트레스정도_가정생활코드', '분류코드_건강평가코드', '건강관리_적정수면여부',
                '성취만족도코드', '스트레스정도_전반적일상생활코드', '암에대한두려움정도코드',
                '행정구역시도코드', '결혼문화_결혼출산필수코드', '가족관계만족도_전반적가족코드',
                '유병기간_유병여부'
            ]
            existing_columns = [col for col in input_columns if col in selected_data.columns]
            input_df = selected_data[existing_columns].copy()
            display_data = {
                '가정생활 스트레스': STRESS_MAP.get(selected_data['스트레스정도_가정생활코드'].values[0], '알 수 없음'),
                '건강상태 평가': HEALTH_MAP.get(selected_data['분류코드_건강평가코드'].values[0], '알 수 없음'),
                '적정수면 여부': SLEEP_MAP.get(selected_data['건강관리_적정수면여부'].values[0], '알 수 없음'),
                '성취만족도': ACHIEVE_MAP.get(selected_data['성취만족도코드'].values[0], '알 수 없음'),
                '일상생활 스트레스': DAILY_STRESS_MAP.get(selected_data['스트레스정도_전반적일상생활코드'].values[0], '알 수 없음'),
                '암에 대한 두려움': CANCER_FEAR_MAP.get(selected_data['암에대한두려움정도코드'].values[0], '알 수 없음'),
                '거주 지역': REGION_MAP.get(selected_data['행정구역시도코드'].values[0], '알 수 없음'),
                '결혼출산 필수성': MARRIAGE_MAP.get(selected_data['결혼문화_결혼출산필수코드'].values[0], '알 수 없음'),
                '가족관계 만족도': FAMILY_MAP.get(selected_data['가족관계만족도_전반적가족코드'].values[0], '알 수 없음'),
                '만성질환 여부': DISEASE_MAP.get(selected_data['유병기간_유병여부'].values[0], '알 수 없음')
            }
            st.sidebar.markdown("---")
            st.sidebar.subheader("선택한 ID의 전체 데이터")
            st.sidebar.dataframe(pd.DataFrame([display_data]).T.rename(columns={0: '값'}))
        else:
            st.sidebar.error("사용 가능한 ID가 없습니다.")
            input_df = pd.DataFrame()
            display_data = {}
    else:
        st.sidebar.error("데이터를 불러올 수 없습니다.")
        input_df = pd.DataFrame()
        display_data = {}

st.sidebar.markdown("---")
st.sidebar.info("© 2025 자살예측 시스템 v1.0")

# =========================
# 메인: 입력값 및 예측/SHAP
# =========================
if not input_df.empty:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader('📋 입력된 정보')
    col1, col2, col3 = st.columns(3)
    for i, (key, value) in enumerate(display_data.items()):
        [col1, col2, col3][i % 3].metric(label=key, value=value)
    st.markdown("**전체 입력 데이터:**")
    st.dataframe(input_df.style.background_gradient(cmap='RdYlBu_r'))
    st.markdown('</div>', unsafe_allow_html=True)

    if model is not None:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader('🎯 자살 위험도 예측 결과')
        try:
            prediction_proba = model.predict_proba(input_df)
            prob_class_1 = prediction_proba[0][1]
            threshold = 0.1
            prediction_class_1 = 1 if prob_class_1 >= threshold else 0
            st.markdown('<div class="probability-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 자살 위험도 예측 확률")
            st.markdown(f'<div class="probability-value">{prob_class_1:.4f} ({prob_class_1:.1%})</div>', unsafe_allow_html=True)
            st.markdown("**클래스 1 (위험도 높음)으로 예측할 확률**")
            st.markdown('</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("클래스 1 확률 (높은 위험도)", f"{prob_class_1:.4f}", f"{prob_class_1:.1%}")
            col2.metric("임계값", f"{threshold}", "기준점")
            col3.metric("최종 예측", "높은 위험도" if prediction_class_1 == 1 else "낮은 위험도", f"임계값: {threshold}")
            col1, col2 = st.columns(2)
            with col1:
                if prediction_class_1 == 1:
                    st.markdown('<div class="prediction-high-risk">⚠️ 높은 위험도 감지</div>', unsafe_allow_html=True)
                    st.markdown(f"""<div class="warning-box">
                        <strong>예측 확률: {prob_class_1:.4f} ({prob_class_1:.1%})</strong><br>
                        <strong>임계값 {threshold} 이상으로 위험도가 감지되었습니다.</strong><br>
                        <strong>즉시 전문가 상담을 권장합니다:</strong><br>
                        • 청소년상담복지개발원: 1388<br>
                        • 생명의전화: 1588-9191<br>
                        • 정신건강위기상담전화: 1577-0199
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-low-risk">✅ 낮은 위험도</div>', unsafe_allow_html=True)
                    st.info(f"예측 확률: {prob_class_1:.4f} ({prob_class_1:.1%}) - 임계값 {threshold} 미만으로 현재 상태는 양호하지만, 지속적인 관심과 케어가 필요합니다.")
            with col2:
                prob_df = pd.DataFrame({
                    '항목': ['클래스 1 확률', '임계값', '예측 결과'],
                    '값': [f"{prob_class_1:.4f}", f"{threshold}", "위험도 높음" if prediction_class_1 == 1 else "위험도 낮음"],
                    '백분율/상태': [f"{prob_class_1:.1%}", "기준", "✅" if prediction_class_1 == 0 else "⚠️"]
                })
                st.dataframe(prob_df, use_container_width=True)
            st.markdown("---")
            st.markdown("### 📊 확률값 해석")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""**확률값 세부 정보:**
                - 클래스 1 확률: `{prob_class_1:.6f}`
                - 임계값: `{threshold}`
                - 확률 >= 임계값: `{prob_class_1 >= threshold}`""")
            with col2:
                st.markdown(f"""**예측 기준:**
                - 임계값: {threshold} (낮은 임계값으로 민감하게 감지)
                - 예측 클래스: {prediction_class_1}
                - 신뢰도: {prob_class_1:.1%}""")
            st.markdown("#### 📊 예측 확률 시각화")
            col1, col2 = st.columns(2)
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_class_1 * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"자살 위험도 ({prob_class_1:.4f})"},
                    delta={'reference': threshold * 100},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prob_class_1 >= threshold else "darkgreen"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgreen"},
                            {'range': [5, 10], 'color': "yellow"},
                            {'range': [10, 25], 'color': "orange"},
                            {'range': [25, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col2:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=['클래스 1 확률'],
                    y=[prob_class_1],
                    name='예측 확률',
                    marker_color='red' if prob_class_1 >= threshold else 'green',
                    text=[f'{prob_class_1:.4f}'],
                    textposition='auto'
                ))
                fig_bar.add_hline(
                    y=threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"임계값: {threshold}"
                )
                fig_bar.update_layout(
                    title="확률값과 임계값 비교",
                    yaxis_title="확률",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # ======= SHAP 분석 =======
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader('🔍 예측 근거 분석 (SHAP)')
        st.markdown("각 요인이 예측 결과에 미친 영향을 분석합니다.")
        try:
            with st.spinner('SHAP 분석 중입니다...'):
                explainer = shap.TreeExplainer(model)
                
                # SHAP 값 계산 (LightGBM 이진 분류 최신 버전 대응)
                shap_values = explainer.shap_values(input_df)
                
                # SHAP 값 구조 확인 및 처리
                if isinstance(shap_values, list):
                    # LightGBM 이진 분류: shap_values[0]에 양성 클래스(1)에 대한 SHAP 값이 포함됨
                    shap_vals = shap_values[0][0]  # 첫 번째 샘플의 SHAP 값 추출
                else:
                    shap_vals = shap_values[0]  # 단일 배열인 경우
                
                feature_names = [
                    '가정생활 스트레스', '건강상태 평가', '적정수면 여부', '성취만족도',
                    '일상생활 스트레스', '암에 대한 두려움', '거주 지역', '결혼출산 필수성',
                    '가족관계 만족도', '만성질환 여부'
                ]

                tab1, tab2, tab3 = st.tabs(["📊 SHAP Force Plot", "📈 Feature Impact", "📋 Summary"])
                
                # Force Plot 수정
                with tab1:
                    st.markdown("#### SHAP Force Plot")
                    try:
                        if isinstance(shap_values, list):
                            st_shap(shap.force_plot(
                                explainer.expected_value[1],  # 양성 클래스(1)의 기대값 사용
                                shap_values[1][0],            # 양성 클래스 SHAP 값
                                input_df.iloc[0],
                                feature_names=feature_names,
                                matplotlib=False
                            ), height=200)
                        else:
                            st_shap(shap.force_plot(
                                explainer.expected_value,
                                shap_values[0],
                                input_df.iloc[0],
                                feature_names=feature_names,
                                matplotlib=False
                            ), height=200)
                    except Exception as e:
                        st.warning(f"Force plot 표시 중 오류: {e}")

                # Feature Impact Plot 수정
                with tab2:
                    st.markdown("#### 각 요인별 영향도")
                    fig_bar = go.Figure()
                    colors = ['red' if x > 0 else 'blue' for x in shap_vals]
                    fig_bar.add_trace(go.Bar(
                        x=shap_vals,
                        y=feature_names,
                        orientation='h',
                        marker_color=colors,
                        text=[f'{val:.3f}' for val in shap_vals],
                        textposition='auto',
                    ))
                    fig_bar.update_layout(
                        title="SHAP 값 - 각 요인별 위험도 기여도",
                        xaxis_title="SHAP 값 (위험도에 미치는 영향)",
                        yaxis_title="특성",
                        height=500,
                        showlegend=False
                    )
                    fig_bar.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                # Summary Plot 수정
                with tab3:
                    st.markdown("#### SHAP 값 요약")
                    shap_df = pd.DataFrame({
                        '특성': feature_names,
                        'SHAP 값': shap_vals,
                        '절댓값': np.abs(shap_vals),
                        '영향': ['위험도 증가' if x > 0 else '위험도 감소' for x in shap_vals],
                        '중요도 순위': np.argsort(np.abs(shap_vals))[::-1] + 1
                    })
                    # ... [이하 동일] ...
                    
        except Exception as e:
            st.error(f"SHAP 분석 중 오류: {e}")


# =========================
# 도움말 및 푸터
# =========================
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.subheader('📞 도움이 필요하시다면')
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **청소년상담복지개발원**
    - 전화: 1388
    - 24시간 운영
    - 청소년 전문 상담
    """)
with col2:
    st.markdown("""
    **생명의전화**
    - 전화: 1588-9191
    - 24시간 운영
    - 자살예방 전문상담
    """)
with col3:
    st.markdown("""
    **정신건강위기상담**
    - 전화: 1577-0199
    - 24시간 운영
    - 정신건강 응급상담
    """)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #666;">
    <p>© 2025 자살예측 모니터링 시스템 | 버전 1.0</p>
    <p><small>이 시스템은 연구 및 참고 목적으로만 사용하시고, 실제 상황에서는 반드시 전문가의 도움을 받으시기 바랍니다.</small></p>
</div>
""", unsafe_allow_html=True)

