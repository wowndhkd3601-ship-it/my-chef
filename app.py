import streamlit as st
import pandas as pd
import numpy as np
import whisper
import os
from transformers import pipeline
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="퀀트 셰프 AI", page_icon="👨‍🍳", layout="wide")

@st.cache_resource
def load_ai():
    stt_model = whisper.load_model("base")
    nlp_model = pipeline("text-classification", model="snunlp/KR-FinBert-SC")
    
    db_data = {
        '단어 수': [150, 180, 120, 100, 150, 130, 110, 150, 160, 140],
        '불확실단어_비율': [0.0, 0.16, 0.0, 0.0, 0.13, 0.0, 0.27, 0.13, 0.18, 0.14],
        'NLP_감성점수': [1, -1, 1, 1, -1, 1, 1, 0, -1, -1],
        '주제_코드': [1, 1, 1, 2, 3, 4, 1, 1, 3, 2],
        '실제_초과수익률': [13.1, -5.7, 3.7, 0.5, -8.1, 3.1, -2.7, -0.7, -27.0, -0.7]
    }
    df = pd.DataFrame(db_data)
    df['주가상승_여부'] = np.where(df['실제_초과수익률'] > 0, 1, 0)
    
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    ann.fit(df[['단어 수', '불확실단어_비율', 'NLP_감성점수', '주제_코드']], df['주가상승_여부'])
    
    return stt_model, nlp_model, ann

st.title("👨‍🍳 똑똑한 퀀트 셰프 (프로 대시보드 버전)")
st.write("CEO의 인터뷰나 연설 음성 파일(mp3, m4a 등)을 올리면 AI가 분석 근거와 함께 투자 레시피를 짜줍니다.")

with st.spinner("AI 두뇌와 귀를 세팅 중입니다..."):
    stt_model, nlp_model, ann_model = load_ai()

uploaded_file = st.file_uploader("🎙️ 분석할 오디오 파일을 올려주세요", type=['mp3', 'wav', 'm4a', 'mp4'])

if st.button("🍳 요리 시작 (음성 분석하기)", type="primary"):
    if uploaded_file is None:
        st.warning("오디오 파일을 먼저 업로드해주세요!")
    else:
        with st.spinner("AI가 음성을 듣고 대본을 작성 중입니다... (잠시만 기다려주세요)"):
            temp_path = "temp_audio_file"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # 1. 귀 (음성 인식)
                script = stt_model.transcribe(temp_path, language="ko")["text"]
                st.success("대본 추출 완료!")
                with st.expander("📝 AI가 작성한 전체 대본 보기 (클릭하여 펼치기)"):
                    st.write(script)
                
                # 2. 데이터 분석
                words = script.split()
                w_count = len(words)
                uc_count = sum(1 for w in words if any(x in w for x in ['아마', '가능성', '도전', '불확실', '어렵', '예상', '단기적', '정체', '부족']))
                uc_ratio = uc_count / w_count if w_count > 0 else 0
                
                if any(k in script for k in ['투자', '미래', '비전', 'AI']): 
                    topic = 1; topic_name = "🚀 미래/투자"
                elif any(k in script for k in ['비용', '구조조정', '효율']): 
                    topic = 2; topic_name = "✂️ 비용/구조조정"
                elif any(k in script for k in ['규제', '위기', '소송']): 
                    topic = 3; topic_name = "⚠️ 위험/리스크"
                else: 
                    topic = 4; topic_name = "🏢 일반 경영"
                
                nlp_res = nlp_model(script[:500])[0]
                if nlp_res['label'] == 'positive':
                    nlp_score = 1; nlp_name = "긍정적 (호재)"
                elif nlp_res['label'] == 'negative':
                    nlp_score = -1; nlp_name = "부정적 (악재)"
                else:
                    nlp_score = 0; nlp_name = "중립적"
                
                # 3. 뇌 (최종 예측)
                pred = ann_model.predict([[w_count, uc_ratio, nlp_score, topic]])[0]
                prob = ann_model.predict_proba([[w_count, uc_ratio, nlp_score, topic]])[0]
                conf = prob[1] * 100 if pred == 1 else prob[0] * 100
                
                # 🌟 화면 출력 디자인 (대시보드)
                st.markdown("---")
                st.subheader("📊 AI 셰프의 분석 리포트 (판단 근거)")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("📌 감지된 핵심 주제", topic_name)
                col_b.metric("📈 금융 AI 감성 평가", nlp_name)
                col_c.metric("🤔 불확실성 어휘 비율", f"{uc_ratio*100:.1f}%")
                
                st.markdown("---")
                st.subheader("👨‍🍳 최종 투자 레시피")
                
                col1, col2 = st.columns(2)
                if pred == 1:
                    col1.metric("셰프의 예측", "🟢 강력 매수 (상승)")
                    st.info("💡 **셰프의 코멘트:** 감성 평가가 긍정적이고 불확실성 어휘가 적어, 과거 주가 상승 패턴과 매우 일치합니다.")
                else:
                    col1.metric("셰프의 예측", "🔴 매도 (하락/관망)")
                    st.warning("💡 **셰프의 코멘트:** 모호한 발언이나 부정적인 뉘앙스가 감지되었습니다. 리스크 관리가 필요합니다.")
                col2.metric("예측 확신도", f"{conf:.1f}%")

            except Exception as e:
                st.error(f"오디오 분석 중 오류가 발생했습니다: {e}")
