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
                script = stt_model.transcribe(temp_path, language="ko")["
