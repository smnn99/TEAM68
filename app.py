import streamlit as st
from transformers import pipeline
import re

# Streamlit 페이지 설정
st.set_page_config(page_title="뉴스 기사 분위기 파악하기", layout="wide")


# 감성 분석 모델 로드
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


model = load_model()

# 앱 인터페이스
st.title("뉴스 기사 분위기 파악 애플리케이션")
st.markdown("뉴스 기사의 제목이나 내용을 입력하고, 분위기를 살펴보세요.")

# 사용자 입력
user_input = st.text_area("뉴스 제목 또는 내용을 입력하세요:", height=200)


# 텍스트 전처리 함수: 특수 문자 제거 및 소문자화
def preprocess_text(text):
    # 텍스트에서 특수 문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 텍스트를 소문자로 변환
    text = text.lower()
    return text


# 부정적인 키워드 처리
def post_process(sentiment, text):
    # 부정적인 키워드 리스트
    negative_keywords = [
        "연루", "혐의", "사망", "추락", "부패", "폭력", "사건", "위기", "적발", "탈세",
        "allegation", "suspension", "accusation", "scandal", "crisis", "investigation",
        "collapse", "controversy", "failure", "depletion", "deterioration"
    ]

    # 텍스트에 부정적인 키워드가 포함되면 강제로 부정적 감성으로 처리
    if any(keyword in text.lower() for keyword in negative_keywords):
        sentiment = "NEGATIVE"

    return sentiment

if st.button("분석하기"):
    if user_input.strip():
        with st.spinner("분석 중..."):
            # 텍스트 전처리
            preprocessed_text = preprocess_text(user_input)

            # 감성 분석 수행
            result = model(preprocessed_text)

        st.success("분석 완료!")
        sentiment = result[0]['label']
        score = result[0]['score']

        # 후처리: 부정적인 키워드가 포함된 경우 부정적 결과로 수정
        sentiment = post_process(sentiment, user_input)

        # 결과 출력
        st.subheader("분석 결과:")
        st.write(f"**분위기:** {sentiment}")
        st.write(f"**확신 점수:** {score:.2f}")
    else:
        st.error("입력값이 비어 있습니다. 텍스트를 입력해주세요!")
