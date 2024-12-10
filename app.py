import streamlit as st
from transformers import pipeline
import re
import matplotlib.pyplot as plt
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import torch

# NLTK 리소스 다운로드 (최초 실행 시)
nltk.download('stopwords')
nltk.download('wordnet')

# 로깅 설정
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Streamlit 페이지 설정
st.set_page_config(page_title="뉴스 기사 감성 분석", layout="wide")

# 감성 분석 모델 로드
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

model = load_model()

# 앱 인터페이스
st.title("뉴스 기사 감성 분석 애플리케이션")
st.markdown("뉴스 기사의 제목이나 내용을 입력하고, 분위기를 분석하세요.")

# 사용자 입력
user_input = st.text_area("뉴스 제목 또는 내용을 입력하세요:", height=200)

# 텍스트 전처리 함수: 특수 문자 제거, 소문자화, 불용어 제거, 표제어 추출
def preprocess_text(text):
    # 특수 문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 소문자화
    text = text.lower()
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# 부정적인 키워드 처리
def post_process(sentiment, text):
    # 부정적인 키워드 리스트
    negative_keywords = [
        "allegation", "suspension", "accusation", "scandal", "crisis", "investigation",
        "collapse", "controversy", "failure", "depletion", "deterioration",
        "lawsuit", "fraud", "bankruptcy", "protest", "strike", "retirement"
    ]

    # 텍스트에 부정적인 키워드가 포함되면 강제로 부정적 감성으로 처리
    if any(keyword in text.lower() for keyword in negative_keywords):
        sentiment = "NEGATIVE"

    return sentiment

# 감성 분포 시각화 함수
def visualize_sentiment(sentiment, score):
    labels = ['Positive', 'Negative', 'Neutral']
    colors = ['green', 'red', 'grey']
    values = [0, 0, 0]

    if sentiment.upper() == "POSITIVE":
        values[0] = score
    elif sentiment.upper() == "NEGATIVE":
        values[1] = score
    elif sentiment.upper() == "NEUTRAL":
        values[2] = score

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=colors)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Confidence Score')
    st.pyplot(fig)

if st.button("분석하기"):
    if user_input.strip():
        try:
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

            # 중립 감성 추가
            if score < 0.6:
                sentiment = "NEUTRAL"

            # 결과 출력
            st.subheader("분석 결과:")
            st.write(f"**감정:** {sentiment}")
            st.write(f"**확신 점수:** {score:.2f}")

            # 시각화
            visualize_sentiment(sentiment, score)

        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")
            st.error("분석 중 오류가 발생했습니다. 나중에 다시 시도해주세요.")
    else:
        st.error("입력값이 비어 있습니다. 텍스트를 입력해주세요!")
