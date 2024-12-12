# News Article Sentiment Analysis Web - TEAM68

This project is a web application that accepts the title or content of a news article and analyzes its sentiment (positive, negative). It is implemented using HuggingFace's pre-trained models for simplicity and accuracy.

![Python](https://img.shields.io/badge/Language-Python-blue)


## Features
- Sentiment analysis for news articles based on their title or content.
- Simple and interactive web interface built using Streamlit.
- Visual representation of sentiment analysis results.


### Dependencies
- **Programming Language**: Python 3.7+
- **Environment Manager**: Anaconda
- **Framework**: Streamlit (for the web interface)
- **Libraries**:
  - transformers: For the HuggingFace sentiment analysis model.
  - nltk: For text preprocessing (stopwords and lemmatization).
  - matplotlib: For visualizing sentiment scores.
  - torch: For GPU-accelerated model inference (optional).
- **Model**:
  - distilbert-base-uncased-finetuned-sst-2-english: A lightweight and fine-tuned BERT model for sentiment analysis.


## Demo
Below is a demo screenshot of analyzing sample news articles:
![Demo Screenshot](https://github.com/smnn99/TEAM68/tree/main/demo_images)


Here are some of the URLs of the analyzed news articles:
- [Spotoday News](https://www.spotoday.kr/news/articleView.html?idxno=18655)
- [BBC Korean - Article 1](https://www.bbc.com/korean/articles/cq5kp9ze7lxo)
- [BBC Korean - Article 2](https://www.bbc.com/korean/articles/cvgejke5z5lo)
- [Yonhap News TV](https://www.yonhapnewstv.co.kr/news/MYH20241206016500640?input=1825m)




## Prerequisites
- **Rust**: Required for installing the `tokenizers` library.
  - Install Rust from [https://rustup.rs/](https://rustup.rs/)
 
    
### Future Plans
- Add multilingual support (e.g., Korean, Japanese).
- Improve model inference speed for larger inputs.
- Deploy the application to cloud platforms (e.g., AWS, Heroku).


### Contributors
- [박성민](https://github.com/smnn99)
- [김서연](https://github.com/seoyeon145)
- [김도현](https://github.com/KDH122)
- [송지민](https://github.com/jimin123456)
