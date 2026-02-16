import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Spam Classifier", layout="centered")

# -------------------- STYLE --------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 600;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.result {
    padding: 18px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: 600;
    text-align: center;
    margin-top: 20px;
}
.spam { background-color: #fff1f0; color: #a8071a; border:1px solid #ffa39e;}
.ham { background-color: #f6ffed; color: #135200; border:1px solid #b7eb8f;}
.footer {
    text-align:center;
    color:gray;
    font-size:13px;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">Spam Message Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning based text classification</div>', unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# -------------------- PREPROCESS --------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -------------------- INPUT --------------------
input_sms = st.text_area("Enter message", height=150)

predict = st.button("Analyze")

# -------------------- PREDICTION --------------------
if predict and input_sms.strip() != "":

    with st.spinner("Processing..."):
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]
        confidence = round(max(prob)*100,2)

    if result == 1:
        st.markdown(f'<div class="result spam">Spam message<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)
        st.progress(int(confidence))
    else:
        st.markdown(f'<div class="result ham">Not spam<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)
        st.progress(int(confidence))

