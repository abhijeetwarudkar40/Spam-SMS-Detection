# Spam SMS Detection using Machine Learning

A web application that classifies SMS or Email messages as **Spam** or **Not Spam** using Natural Language Processing and Machine Learning.

The system preprocesses text, converts it into numerical features using TF-IDF, and predicts the category using a trained classifier.
The model is deployed through a Streamlit web interface.

---

## Features

* Instant spam detection
* NLP based text preprocessing
* TF-IDF feature extraction
* Machine learning classification
* Confidence score display
* Clean Streamlit interface

---

## Tech Stack

**Language**

* Python

**Libraries**

* scikit-learn
* pandas
* numpy
* nltk
* streamlit
* pickle

---

## Machine Learning Workflow

1. Data Cleaning
2. Text Preprocessing

   * Lowercasing
   * Tokenization
   * Stopword Removal
   * Stemming
3. Feature Extraction (TF-IDF Vectorization)
4. Model Training (Naive Bayes Classifier)
5. Model Serialization
6. Web App Deployment

---

## Project Structure

```
Spam-SMS-Detection/
│
├── app.py
├── vectorizer.pkl
├── model.pkl
├── requirements.txt
├── README.md
└── notebook/
    └── SMS_Spam_detection.ipynb
```

---

## Installation

Clone the repository

```
git clone https://github.com/abhijeetwarudkar40/Spam-SMS-Detection.git
cd Spam-SMS-Detection
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Run the Application

```
streamlit run app.py
```

The application will open automatically in your browser.

---

## Example

**Input**

```
Congratulations! You have won a free lottery ticket
```

**Output**

```
Spam
```

---

## Model Details

* Vectorizer: TF-IDF (3000 features)
* Algorithm: Multinomial Naive Bayes
* Problem Type: Binary Text Classification

---

## Future Improvements

* Online deployment
* Message history tracking
* Upload custom dataset
* Improve model accuracy

---

## Author

Abhijeet Warudkar

---

## License

This project is created for educational and learning purposes.
