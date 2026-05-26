# 📧 Spamify - An Email Classifier using Machine Learning

An AI-powered Spam Email Classification Web Application built using **Machine Learning**, **Natural Language Processing (NLP)**, and **Flask**.  
This project detects whether an email/message is **Spam** or **Not Spam** with probability scores and an interactive modern UI.

---

# 🚀 Live Features

✅ Spam Email Detection  
✅ Machine Learning-based Prediction  
✅ Probability Score Visualization  
✅ Drag & Drop Email Upload  
✅ Animated Glassmorphism UI  
✅ Real-time Prediction  
✅ Responsive Web Design  
✅ NLP Text Processing  
✅ File Upload Support  

---

# 🧠 Machine Learning Workflow

The model follows the complete NLP pipeline:

```txt
Email Text
   ↓
Text Preprocessing
   ↓
TF-IDF / Count Vectorization
   ↓
Multinomial Naive Bayes
   ↓
Spam Prediction
```

---

# 🛠️ Tech Stack

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- Python
- Flask

## Machine Learning

- Scikit-learn
- Multinomial Naive Bayes
- TF-IDF Vectorizer
- NLP

---

# 📂 Project Structure

```txt
Spam-Email-Classifier-Debdut-Nandy/
│
├── app.py
├── train_model.py
├── Spam_Email_Classifier_Debdut_Nandy.ipynb
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── spam.csv
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Deb124-source/Spam-Email-Classifier-Debdut-Nandy.git
```

---

## 2️⃣ Move into Project Folder

```bash
cd Spam-Email-Classifier-Debdut-Nandy
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
python app.py
```

The application will start on:

```txt
http://127.0.0.1:5000
```

---

# 🧪 Model Training

The machine learning model is trained using:

- Multinomial Naive Bayes
- TF-IDF / Count Vectorizer
- NLP preprocessing

To retrain model:

```bash
python train_model.py
```

---

# 💡 Features Explained

## 📧 Spam Detection

Predicts whether an email is:

- Spam 🚨
- Safe ✅

---

## 📈 Probability Score

Displays confidence score for:

- Spam Probability
- Safe Probability

---

## 🎨 Modern Animated UI

Includes:

- Glassmorphism Design
- Smooth Animations
- Responsive Layout
- Hover Effects

---

## 📸 Screenshots

<img src="images/Screenshot 2026-05-26 110319.png" width="700"/>

---

# 🔥 Future Improvements

- Gmail API Integration
- Real-time Inbox Scanning
- AI-based Phishing Detection
- Deep Learning (LSTM/BERT)
- User Authentication
- MongoDB Integration
- Dashboard Analytics
- Dark/Light Theme Toggle
- Email History Tracking

---

# 📦 requirements.txt

```txt
flask
scikit-learn
pandas
numpy
gunicorn
```

---

# 🧠 NLP Techniques Used

- Tokenization
- Stopword Removal
- TF-IDF Vectorization
- Text Classification

---

# 📌 Model Information

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes | High Accuracy(~98%)|

---

# 🌐 Deployment

This project can be deployed on:

- Render
- Railway
- Vercel
- PythonAnywhere

---

# 🚀 Deployment Command

## Start Command

```bash
gunicorn app:app
```

---

# 👨‍💻 Author

## Debdut Nandy

Machine Learning & Data Analytics Enthusiast

---

# ⭐ If You Like This Project

Give this repository a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the MIT License.
