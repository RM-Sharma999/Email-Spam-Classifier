# Email Spam Classifier

This project focuses on classifying email messages as either spam or ham (non-spam) using classical machine learning models, ensemble methods, and natural language processing techniques. The goal is to develop a high-precision email spam filter that is reliable and practical for real-world deployment.

---

## Objective  
To build a reliable email spam classifier that accurately detects spam messages using machine learning and text preprocessing techniques, with a strong focus on high precision.

---

## Dataset Overview 
The dataset, obtained from the [RiverML](https://riverml.xyz/latest/api/datasets/trec07p/) repository, contains over 75,000 labeled text messages classified as either ham or spam. Although there is a slight class imbalance—approximately 61% ham and 39% spam—the distribution remains well within acceptable limits.

Understanding this distribution is essential for evaluating real-world model performance. Therefore, confusion matrices and precision-based metrics are prioritized to ensure reliable and effective spam detection.

---

## Exploratory Data Analysis (EDA)

### Distribution of Characters, Words, and Sentences by Class  
> 📊 Includes 3 subplots comparing char, word, and sentence counts in spam vs ham emails.

### 🔸 WordClouds  
> ☁️ Separate WordClouds for spam and ham to highlight commonly used terms.

### 🔸 Top 30 Most Common Words in Spam and Ham  
> 📈 Bar plot using TF-IDF vectorized terms for both spam and ham messages.

---

## 🧹 Text Preprocessing  

Steps performed:
- Merged subject + body  
- Converted to lowercase  
- Removed punctuation & symbols  
- Removed stopwords (NLTK)  
- Tokenized text  
- Lemmatized words using WordNet  
- Vectorized using `TfidfVectorizer(max_features=3000, min_df=5)`

---

## 🧪 Model Training & Evaluation  

> 📌 All models were evaluated using **accuracy, precision, and recall**.  
> 📉 Visualizations include:
- Confusion matrix grid (1x3) comparing:
  - MultinomialNB
  - BernoulliNB
  - ComplementNB  
- Bar plot comparing performance of:
  - Logistic Regression  
  - Random Forest  
  - Extra Trees  
  - SVC  
  - XGBoost  
  - BaggingClassifier  

---

## 🧠 Ensemble Methods: Voting Classifier  

Used a **soft voting** approach with six models:
- ExtraTreesClassifier  
- RandomForestClassifier  
- BaggingClassifier  
- XGBClassifier  
- SVC (sigmoid kernel)  
- Logistic Regression  

Improvements added:
- `class_weight={0:2, 1:1}` in most models  
- `scale_pos_weight` in XGBoost  
- Weighted ensemble: `[2, 2, 2, 1, 1, 1]`  
- Wrapped in a full preprocessing + model `Pipeline`

---

## 🎯 Threshold Tuning (Precision Focused)  

Rather than using the default 0.5 threshold, predicted spam probabilities were generated using `predict_proba()`.  
Then, thresholds were adjusted to **maximize precision** (i.e., fewer false positives).

> ✔️ Final threshold: **0.80**  
> 🎯 Result:
- Precision: 0.999  
- Recall: 0.962  
- False Positives: only 8  

> 📊 Includes precision-recall vs threshold graph to explain trade-offs.

---

## 🚀 Deployment  

App is deployed using **Streamlit**.  

### ✨ Features:  
- Input box for pasting email content  
- Real-time prediction result: **Spam** or **Not Spam**  
- Automatically clears the input box after showing results  
- Fast and light frontend experience

> ✅ Backend model uses optimized `VotingClassifier` pipeline with threshold filtering logic.

---

## 🛠 Technologies Used  
- Python  
- Scikit-learn  
- XGBoost  
- NLTK  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit  
- Pickle  
