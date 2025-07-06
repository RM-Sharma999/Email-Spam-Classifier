# Email Spam Classification Using Machine Learning

This project focuses on classifying email messages as either spam or ham (non-spam) using classical machine learning models, ensemble methods, and natural language processing techniques. The goal is to develop a high-precision email spam filter that is reliable and practical for real-world deployment.

---

## Objective  
To build a reliable email spam classifier that accurately detects spam messages using machine learning and text preprocessing techniques, with a strong focus on high precision.

---

## Dataset Overview 
The dataset, obtained from the [RiverML](https://riverml.xyz/latest/api/datasets/trec07p/) repository, contains over 75,000 labeled text messages classified as either ham or spam. Although there is a slight class imbalance—approximately 61% ham and 39% spam—the distribution remains well within acceptable limits.

Understanding this distribution is essential for evaluating real-world model performance. Therefore, confusion matrices and precision-based metrics are prioritized to ensure reliable and effective spam detection.

![](https://i.ibb.co/hxR16L6h/Screenshot-2025-07-02-204153.png)

---

## Exploratory Data Analysis (EDA)

### Distribution of Characters, Words, and Sentences by Class  
> Log-scaled distributions show that spam messages tend to be shorter in terms of words, characters, and sentence count, but with noticeable variance.

![](https://i.ibb.co/N230NJ0t/download.png)

### WordClouds  
> Word clouds were created separately for **spam** and **ham** emails to highlight the most frequently used terms in each category.

- **Spam emails** commonly include words such as:  
  `canadianpharmacy`, `cialis`, `viagra`, `pill`, `discount`, `buy now`

- **Ham emails** often feature terms like:  
  `reproducible`, `commit`, `reply`, `patch`, `function`, `file`

These visualizations provide a quick and intuitive understanding of the distinct vocabulary used in each type of email.

![](https://i.ibb.co/fG2hHmCz/download-2.png)

### Top 30 Most Common Words in Spam and Ham  
> Separate bar plots were generated for **spam** and **ham** emails to showcase their most frequently used words. The results reveal distinct language patterns:

- **Spam emails** are dominated by promotional, commercial, and medical terms.
- **Ham emails** primarily include technical, informational, or casual language.

These insights help illustrate the contrasting vocabulary used in each email type, which is valuable for feature engineering and model interpretation.

![](https://i.ibb.co/q3gkT4Lc/download-3.png)
![](https://i.ibb.co/bMZW7fqW/download-4.png)
---

## Text Preprocessing  

Steps performed:
- Merged subject + body  
- Lowercasing 
- Removing punctuation and stopwords
- Tokenization
- Lemmatization (with NLTK)
- TF-IDF vectorization`

---

## Model Training & Evaluation  

### Naive Bayes Models

Three Naive Bayes variants were evaluated:

- Multinomial NB  
- Bernoulli NB  
- Gaussian NB  

Multinomial NB emerged as the most effective due to its compatibility with count-based text features.

![](https://i.ibb.co/2783YSQj/Screenshot-2025-07-03-125133.png)

### Baseline Models

Multiple traditional models—including linear classifiers, tree-based algorithms, and boosting methods—were trained using TF-IDF features. Their performance was evaluated and visualized using bar plots, offering a clear benchmark.

![](https://i.ibb.co/gMMmtFsv/download-5.png)

---

## Ensemble Methods

To achieve more robust results, ensemble techniques were adopted by combining diverse models and leveraging their collective strength.

### Soft Voting Classifier

This ensemble combined predictions from:

- ExtraTreesClassifier  
- RandomForestClassifier  
- BaggingClassifier  
- XGBClassifier  
- SVC (sigmoid kernel)  
- Logistic Regression  

The final decision was made by averaging the predicted probabilities. This method improved generalization and precision, achieving ~98% accuracy.

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
