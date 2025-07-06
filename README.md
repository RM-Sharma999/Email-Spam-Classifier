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

The final prediction was made by averaging the predicted probabilities across models, enhancing generalization and stability. This ensemble approach achieved approximately **99% accuracy**, **precision**, and **recall**, indicating highly reliable performance.

![](https://i.ibb.co/FLhVHnQr/download-6.png)

---

## Threshold Tuning (Precision Focused)  

By fine-tuning the probability threshold, the **voting classifier** was optimized to reduce false positives while maintaining strong recall. This trade-off is essential in spam detection, where incorrectly classifying legitimate (ham) emails as spam must be avoided.

![](https://i.ibb.co/GQv1nR0c/download-9.png)

---

## Technologies Used 

- **Programming Language:** Python  
- **NLP Tools:** `NLTK`, `re`
- **Visualization:** `Matplotlib`, `Seaborn`, `Wordcloud`  
- **Machine Learning:** `Scikit-learn`, `Xgboost`  
- **Web Interface:** `Streamlit`  
- **Deployment Platform:** `Render`

---

## Deployment

The final model was deployed using **Streamlit** to create an intuitive web-based interface, and hosted on **Render** for public accessibility.

[Email Spam Classifier Live App](https://email-spam-classifier-a61k.onrender.com)

---

## Key Takeaways

- **Random Forest and Extra Trees** classifiers showed outstanding standalone performance, achieving approximately **99% accuracy**, along with high precision and recall.
- The **Voting Classifier** delivered the **best results** with **99% accuracy, precision, and recall**, making it the most reliable choice for this task.  
- **Threshold tuning** significantly improved spam detection, reaching **96.3% recall**, **99.9% precision**, and leaving **only 8 false positives**.  
- Using **Streamlit** and **Render** simplified deployment and made the model usable in real-world applications.

