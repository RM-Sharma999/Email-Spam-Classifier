# Email Spam Classification Using Machine Learning

This project focuses on **classifying emails as either spam or ham (non-spam)** using **classical machine learning models, ensemble methods, and natural language processing techniques**. The goal is to develop a **high-precision email spam filter** that is **reliable and practical for real-world deployment**.

---

## Objective  
To build a **reliable email spam classifier** that **accurately detects spam emails** using **machine learning and text preprocessing techniques**, with a strong focus on **high precision**.

---

## Dataset Overview 
The dataset, obtained from the [RiverML](https://riverml.xyz/0.19.0/api/datasets/TREC07/) repository, contains **over 75,000 labeled emails** classified as either **ham or spam**. Although there is a slight class imbalance—approximately **61% ham and 39% spam**—the distribution remains **well within acceptable limits**.

Understanding this distribution is essential for evaluating **real-world model performance**. Therefore, **confusion matrices and precision-based metrics** are prioritized to ensure **reliable and effective spam detection**.

<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/b974d259-40d3-4cc0-a7d1-124f51124f92" />

---

## Exploratory Data Analysis (EDA)

### Distribution of Characters, Words, and Sentences by Class  
> Log-scaled distributions show that **spam emails tend to be shorter** in terms of **words, characters, and sentences**, but with noticeable variance.

![](https://i.ibb.co/N230NJ0t/download.png)

### Word Clouds for Spam and Ham Emails
> Word clouds highlight the **most frequently used terms** in **spam** and **ham** emails.

![](https://i.ibb.co/fG2hHmCz/download-2.png)

- **Spam emails** commonly include words such as:  
  `canadianpharmacy`, `cialis`, `viagra`, `pill`, `discount`, and `buy now`

- **Ham emails** often feature terms such as:  
  `reproducible`, `commit`, `reply`, `patch`, `function`, and `file`

### Top 30 Most Common Words in Spam and Ham  
> Separate bar plots were generated for **spam** and **ham** emails to showcase their **most frequently used words**. The results reveal distinct language patterns:

- **Spam emails** are dominated by promotional, commercial, and medical terms.
- **Ham emails** primarily include technical, informational, or casual language.

These insights help illustrate the contrasting vocabulary used in each email type, which is valuable for feature engineering and model interpretation.

![](https://i.ibb.co/q3gkT4Lc/download-3.png)
![](https://i.ibb.co/bMZW7fqW/download-4.png)

---

## Text Preprocessing  

**Steps performed:**
- Merged subject + body  
- Lowercasing 
- Removing punctuation and stopwords
- Tokenization
- Lemmatization (with NLTK)
- TF-IDF vectorization`

---

## Model Training & Evaluation  

### Naive Bayes Models

Three **Naive Bayes** variants were evaluated:

- Multinomial NB  
- Bernoulli NB  
- Gaussian NB  

**Multinomial NB** emerged as the most effective due to its compatibility with **count-based text features**.

![](https://i.ibb.co/2783YSQj/Screenshot-2025-07-03-125133.png)

### Baseline Models

Multiple traditional models—including **linear classifiers, tree-based algorithms, and boosting methods**—were trained using **TF-IDF features**. Their performance was evaluated and visualized using bar plots, offering a clear benchmark.

<img width="1001" height="1099" alt="image" src="https://github.com/user-attachments/assets/10aee6b3-3fe1-4ea3-a2aa-3a95c482faa0" />

---

## Ensemble Methods

To achieve more robust results, **ensemble techniques** were adopted by combining **diverse models** and leveraging their **collective strength**.

### Soft Voting Classifier

This ensemble combined predictions from:

- ExtraTreesClassifier  
- RandomForestClassifier  
- BaggingClassifier  
- XGBClassifier  
- SVC (sigmoid kernel)  
- Logistic Regression  

The final prediction was made by **averaging the predicted probabilities** across models, enhancing generalization and stability. This ensemble approach achieved approximately **99% accuracy**, **precision**, and **recall**, indicating **highly reliable performance**.

![](https://i.ibb.co/FLhVHnQr/download-6.png)

---

## Threshold Tuning (Precision Focused)  

By fine-tuning the **probability threshold**, the **voting classifier** was optimized to **reduce false positives** while maintaining **strong recall**. This trade-off is essential in spam detection, where incorrectly classifying legitimate (ham) emails as spam must be avoided.

![](https://i.ibb.co/GQv1nR0c/download-9.png)

---

## Technologies Used 

- **Programming Language:** `Python`  
- **NLP Tools:** `NLTK`, `Regex`
- **Visualization:** `Matplotlib`, `Seaborn`, `Wordcloud`  
- **Machine Learning:** `Scikit-learn`, `Xgboost`  
- **Web Interface:** `Streamlit`  
- **Deployment Platform:** `Render`

---

## Deployment

The final model was deployed using **Streamlit** to create an **intuitive web-based interface**, and hosted on **Render** for public accessibility.

[Email Spam Classifier Live App](https://email-spam-classifier-a61k.onrender.com)

---

## Key Takeaways

- **Random Forest and Extra Trees** classifiers delivered strong standalone performance with ~**99% accuracy**, high precision, and recall.
- The **Voting Classifier** delivered the **best results** with **99% accuracy, precision, and recall**, making it the most reliable choice for this task.  
- **Threshold tuning** significantly improved spam detection, reaching **96.3% recall**, **99.9% precision**, and leaving **only 8 false positives**.  
- Using **Streamlit** and **Render** simplified deployment and made the model usable in real-world applications.

