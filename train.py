import numpy as np
import pandas as pd
import os, pickle
from scipy import sparse

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score

from utils import transform_text, TextCleaner, to_dense

df = pd.read_csv("trec2007.csv")

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc = True)

df = df[['subject', 'body', 'label']].copy()

# print(df.shape)

## `Data Cleaning`

# Replace empty strings with NAN
df['subject'] = df['subject'].replace('', np.nan)
df['body'] = df['body'].replace('', np.nan)

# rename the columns
df.rename(columns = {'subject':'email_subject', 'body': 'email_body', 'label':'target'}, inplace = True)

# label encode the target values
encoder = LabelEncoder()

df["target"] = encoder.fit_transform(df["target"])

# drop the missing values
df.dropna(subset = ['email_subject', 'email_body'], inplace = True)

# drop duplicates (true duplicates)
df.drop_duplicates(subset = ['email_subject', 'email_body', 'target'], keep = "first", inplace = True)

df["email_text"] = df["email_subject"] + " " + df["email_body"]

## `Data Splitting`

# Separate the data into feature and target variables
X = df[["email_text"]]  # pandas DataFrame
y = df["target"]  # pandas Series

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2, stratify = y)

## `Data Preprocessing`

# Function to convert numeric data to sparse matrix (avoids sparse-dense mix errors)
def to_sparse(X):
  return sparse.csr_matrix(X)

to_sparse_transformer = FunctionTransformer(to_sparse, accept_sparse = True)

#---------------------------------------------------

# Preprocessing pipelines

# Text pipeline
text_pipeline = Pipeline([
    ('text_cleaner', TextCleaner()),
    ('text_vectorizer', TfidfVectorizer(max_features = 3000))
    # ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse = True)),       # Uncomment if wanna use it
    # ('scaler', MinMaxScaler())
])

# Numeric pipeline (for features like 'num_characters' etc)
num_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),           # Scaling optional
    ('to_sparse', to_sparse_transformer)    # Convert to sparse for TF-IDF compatibility
])

# Full preprocessing pipeline
full_preprocessing = ColumnTransformer([
    ('text', text_pipeline, 'email_text')
    # ('num', num_pipeline, ['num_characters'])    # Uncomment if wanna use it
])

## Model Building

# Voting Classifier
class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])   # for Xgb (similiar to class weight)

svc_dense = make_pipeline(
    FunctionTransformer(to_dense, accept_sparse = True),
    SVC(kernel = 'sigmoid', gamma = 1.0, probability = True, class_weight = {0:2, 1:1})
)

base_models_vc = [
('etc', ExtraTreesClassifier(n_estimators = 50, random_state = 2, class_weight={0:2, 1:1})),
('rfc', RandomForestClassifier(n_estimators = 50, random_state = 2, class_weight={0:2, 1:1})),
('bc', BaggingClassifier(n_estimators = 50, random_state = 2)),
('xgb', XGBClassifier(n_estimators = 50, scale_pos_weight = class_ratio, random_state = 2)),
('svc', svc_dense),
('lrc', LogisticRegression(solver = "liblinear", penalty = "l1", class_weight={0:2, 1:1}))
]

# Define the Voting Classifier
voting_clf = VotingClassifier(estimators = base_models_vc, voting = 'soft', weights = [2, 2, 2, 1, 1, 1], n_jobs = -1)

# Wrap the Voting Classifier inside a Pipeline
voting_pipeline = Pipeline([
    ('preprocessing', full_preprocessing),
    ('voting', voting_clf)
])

# Train the Voting Classifier
voting_pipeline.fit(X_train, y_train)

# Predict on X_test
y_pred = voting_pipeline.predict(X_test)

# print("Accuracy Score for Voting Classifier:", accuracy_score(y_test, y_pred))
# print("Precision Score for Voting Classifier:", precision_score(y_test, y_pred, zero_division = 0))
# print("Recall Score for Voting Classifier:", recall_score(y_test, y_pred, zero_division = 0))

file_path = os.path.join(os.path.dirname(__file__), 'voting_pipeline_with_threshold.pkl')

model_with_new_threshold = {
    "pipeline": voting_pipeline,
    "threshold": 0.80
}

with open(file_path, 'wb') as f:
    pickle.dump(model_with_new_threshold, f)

print("Model saved to:", file_path)