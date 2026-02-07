import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re

df = pd.read_csv('processed_amazon_reviews.csv')

labeled_df, unlabeled_df = train_test_split(df, test_size=0.8, random_state=42, stratify=df['sentiment'])
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
@labeling_function()
def lf_positive_keywords(x):
    positive_words = ['excellent', 'amazing', 'great', 'love', 'best', 'perfect', 'awesome', 'fantastic']
    return POSITIVE if any(word in x.reviewText.lower() for word in positive_words) else ABSTAIN

@labeling_function()
def lf_negative_keywords(x):
    negative_words = ['terrible', 'bad', 'worst', 'hate', 'awful', 'horrible', 'disappointed', 'poor']
    return NEGATIVE if any(word in x.reviewText.lower() for word in negative_words) else ABSTAIN

@labeling_function()
def lf_rating_high(x):
    return POSITIVE if x.overall >= 4 else ABSTAIN

@labeling_function()
def lf_rating_low(x):
    return NEGATIVE if x.overall <= 2 else ABSTAIN

@labeling_function()
def lf_long_review_positive(x):
    return POSITIVE if len(x.reviewText) > 500 and '!' in x.reviewText else ABSTAIN

@labeling_function()
def lf_short_negative(x):
    return NEGATIVE if len(x.reviewText) < 100 and any(word in x.reviewText.lower() for word in ['bad', 'terrible']) else ABSTAIN

lfs = [lf_positive_keywords, lf_negative_keywords, lf_rating_high, lf_rating_low, lf_long_review_positive, lf_short_negative]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=unlabeled_df)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=100, seed=123)

probs_train = label_model.predict_proba(L_train)
X_train = unlabeled_df['reviewText'] 
y_train = probs_train.argmax(axis=1)  
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

clf = LogisticRegression(random_state=42)
clf.fit(X_train_vec, y_train)
X_test = labeled_df['reviewText']
y_test = labeled_df['sentiment']
X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)
print("Snorkel Experiment Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

print("LF Summary:")
lf_names = ['lf_positive_keywords', 'lf_negative_keywords', 'lf_rating_high', 'lf_rating_low', 'lf_long_review_positive', 'lf_short_negative']
for i, lf in enumerate(lfs):
    coverage = (L_train[:, i] != -1).sum() / len(L_train)
    print(f"LF {i}: {lf_names[i]}, Coverage: {coverage:.2f}")