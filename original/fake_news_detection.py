import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

fake_df = pd.read_csv('Fake.csv')
real_df = pd.read_csv('True.csv')
fake_df['label'] = 0
real_df['label'] = 1

real_df['content'] = real_df['title'].fillna('') + ' ' + real_df['text'].fillna('')
fake_df['content'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')

df = pd.concat([real_df[['content','label']], fake_df[['content','label']]], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
y_proba_lr = lr_model.predict_proba(X_test_tfidf)[:,1]

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
y_proba_nb = nb_model.predict_proba(X_test_tfidf)[:,1]

print("=== Logistic Regression ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1       : {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_proba_lr):.4f}")

print("\n=== Naive Bayes ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nb):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_nb):.4f}")
print(f"F1       : {f1_score(y_test, y_pred_nb):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_proba_nb):.4f}")

lr_cm = confusion_matrix(y_test, y_pred_lr)
nb_cm = confusion_matrix(y_test, y_pred_nb)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, matrix, title in zip(axes, [lr_cm, nb_cm], ['Logistic Regression', 'Naive Bayes']):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{title} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
print("\nSaved confusion_matrices.png")

logreg_fpr, logreg_tpr, _ = roc_curve(y_test, y_proba_lr)
nb_fpr, nb_tpr, _ = roc_curve(y_test, y_proba_nb)
plt.figure(figsize=(8,6))
plt.plot(logreg_fpr, logreg_tpr, label=f'Logistic Regression (AUC={auc(logreg_fpr,logreg_tpr):.2f})')
plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC={auc(nb_fpr,nb_tpr):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')
print("Saved roc_curve.png")

print("\nDone!")

print("\n" + "="*50)
print("FAKE NEWS DETECTOR - Enter any news article")
print("="*50)
while True:
    print("\nPaste your news article below (or type 'quit' to exit):")
    user_input = input()
    if user_input.lower() == 'quit':
        break
    if len(user_input.strip()) == 0:
        print("Please enter some text!")
        continue
    input_tfidf = tfidf.transform([user_input])
    lr_prob = lr_model.predict_proba(input_tfidf)[0][1]
    nb_prob = nb_model.predict_proba(input_tfidf)[0][1]
    hybrid_prob = (lr_prob + nb_prob) / 2
    print("\n--- RESULT ---")
    if hybrid_prob >= 0.5:
        print(f"REAL NEWS (Confidence: {hybrid_prob*100:.1f}%)")
    else:
        print(f"FAKE NEWS (Confidence: {(1-hybrid_prob)*100:.1f}%)")
    print(f"Logistic Regression: {lr_prob*100:.1f}% real")
    print(f"Naive Bayes: {nb_prob*100:.1f}% real")
    print(f"Hybrid Score: {hybrid_prob*100:.1f}% real")
