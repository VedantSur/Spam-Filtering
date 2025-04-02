import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

spam_data_path = 'spam.csv' 
spam_df = pd.read_csv(spam_data_path, encoding='latin-1')
spam_df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
spam_df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

spam_df['text'] = spam_df['text'].str.lower()  
vectorizer = CountVectorizer(stop_words='english') 
X = vectorizer.fit_transform(spam_df['text']) 
y = np.where(spam_df['label'] == 'spam', 1, 0) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

alphas = [0.01, 0.1, 1.0]
results = {}

for alpha in alphas:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[alpha] = accuracy
    
    cm_df = pd.DataFrame(conf_matrix, 
                         index=['Actual Ham', 'Actual Spam'], 
                         columns=['Predicted Ham', 'Predicted Spam'])
    print(f'Alpha: {alpha}')
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm_df)
    print("Classification Report:\n", report)

plt.figure(figsize=(10, 6))
plt.bar(range(len(results)), list(results.values()), align='center', color='blue', alpha=0.7)
plt.xticks(range(len(results)), [f'Alpha={k}' for k in results.keys()])
plt.xlabel('Alpha Values')
plt.ylabel('Accuracy')
plt.title('Comparison of Different Alpha Values for MNB Classifier')
plt.show()

def plot_most_common_words(text, title, ax):
    vec = CountVectorizer(stop_words='english')
    data_vect = vec.fit_transform(text)
    sum_words = data_vect.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:15]
    df = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])
    sns.barplot(x='Frequency', y='Word', data=df, ax=ax, palette='viridis')
    ax.set_title(title)

def plot_most_common_ngrams(text, title, n=2, ax=None):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english')
    data_vect = vec.fit_transform(text)
    sum_words = data_vect.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:15]
    df = pd.DataFrame(words_freq, columns=['Bi-gram', 'Frequency'])
    sns.barplot(x='Frequency', y='Bi-gram', data=df, ax=ax, palette='plasma')
    ax.set_title(title)

spam_messages = spam_df[spam_df['label'] == 'spam']['text']
ham_messages = spam_df[spam_df['label'] == 'ham']['text']

fig, axes = plt.subplots(3, 2, figsize=(18, 15))
plot_most_common_words(spam_messages, "Top 15 most common words in spam messages", axes[0, 0])
plot_most_common_words(ham_messages, "Top 15 most common words in ham messages", axes[0, 1])
plot_most_common_ngrams(spam_messages, "Top 15 most common bi-grams in spam messages", ax=axes[1, 0])
plot_most_common_ngrams(ham_messages, "Top 15 most common bi-grams in ham messages", ax=axes[1, 1])

spam_df['message_length'] = spam_df['text'].apply(len)
sns.histplot(data=spam_df, x='message_length', hue='label', element='step', stat='density', common_norm=False, ax=axes[2, 0])
axes[2, 0].set_title('Message Length Distribution')
axes[2, 1].axis('off') 

plt.tight_layout()
plt.show()

sample_indices = random.sample(range(X_test.shape[0]), 10)
X_sample = X_test[sample_indices]
y_sample = y_test[sample_indices]
y_pred_sample = mnb.predict(X_sample)
sample_accuracy = accuracy_score(y_sample, y_pred_sample)
print("Accuracy on 10 random samples:", sample_accuracy)
