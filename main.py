import os
import re
import sys
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec


CSV_PATH = "IMDB Dataset.csv" 
TEXT_COL = "review"             
SAMPLE_SIZE = None              
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 5
W2V_EPOCHS = 10
TOP_N_WORDS = 200
USE_TSNE = True                # False -> PCA , True -> t-SNE 
RANDOM_STATE = 42


if not os.path.exists(CSV_PATH):
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
if SAMPLE_SIZE:
    df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

texts = df[TEXT_COL].astype(str).tolist()
print(f"{len(texts)} loaded successful.")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"<.*?>", " ", text)                           
    text = re.sub(r"http\S+|www\.\S+", " ", text)                
    text = re.sub(r"[^A-Za-z']", " ", text)                      
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1 and not t.isdigit()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

print("Prepocessing")
sentences = [preprocess_text(t) for t in texts]
sentences = [s for s in sentences if len(s) > 0]
print(f"After prepocess count: {len(sentences)}")


all_tokens = [tok for sent in sentences for tok in sent]
word_counts = Counter(all_tokens)
most_common_words = [w for w, _ in word_counts.most_common(TOP_N_WORDS)]
print(f"Top-{TOP_N_WORDS} words was taken.")

print("Word2Vec Trainin")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=W2V_VECTOR_SIZE,
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    workers=4,
    epochs=W2V_EPOCHS,
    seed=RANDOM_STATE
)

MODEL_PATH = "w2v_imdb.model"
w2v_model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")

words_for_plot = [w for w in most_common_words if w in w2v_model.wv.key_to_index]
vectors = np.array([w2v_model.wv[w] for w in words_for_plot])
print(f"{len(words_for_plot)} will be rendered from TOP-{TOP_N_WORDS}).")


if USE_TSNE:
    print("Using t-SNE")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca")
    coords = tsne.fit_transform(vectors)
else:
    print("Using PCA")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(vectors)


plt.figure(figsize=(12, 9))
plt.scatter(coords[:,0], coords[:,1], s=20, alpha=0.7)
plt.title("Word2Vec: vector visualisation prepare")

for i, word in enumerate(words_for_plot):
    x, y = coords[i,0], coords[i,1]
    plt.text(x+0.01, y+0.01, word, fontsize=9)

plt.grid(True)
plt.tight_layout()
plt.savefig("word2vec_viz.png", dpi=150)
print("Visualisation saved as 'word2vec_viz.png'")

plt.show()

df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})


def vectorize_review(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv.key_to_index:
            vectors.append(model.wv[token])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(vectors, axis=0)

print("Векторизация отзывов...")
review_vectors = np.array([vectorize_review(tokens, w2v_model) for tokens in sentences])
labels = df['label'].values[:len(review_vectors)]

print("Форма матрицы признаков:", review_vectors.shape)
print("Форма вектора меток:", labels.shape)

pca = PCA(n_components=2, random_state=42)
review_vecs_2d = pca.fit_transform(review_vectors)

df_vis = pd.DataFrame(review_vecs_2d, columns=['x', 'y'])
df_vis['sentiment'] = df['sentiment'][:len(review_vecs_2d)]
plt.figure(figsize=(10, 8))
colors = {'positive': 'green', 'negative': 'red'}

for sentiment, color in colors.items():
    subset = df_vis[df_vis['sentiment'] == sentiment]
    plt.scatter(subset['x'], subset['y'], c=color, label=sentiment, alpha=0.5)

plt.title('Визуализация отзывов после Word2Vec (PCA 2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    review_vectors, labels, test_size=0.2, random_state=42
)

# --- Регрессия ---
#model = LogisticRegression(max_iter=1000)

# --- SVC ---
# model = SVC(kernel='linear', probability=True, random_state=42)

# --- Лес ---
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None, n_jobs=-1)

model.fit(X_train, y_train)
print("Модель обучена.")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")
print("\nОтчёт по классам:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.xlabel("Предсказание")
plt.ylabel("Истинное значение")
plt.title("Матрица ошибок")
plt.show()
