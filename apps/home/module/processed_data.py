import pandas as pd
import re
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import seaborn as sns
from statistics import mean
from sklearn.metrics import accuracy_score, confusion_matrix
nltk.download('stopwords')
nltk.download('punkt')
import re
import string
import matplotlib.pyplot as plt

def preprocess_text(text):
    # Menghapus tanda baca
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text_cleaning=text
    # Case folding (mengubah huruf besar ke huruf kecil)
    text = text.lower()
    text = ''.join([char for char in text if not char.isdigit()])
    text_casefolding=text
    stop_words = set(stopwords.words('english'))
    # Membaca stopwords tambahan dari file .txt
    with open('apps/static/assets/filedata/stop_words_english.txt', 'r') as file:
        custom_stopwords = file.read().splitlines()

    # Menambahkan stopwords tambahan ke dalam set
    stop_words.update(custom_stopwords)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if 3 <= len(word) <= 15]
    text = ' '.join([word for word in tokens if word not in stop_words])
    text_stopword =text
    # Stemming menggunakan Porter Stemmer
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([stemmer.stem(word) for word in tokens])
    text_stemming=text
    #return text,text_cleaning,text_casefolding,text_stopword,text_stemming
    return text

def build_csv_file():
    
    file_path = 'apps/static/assets/filedata/output.csv'

    if os.path.exists(file_path):
         return "File 'output.csv' masih ada."
    else:
        df = pd.read_csv("apps/static/assets/filedata/paper.csv", usecols=['Title','Summary','Primary Category'])
        df['Title'] = df['Title'].apply(preprocess_text)
        df['Summary'] = df['Summary'].apply(preprocess_text)
        df.to_csv('apps/static/assets/filedata/output.csv', index=False)
        return "SUCCES"
    
def tfidf(X_train, X_test):
  vectorizer = TfidfVectorizer()
  X_train_tfidf = vectorizer.fit_transform(X_train)
  X_test_tfidf = vectorizer.transform(X_test)
  return X_train_tfidf, X_test_tfidf, vectorizer

def modelKnn(n,metric, X_train_tfidf, X_test_tfidf,y_train):
  knn = KNeighborsClassifier(n_neighbors=n, metric=metric, weights="distance")
  knn.fit(X_train_tfidf, y_train)
  y_pred = knn.predict(X_test_tfidf)
  return y_pred, knn
knn_model = []
vectorizer = []
best_accuracy = []
def processing(info,input_n,n_metrick):
    global best_n
    global best_metric
    global best_accuracy
    global vectorizer
    global knn_model
    global perf
    df = pd.read_csv("apps/static/assets/filedata/output.csv", usecols=['Title','Summary','Primary Category'])
    df.dropna(inplace=True)
    df['Text'] = df['Title'] + ' ' + df['Summary']
    X = df['Text']
    y = df['Primary Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_knn = input_n
    matricts = n_metrick
    perf=[]
    best_accuracy = 0
    for n in n_knn:
        #print("=== n : ", e)
        for matrict in matricts:
            X_train_tfidf, X_test_tfidf, vectorizer = tfidf(X_train, X_test)
            y_pred, knn = modelKnn(n,matrict, X_train_tfidf, X_test_tfidf,y_train)
            report = classification_report(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            #  confusion = confusion_matrix(y_test, y_pred)
            report_precision = mean(precision_score(y_test, y_pred,average=None))
            report_recall = mean(recall_score(y_test, y_pred,average=None))
            report_f1_score=mean(f1_score(y_test,y_pred,average=None))
            scores = cross_val_score(knn, X_train_tfidf, y_train, cv=5)
            mean_cv_accuracy = scores.mean()
            cv_accuracy_std_dev = scores.std()
            confusion = confusion_matrix(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                knn_model = knn
                best_n = n
                best_metric = matrict
            if info == "confusion":
                # Buat plot matriks konfusi
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                plt.savefig("apps/static/assets/images/confusion/cm_"+matrict+str(n)+".png", dpi=240)
            else:
                perf.append({"n":n,"metric":matrict,"accuracy":accuracy,"report":report,"f1_score":report_f1_score,"precision":report_precision,"recall":report_recall,'mean_cv_accuracy':mean_cv_accuracy,'cv_accuracy_std_dev':cv_accuracy_std_dev})
    return perf

# def predict(new_text):
#     model = knn_model
#     vectorizer = vectorizer
#     new_text_tfidf = vectorizer.transform([new_text])
#     predicted_category = model.predict(new_text_tfidf)
#     best_accuracy = best_accuracy
#     best_n = best_n
#     best_metric = best_metric

def predicted(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text_cleaning = text
    # Case folding (mengubah huruf besar ke huruf kecil)
    text = text.lower()
    text = ''.join([char for char in text if not char.isdigit()])
    text_casefolding = text
    # Tokenisasi dan menghapus stopwords
    stop_words = set(stopwords.words('english'))

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if 3 <= len(word) <= 15]
    text = ' '.join([word for word in tokens if word not in stop_words])
    text_stopword = text
    # Stemming menggunakan Porter Stemmer
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([stemmer.stem(word) for word in tokens])
    text_stemming = text
    ###############
    model = knn_model
    vectorizers = vectorizer
    new_text_tfidf = vectorizers.transform([text])
    predicted_category = model.predict(new_text_tfidf)
    best_accuracys = best_accuracy
    best_ns = best_n
    best_metrics = best_metric
    hasil=[]
    hasil.append({"predicted_class":predicted_category,"text":text,"text_cleaning":text_cleaning,"text_casefolding":text_casefolding,"text_stopword":text_stopword,"text_stemming":text_stemming,"best_accuracy":best_accuracys,'best_n':best_ns,'best_metric':best_metrics})  
    return hasil