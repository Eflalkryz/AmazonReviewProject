
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


#Text Processing

df = pd.read_excel("amazon.xlsx")
df.head()

#NORMALİZİNG

df['Review']=df['Review'].str.lower() #Make them lower

#Punctuations

df['Review'] = df['Review'].str.replace('[^\w\s]', '') #\w=alphanumeric characters \s=space char

#Numbers

df['Review'] = df['Review'].str.replace('[\d]', '') #\d decimal
df['Review'].head(2)

#Stopwords

import nltk

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw)) #Bütün satırları dolaş bağlaç zamir gibi gereksiz kısımları at

## 1000 den az geçen kelimeleri çıkart.

temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]



df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in temp_df))


##LEMMATİZATİON : CÜMLENİN SONUNDAKİ EKLERİ KALDIRIR.

df['Review']= df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

tf= df['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns=["words", "tf"]
tf.sort_values("tf", ascending=False)

tf[tf["tf"]>500].plot.bar(x="words", y="tf")
plt.show()

#WORD CLOUD İLE KELİME FREKANSLARI


text=" ".join(i for i in df.Review) #bütün kelimeleri topla
wordcloud= WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud= WordCloud(max_font_size=50,
                     max_words=100,
                     background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")


#SENTİMENT ANALYSİS

#BİR METİN POZİTİF Mİ NEGATİF Mİ BAK

df["Review"].head()
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was great") #Önemli olan compound skoru

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_scores"]=df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"]) #0 dan büyükse pozitif

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"]= df["Review"].apply(lambda x:"pos" if sia.polarity_scores(x)["compound"] >0 else "neg")

df["sentiment_label"].value_counts() #pozitif yorum sayısı negatiften fazla

df.groupby("sentiment_label")["Star"].mean() #negatif verenlerin ortalama verdiği yıldız sayısı


#ENCODİNG


df["sentiment_label"]=LabelEncoder().fit_transform(df["sentiment_label"]) #label encoder kategorik değişkenler için

y=df["sentiment_label"] #bağımlı
X=df["Review"] #bağımsız değişken

#TD IDF VEXTORİZER

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf__idf_word = tf_idf_word_vectorizer.fit_transform(X) #Bağımsız değişkenleri de sayısallaştırma

#LOGİSTİC REGRESSİON

log_model = LogisticRegression().fit(X_tf__idf_word, y)

cross_val_score(log_model,
                X_tf__idf_word,
                y,
                scoring="accuracy",
                cv=5).mean() #%89 doğruluk oranı

random_rewiew = pd.Series(df["Review"].sample(1).values)

new_review = TfidfVectorizer().fit(X).transform(random_rewiew)
log_model.predict(new_review) #olumlu

#RANDOM FOREST

rf_model = RandomForestClassifier().fit(X_tf__idf_word, y)
cross_val_score(rf_model, X_tf__idf_word,y, cv=5, n_jobs=-1).mean() # Doğruluk oranı %91 çıktı

#HIPERPARAMETRE OPTİMİZAZYONU YAPARAK SONUÇLARIMIZI GELİŞTİREBİLİRİZ.


rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_tf__idf_word, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_tf__idf_word, y)

cross_val_score(rf_final, X_tf__idf_word, y, cv=5, n_jobs=-1).mean()
