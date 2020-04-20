# Use scikit-learn to train model
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

# Load the data
df_arxiv = pd.read_parquet('../data/arxiv_papers.parquet')
df_cord = pd.read_parquet('../data/cord_papers.parquet')

# Transform the text to add weight for the title
vt_arxiv = (df_arxiv.loc[:, 'title'] + ' ') * 2 + df_arxiv.loc[:, 'abstract']
vt_cord = (df_cord.loc[:, 'title'] + ' ') * 2 + df_cord.loc[:, 'abstract']

# Fit each TF-IDF vectorizer with the corresponding data
arxiv_vectorizer = TfidfVectorizer(stop_words='english')
cord_vectorizer = TfidfVectorizer(stop_words='english')

arxiv_tfidf = arxiv_vectorizer.fit_transform(vt_arxiv)
cord_tfidf = cord_vectorizer.fit_transform(vt_cord)

# Create the model
kmeans = MiniBatchKMeans(n_clusters=800, batch_size=1000, verbose=3)

# Train the model
arxiv_model = kmeans.fit(arxiv_tfidf)
cord_model = kmeans.fit(cord_tfidf)

# Save the vectorizers and models
with open('..model/arxiv_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(arxiv_vectorizer, f)

with open('../model/arxiv_model.pkl', 'wb') as f:
    pickle.dump(arxiv_model, f)

with open('..model/covid_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(cord_vectorizer, f)

with open('../model/covid_model.pkl', 'wb') as f:
    pickle.dump(cord_model, f)


