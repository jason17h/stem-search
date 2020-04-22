import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import logging as log
import numpy as np
import pandas as pd
# from utils import *

log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=log.INFO)

sse = []
x = []
BATCH = 2500

log.info('Reading data...')
df = pd.read_parquet('../data/arxiv_papers.parquet')

# Vectorize the text data
log.info('Vectorize the text data...')
vec = TfidfVectorizer(stop_words='english')
tfidf = vec.fit_transform(df.loc[:, 'title'] * 2 + df.loc[:, 'abstract'])

log.info('Begin elbow method')

for k in range(800, 2000, 100):
    log.info('k = {}'.format(k))

    # Create model
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=BATCH, verbose=1)

    # Fit the model
    log.info('Fit the model...')
    try:
        kmeans.fit(tfidf)
    except:
        log.error('FAILED: k = {}'.format(k))

    # Update the lists
    log.info('Inertia {}: {}'.format(k, kmeans.inertia_))
    x.append(k)
    sse.append(kmeans.inertia_)
    print(sse)

log.info('Plot the data...')
plt.plot(x, sse, '-o', label='cord (batch size {})'.format(BATCH))
plt.legend()
plt.xlabel('k')
plt.ylabel('SSE')
plt.show()
plt.savefig('../arxiv_sse_2.png')
