# stem-search
STEMSearch – Research made easy! Find academic articles from a variety of sources to further your research projects.

Created by: Jason Huang (@jason17h)

## Setup
Note that the setup process will take many hours.

### Retrieving data
Clone the following repository and follow the instructions for harvesting the metadata: 
https://github.com/mattbierbaum/arxiv-public-datasets

`python bin/metadata.py [OPTIONAL filepath.json.gz]`

This will download the entire ArXiv metadata set at the default location of 
`$ARXIV_DATA/arxiv-metadata-oai-<date>.json.gz`. Move this to `stem-search/data` and unzip the files.

Next, find the COVID-19 Open Research Dataset (CORD-19) on Kaggle and download the metadata file:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge#metadata.csv. Rename this file as 
`cord_metadata.csv` to distinguish between our arXiv and CORD-19 metadata sets, and place it in `stemsearch/data`.

### Processing data
The script to process the data can be found in `scripts/process_data.py`. Run this via `python scripts/process_data.py`.
Steps in the data processing include:
- Processing the arXiv metadata into a DataFrame
- Filtering `cord_papers.csv` for entries with both a title and an abstract
- Extracting useful columns in each metadata set to save space
- Exporting the processed data into parquet files

### Fitting the model
The script to train and fit the machine learning model can be found in `scripts/model.py`. 
Run this via `python scripts/model.py`.

STEMSearch uses k-means clustering to group similar articles based on the term frequency-inverse document frequency
(TF-IDF) of their titles and abstracts. The models are implemented using scikit-learn. Learn more at the following:
- `MiniBatchKMeans`: 
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
    - https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
- `TfidfVectorizer`: 
    - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    - https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- `cosine_similarity`: 
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    - https://en.wikipedia.org/wiki/Cosine_similarity

## Usage

### Running the application
STEMSearch was created using Dash which runs on a Flask server. To run the application, run `python app.py` and go to
`localhost:8050` in your browser. It will take a few seconds to load.

### Using the application
The application is straightforward: to get article recommendations, enter the titles/abstracts of the journals you have
consulted already under the *My articles* tab and view the recommendations under the *Recommended* tab.

By default, the *Recommended* tab will show the top 20 similar articles. This can be changed by setting the `NUM_OF_RECS`
constant in `utils.py`. The articles are listed from most to least similar among those 20. Use the switch to toggle
between data sets – this will determine from which data set to make recommendations.

The green badges above each recommended article abstract represent the most important terms in the text (i.e. terms
exceeding a TF-IDF of 0.25) listed from most to least important. The *My TF-IDF* tab shows a graphical representation
of the most important terms in the user-provided corpus. The slider can be used to control how many words to display
in the graph.

The COVID-19 dashboard can be found at the bottom of the page below the article recommendation system. 

The *Scatter* tab will display the data in a scatter plot. Set the unit and scale for each axis as desired. 
Select a subset of the plot using box selection.

The *Map* tab will display the data in a bubble map. Select which figure to measure using the dropdown at the top.
