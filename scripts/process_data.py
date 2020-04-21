# Script to process and save the raw data
import pandas as pd
import json


columns = ['title', 'authors', 'abstract']  # these are the relevant columns we'll extract from each data set

# Load the data – sourced from arXiv.org
arxiv_data = []

# Set the path as needed
arxiv_metadata_path = '../data/arxiv-metadata-oai-2020-03-23.json'
for line in open(arxiv_metadata_path):
    arxiv_data.append(json.loads(line))

# Store the data in a DataFrame and save it as a parquet file
df_arxiv = pd.DataFrame(arxiv_data)

# Save storage space by only saving relevant columns
df_arxiv.loc[:, ['title', 'authors', 'abstract']].to_parquet('../data/arxiv_papers.parquet')


# Load the data – sourced from CORD-19
df_cord = pd.read_csv('../data/cord_metadata_raw.csv')

# Extract only the articles with both a title and an abstract
has_title = ~df_cord.loc[:, 'title'].isna()
has_abstract = ~df_cord.loc[:, 'abstract'].isna()

# Extract the useful columns and save the articles as a parquet file
df_cord[has_title & has_abstract].reset_index(drop=True).loc[:, columns].to_parquet('../data/cord_papers.parquet')



