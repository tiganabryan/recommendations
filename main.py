import pandas as pd
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

tasks = pd.read_csv("tasks.csv")
difficulty_ratings = pd.read_csv("difficultyRatings.csv")

# remove duplicate rows
tasks.drop_duplicates(inplace=True)
difficulty_ratings.drop_duplicates(inplace=True)

# remove missing values
tasks.dropna(inplace=True)
difficulty_ratings.dropna(inplace=True)

# extracting categories column from tasks csv
categories = tasks["categories"]
# print(categories)

# instance of onehotencoder. to create [0,0,0,1,0,0] identifiers
encoder = OneHotEncoder()

# creating categories column
categories_encoded = encoder.fit_transform(categories.values.reshape(-1, 1))

# print(categories_encoded)

# instance of NearestNeighbors class
recommender = NearestNeighbors(metric="cosine")

recommender.fit(categories_encoded.toarray())

# tasks user did
task_index = 0

# number of recommendations to return
number_recommendations = 4

_, recommendations = recommender.kneighbors(
    categories_encoded[task_index].toarray(), n_neighbors=number_recommendations
)

# hardest_task_predictions = tasks.iloc[recommendations[0]["title"]]
hardest_task_predictions = tasks.iloc[recommendations[0]]

print(hardest_task_predictions)

# print(recommendations)
