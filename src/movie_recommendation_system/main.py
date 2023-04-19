import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


def get_movie_recommendations(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    result = movies.iloc[indices][::-1]
    return result


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_users_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]

    similar_users_recs = similar_users_recs.value_counts() / len(similar_users)
    similar_users_recs = similar_users_recs[similar_users_recs > .10]

    all_users = ratings[(ratings["movieId"].isin(similar_users_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_users_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


# Preprocess movie titles for cosine similarity
movies["clean_title"] = movies["title"].apply(clean_title)

# Create TF-IDF vectorizer and fit on movie titles
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Example usage of get_movie_recommendations() function
movie_title = input("Enter a movie title: ")
recommendations = get_movie_recommendations(movie_title)
print("Movie Recommendations:")
print(recommendations[["title", "genres"]])


# line1: This line imports the pandas library and assigns it the alias pd.

# line2: This line imports the built-in regular expression module re.

# line3-4: These lines import two classes from the sklearn library: TfidfVectorizer,
# which is used to calculate term frequency-inverse document frequency values for a text corpus,
# and cosine_similarity, which is used to calculate the cosine similarity between two matrices.

# line5:This line imports the numpy library and assigns it the alias np.

# line7-8: These lines use the read_csv() method from pandas to read two CSV files called movies.csv and ratings.csv
# and assign them to the variables movies and ratings, respectively.

# line11-12:This is a function called clean_title() that takes a string as input and returns a new string with all
# non-alphanumeric characters removed.

# line15-21: This is a function called get_movie_recommendations() that takes a movie title as input and returns a
# DataFrame of the top 5 movies most similar to the input title, based on cosine similarity of their titles. The
# function first calls clean_title() on the input title to remove non-alphanumeric characters, then uses
# TfidfVectorizer to calculate the TF-IDF vector of the cleaned input title. The function then calculates the cosine
# similarity between the input title and all movie titles in the movies DataFrame, selects the top 5 most similar
# movies based on the cosine similarity scores, and returns a DataFrame of those movies.

# line24-40:The function is defined with one argument, which is the ID of the movie that we want to find similar
# movies for. The first line finds all the users who have rated the input movie highly (greater than 4) and returns
# their unique user IDs.
# The second line finds all the movies that have been rated highly by these similar users, and returns the IDs of
# these movies.
# The third line calculates the percentage of similar users who have rated each of these recommended movies,
# and keeps only those movies that have been recommended by at least 10% of the similar users.
# The fourth line finds all the users who have rated these recommended movies highly (greater than 4), and calculates
# the percentage of all users who have rated each of these recommended movies.
# The fifth line concatenates the two percentage dataframes (similar users and all users) horizontally, adds a new
# column that calculates the ratio of similar users' ratings to all users' ratings, and sorts the dataframe in
# descending order based on this score.
# The sixth line returns the top 10 recommended movies with their scores, titles, and genres, by merging the
# recommendations dataframe with a separate dataframe containing movie titles and genres.
# Overall, this function is recommending movies that are highly rated by users who have also rated the input movie
# highly, and are also popular among all users who have rated the recommended movies highly. The resulting
# recommendations are sorted by a score that gives higher weight to movies that are recommended by a larger
# percentage of similar users.