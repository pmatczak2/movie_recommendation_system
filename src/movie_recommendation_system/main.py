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

# line 44: This line of code is applying a function called clean_title to the title column of a Pandas DataFrame
# called movies. The apply method in Pandas allows you to apply a function to each element of a column and returns a
# new column with the same length as the original column.
#
# The clean_title function is most likely defined earlier in the code and is used to clean up the movie titles by
# removing any unnecessary characters, converting all characters to lowercase, etc.
#
# By applying this function to the title column, a new column called clean_title is created in the movies DataFrame,
# which contains the cleaned-up versions of the movie titles. This can be useful for data analysis and visualization
# purposes, as it can make it easier to group, filter, or search for movies based on their titles.

# line 47-48: In this code, a TfidfVectorizer object is created with an ngram range of 1 to 2. TfidfVectorizer is a
# method from the scikit-learn library, which converts a collection of raw text documents to a matrix of TF-IDF
# features.
#
# TF-IDF stands for term frequency-inverse document frequency. It is a technique used in text mining to reflect how
# important a word is to a document in a collection of documents. The TF-IDF value increases proportionally to the
# number of times a word appears in a document but is offset by the frequency of the word in the corpus. This helps
# to adjust for the fact that some words are more common in general than others and thus may not be as informative.
#
# The ngram_range parameter specifies the range of n-grams to consider when extracting features from the text. In
# this case, n-grams of size 1 and 2 are considered, which means that individual words as well as pairs of
# consecutive words will be considered as features.
#
# The fit_transform method of the TfidfVectorizer object is then called on the "clean_title" column of the movies
# DataFrame. This method fits the vectorizer to the text data and transforms it into a sparse matrix of TF-IDF
# features. This means that each row of the resulting matrix represents a document (in this case, a movie title) and
# each column represents a feature (in this case, a word or a pair of words). The value in each cell represents the
# TF-IDF score of the corresponding feature in the corresponding document.
#
# The resulting sparse matrix can be used as input for various machine learning models, such as clustering or
# classification algorithms, to analyze the text data and extract insights from it

# line 51-54: This code is asking the user to enter a movie title using the input function and storing the input in a
# variable called movie_title.
#
# Then, the get_movie_recommendations function is called with movie_title as its argument. This function presumably
# takes the input movie title, finds similar movies based on some criteria, and returns a DataFrame of recommended
# movies along with their titles and genres.
#
# Finally, the code prints the recommended movies along with their titles and genres using the print function. The
# DataFrame is accessed using the recommendations variable and the "title" and "genres" columns are selected using
# the double square bracket notation ([["title", "genres"]]).
#
# Overall, this code is a simple way to get movie recommendations based on user input, which can be useful for
# creating movie recommendation systems or for personal movie watching suggestions.