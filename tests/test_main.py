import os
import pandas as pd
from movie_recommendation_system.main import get_movie_recommendations, clean_title

# Import other functions from your main code file

cwd = "/PycharmProjects/movie_recommendation_system/src/movie_recommendation_system/movies.csv"
file_name = 'movies.csv'

file_path = os.path.join(cwd, file_name)

df = pd.read_csv(file_path)


def test_get_movie_recommendations():
    # Test case 1: Valid movie title
    movie_title = "The Dark Knight"
    recommendations = get_movie_recommendations(movie_title)
    assert len(recommendations) == 5, "Expected 5 recommendations"
    assert recommendations.iloc[0]["title"] == "The Dark Knight", "Expected 'The Dark Knight' as first recommendation"

    # Test case 2: Invalid movie title
    movie_title = "Invalid Movie Title"
    recommendations = get_movie_recommendations(movie_title)
    assert len(recommendations) == 0, "Expected 0 recommendations for invalid movie title"

    # Test case 3: Empty movie title
    movie_title = ""
    recommendations = get_movie_recommendations(movie_title)
    assert len(recommendations) == 0, "Expected 0 recommendations for empty movie title"

