import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cosine

# Load and clean data from text file
data = pd.read_csv('data.txt')

# Remove rows with missing or incorrect values
data = data.dropna()

# Handle irregularities
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
data = data.dropna()

# Data Analysis
average_ratings = data.groupby('Movie')['Rating'].mean()
popular_movies = average_ratings.sort_values(ascending=False)

# Collaborative Filtering
user_movie_ratings = data.pivot_table(index='User', columns='Movie', values='Rating', fill_value=0)

def get_movie_recommendations(user_name):
    try:
        # Find the index of the user in the user_movie_ratings DataFrame
        user_index = user_movie_ratings.index.get_loc(user_name)

        # Get the ratings of the user
        user_ratings = user_movie_ratings.iloc[user_index]

        # Compute cosine similarity between the user and all other users
        similarity_scores = user_movie_ratings.apply(lambda row: 1 - cosine(user_ratings, row), axis=1)

        # Identify movies the user has not rated
        unrated_movies = user_movie_ratings.columns[user_movie_ratings.loc[user_name] == 0]

        # Predict ratings for unrated movies based on similarity scores
        predicted_ratings = user_movie_ratings.loc[:, unrated_movies].multiply(similarity_scores, axis=0).sum(axis=0) / similarity_scores.sum()

        # Filter predicted ratings for unrated movies
        user_predicted_ratings = predicted_ratings[unrated_movies]

        # Recommend movies with the highest predicted ratings
        recommendations = user_predicted_ratings.sort_values(ascending=False)

        return recommendations.head(5)

    except KeyError:
        print(f"\nUser {user_name} not found. Please check the entered name.")
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return pd.Series(dtype=float)

# User Interface
user_name = input("Enter your name: ")

recommendations = get_movie_recommendations(user_name)
if not recommendations.empty:
    print(f"\nHello {user_name}, here are some movie recommendations for you:")
    print(recommendations)
