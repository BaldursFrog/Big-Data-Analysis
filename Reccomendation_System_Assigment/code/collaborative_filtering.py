import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['item_id', 'title'])

user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_top_n_recommendations(user_id, n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]  # Exclude self
    target_user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = target_user_ratings[target_user_ratings == 0].index
    predicted_ratings = {}
    for movie in unrated_movies:
        weighted_sum = 0
        similarity_sum = 0
        for similar_user, similarity in similar_users.items():
            if user_item_matrix.loc[similar_user, movie] > 0:
                weighted_sum += similarity * user_item_matrix.loc[similar_user, movie]
                similarity_sum += similarity
        if similarity_sum > 0:
            predicted_ratings[movie] = weighted_sum / similarity_sum

    top_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:n]
    recommendations = [(movie_id, movies.loc[movie_id, 'title'], rating) for movie_id, rating in top_movies]

    return recommendations

recommendations = get_top_n_recommendations(1, 5)
print("Top 5 recommendations for user 1:")
for movie_id, title, predicted_rating in recommendations:
    print(f"Movie ID: {movie_id}, Title: {title}, Predicted Rating: {predicted_rating:.2f}")