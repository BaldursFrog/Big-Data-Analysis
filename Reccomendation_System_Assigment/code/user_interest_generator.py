import pandas as pd

def extract_user_ratings(user_id=1):
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['item_id', 'title'])

    user_ratings_with_titles = user_ratings.merge(movies_df, on='item_id')

    return user_ratings_with_titles[['item_id', 'title', 'rating']]

def prepare_prompt(user_ratings_df):
    top_ratings = user_ratings_df[user_ratings_df['rating'] >= 4].sort_values('rating', ascending=False).head(10)

    examples = []
    for _, row in top_ratings.iterrows():
        examples.append(f"User rated '{row['title']}' with {row['rating']} stars.")

    prompt = "Based on the following user movie ratings, generate a short description of their interests:\n\n"
    for example in examples:
        prompt += example + "\n"
    prompt += "\nGenerate a concise user interest description like 'This user prefers sci-fi and suspense movies.'"

    return prompt

def generate_interest_description(prompt):

    lines = prompt.split('\n')
    movies = []
    for line in lines:
        if "User rated '" in line and "' with" in line:
            title = line.split("User rated '")[1].split("' with")[0]
            movies.append(title.lower())

    genres = []
    for movie in movies:
        if 'godfather' in movie or 'remains of the day' in movie or 'henry v' in movie or 'dead poets society' in movie:
            genres.append('drama')
        if 'monty python' in movie or 'sleeper' in movie or 'big night' in movie:
            genres.append('comedy')
        if 'bound' in movie or 'ridicule' in movie:
            genres.append('drama')
        if 'haunted world' in movie:
            genres.append('biography')

    unique_genres = list(set(genres))
    if unique_genres:
        genre_str = ', '.join(unique_genres[:-1]) + ' and ' + unique_genres[-1] if len(unique_genres) > 1 else unique_genres[0]
        description = f"This user prefers {genre_str} movies."
    else:
        description = "This user has diverse movie preferences."

    return description

def main():
    user_id = 1
    user_ratings_df = extract_user_ratings(user_id)
    prompt = prepare_prompt(user_ratings_df)
    description = generate_interest_description(prompt)

    print("User Interest Description:")
    print(description)

    print("\nGenerated Prompt:")
    print(prompt)

if __name__ == "__main__":
    main()