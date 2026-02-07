import os
import zipfile
import pandas as pd
import requests
from io import BytesIO


if os.path.exists('amazon_reviews.csv'):
    df = pd.read_csv('amazon_reviews.csv')
else:
    import numpy as np
    np.random.seed(42)
    n_samples = 10000
    reviews = [
        "This product is amazing! I love it.",
        "Terrible quality, very disappointed.",
        "Great value for money, highly recommend.",
        "Worst purchase ever, do not buy.",
        "Excellent service and fast delivery.",
        "Bad experience, will not return.",
        "Fantastic item, exceeded expectations.",
        "Poor build quality, broke after a week.",
        "Love this product, best I've bought.",
        "Awful, completely useless."
    ] * (n_samples // 10)
    ratings = np.random.choice([1,2,3,4,5], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    df = pd.DataFrame({
        'reviewText': reviews[:n_samples],
        'overall': ratings
    })
    df.to_csv('amazon_reviews.csv', index=False)
    print("Mock dataset created as 'amazon_reviews.csv'")


df.dropna(subset=['reviewText', 'overall'], inplace=True)

df['sentiment'] = df['overall'].apply(lambda x: 1 if x >= 4 else 0 if x <= 2 else None)
df.dropna(subset=['sentiment'], inplace=True)

df.to_csv('processed_amazon_reviews.csv', index=False)

print("Dataset prepared and saved as 'processed_amazon_reviews.csv'")
print(f"Shape: {df.shape}")
print(df.head())