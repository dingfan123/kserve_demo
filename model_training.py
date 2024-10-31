import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
ratings = pd.read_csv('small_dataset/ratings.csv')
movies = pd.read_csv('small_dataset/movies.csv')

# Map user and movie IDs to continuous indices
user_ids = ratings['userId'].unique().tolist()
movie_ids = ratings['movieId'].unique().tolist()
user_to_index = {id: idx for idx, id in enumerate(user_ids)}
movie_to_index = {id: idx for idx, id in enumerate(movie_ids)}

# Apply mappings to ratings data
ratings['user'] = ratings['userId'].map(user_to_index)
ratings['movie'] = ratings['movieId'].map(movie_to_index)

# Number of unique users and movies
num_users = len(user_ids)
num_movies = len(movie_ids)

# Prepare training data
X = ratings[['user', 'movie']].values
y = ratings['rating'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
user_input = tf.keras.layers.Input(shape=(1,), name='user')
movie_input = tf.keras.layers.Input(shape=(1,), name='movie')

user_embedding = tf.keras.layers.Embedding(num_users, 50, name='user_embedding')(user_input)
movie_embedding = tf.keras.layers.Embedding(num_movies, 50, name='movie_embedding')(movie_input)

dot_product = tf.keras.layers.Dot(axes=2)([user_embedding, movie_embedding])
flatten = tf.keras.layers.Flatten()(dot_product)

model = tf.keras.Model(inputs=[user_input, movie_input], outputs=flatten)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=5, batch_size=64)

# Save the model in SavedModel format
import os
os.makedirs('recommender_model/1', exist_ok=True)
tf.saved_model.save(model, 'recommender_model/1/')
