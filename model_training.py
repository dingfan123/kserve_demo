import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

data = pd.merge(ratings, movies, on='movieId')

user_ids = data['userId'].unique().tolist()
movie_ids = data['movieId'].unique().tolist()
user_to_index = {x: i for i, x in enumerate(user_ids)}
movie_to_index = {x: i for i, x in enumerate(movie_ids)}
data['user'] = data['userId'].map(user_to_index)
data['movie'] = data['movieId'].map(movie_to_index)

num_users = len(user_ids)
num_movies = len(movie_ids)

X = data[['user', 'movie']].values
y = data['rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

user_input = tf.keras.layers.Input(shape=(1,), name='user')
movie_input = tf.keras.layers.Input(shape=(1,), name='movie')

user_embedding = tf.keras.layers.Embedding(num_users, 50, name='user_embedding')(user_input)
movie_embedding = tf.keras.layers.Embedding(num_movies, 50, name='movie_embedding')(movie_input)

dot_product = tf.keras.layers.Dot(axes=2)([user_embedding, movie_embedding])
flatten = tf.keras.layers.Flatten()(dot_product)

model = tf.keras.Model(inputs=[user_input, movie_input], outputs=flatten)
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=5, batch_size=64)

model.save('recommender_model/1/')
