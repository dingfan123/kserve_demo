import requests
import json

url = "http://localhost:8080/v1/models/movie-recommender:predict"
headers = {"Host": "movie-recommender"}

user_index = 1
movie_index = 1

data = {
    "instances": [
        {"user": [user_index], "movie": [movie_index]}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print("Prediction response:", response.json())
