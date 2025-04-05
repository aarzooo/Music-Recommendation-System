import re
import os
import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join('models', 'kmeans_model.pkl')
DATA_PATH = os.path.join('data', 'songs_with_clusters.csv')

# Load the data and model (do this ONCE at app startup)
try:
    data = pd.read_csv(DATA_PATH)
    kmeans_model = pickle.load(open(MODEL_PATH, 'rb'))
    print("Model and data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data or model: {e}")
    data = None
    kmeans_model = None
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    data = None
    kmeans_model = None

def recommend_songs_by_cluster_kmeans(song_name, data):
    print("find_similar_songs : Start")
    if data is None:
        return pd.DataFrame(columns=['track_name', 'track_artist'])

    try:
        selected_song = data[data['track_name'].str.lower() == song_name.lower()]

        if selected_song.empty:
            song_name_escaped = re.escape(song_name)
            selected_song = data[data['track_name'].str.contains(song_name_escaped, case=False, na=False)]

            if selected_song.empty:
                print(f"Song '{song_name}' not found in the dataset.")
                return pd.DataFrame(columns=['track_name', 'track_artist'])

        if len(selected_song) > 1:
            selected_song = selected_song.iloc[[0]]

        if 'cluster_kmeans' not in selected_song.columns:
            return pd.DataFrame(columns=['track_name', 'track_artist'])

        cluster = selected_song['cluster_kmeans'].iloc[0]
        selected_track_name = selected_song['track_name'].iloc[0]

        if 'track_name' not in data.columns:
            return pd.DataFrame(columns=['track_name', 'track_artist'])

        similar_songs = data[
            (data['cluster_kmeans'] == cluster) & (data['track_name'] != selected_track_name)
        ]

        if similar_songs.empty:
            print("No similar songs found.")
            return pd.DataFrame(columns=['track_name', 'track_artist'])

        print("find_similar_songs : End")
        return similar_songs[['track_name', 'track_artist']].head(10)

    except Exception as e:
        print(f"Error during recommendation: {e}")
        return pd.DataFrame(columns=['track_name', 'track_artist'])


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error_message = None
    song_list = data['track_name'].unique().tolist() if data is not None else []
    selected_song_name = None

    if request.method == 'POST':
        song_name = request.form.get('song_name')

        if not song_name:
            error_message = "Please select a song from the list."
        elif data is None:
            error_message = "Error: Song data could not be loaded."
        else:
            selected_song_name = song_name
            result = find_similar_songs(song_name, data)

            if isinstance(result, pd.DataFrame):
                recommendations = result
            else:
                error_message = "Unexpected error occurred. Please try again later."
                recommendations = None

    return render_template('index.html',
                           recommendations=recommendations,
                           error_message=error_message,
                           song_list=song_list,
                           selected_song_name=selected_song_name)





if __name__ == '__main__':
    app.run(debug=True, port=8000)
