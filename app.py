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

    # Basic check if data is usable (added for minimal safety, notebook version might assume valid data)
    if data is None or not isinstance(data, pd.DataFrame):
        print("Error: Input data is invalid.")
        return None # Return None for consistency with "not found" case
    if not all(col in data.columns for col in ['track_name', 'cluster_kmeans', 'track_artist']):
         print("Error: Data is missing required columns ('track_name', 'cluster_kmeans', 'track_artist').")
         return None # Return None for consistency

    print(f"\n--- Running recommend_songs_by_cluster_kmeans for: '{song_name}' ---")

    # Step 1: Search for the selected song using 'contains' (case-insensitive)
    # NOTE: This uses the simpler 'contains' logic ONLY, like the original notebook function.
    # It does NOT use re.escape() or the exact-match-first logic from find_similar_songs.
    print("Step 1: Searching for song using 'contains'...")
    selected_song = data[data['track_name'].str.contains(song_name, case=False, na=False)]

    # Step 2: If the song is not found, display a message and return None
    if selected_song.empty:
        print(f"   Song containing '{song_name}' not found.")
        return None # Return None as per original notebook function's behavior

    # Handle multiple matches implicitly: .values[0] takes the first match found by 'contains'
    if len(selected_song) > 1:
         print(f"   Warning: Found {len(selected_song)} songs containing '{song_name}'. Using the first: '{selected_song['track_name'].values[0]}'")
    else:
         print(f"   Found song: '{selected_song['track_name'].values[0]}'")


    # Step 3: Retrieve the cluster to which the selected song belongs
    # Using .values[0] as in the original notebook function
    try:
        cluster = selected_song['cluster_kmeans'].values[0]
        selected_track_name_val = selected_song['track_name'].values[0] # Get the name of the specific track found
        print(f"Step 2: Selected song '{selected_track_name_val}' belongs to cluster: {cluster}")
    except IndexError:
        print("   Error: Could not retrieve cluster or track name (IndexError). This might happen with empty intermediate data.")
        return None
    except KeyError as e:
         print(f"   Error: Missing column required for cluster/track name retrieval: {e}")
         return None


    # Step 4: Find all songs that belong to the same cluster as the selected song
    print(f"Step 3: Finding all songs in cluster {cluster}...")
    # Note: This includes the selected song initially
    recommended_songs = data[data['cluster_kmeans'] == cluster].copy() # Use copy to avoid warnings
    print(f"   Found {len(recommended_songs)} total songs in cluster {cluster} (including selected).")


    # Step 5: Exclude the specific selected song from the recommendations
    # Using .values[0] for the track name as in the original notebook function
    print(f"Step 4: Excluding the selected song '{selected_track_name_val}' from recommendations...")
    recommended_songs = recommended_songs[recommended_songs['track_name'] != selected_track_name_val]
    print(f"   Found {len(recommended_songs)} other songs in the cluster.")


    # Step 6: Return only the 'track_name' and 'track_artist', limited to the top 10
    if recommended_songs.empty:
        print("   No other similar songs found in the same cluster.")
        # Return empty DataFrame, consistent with returning DF on success
        return pd.DataFrame(columns=['track_name', 'track_artist'])

    print(f"Step 5: Selecting top {min(10, len(recommended_songs))} recommendations.")
    print("--- recommend_songs_by_cluster_kmeans : End ---")
    return recommended_songs[['track_name', 'track_artist']].head(10)

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
