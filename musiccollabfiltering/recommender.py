import implicit
import pandas as pd
import numpy as np
import scipy
from pathlib import Path

from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

def load_user_artists(file_path: Path):
    """Load the user-artist interactions data, return as a CSR matrix.

    Args:
        file_path (Path): The path to the TSV file containing user-artist interaction data.

    Returns:
        csr_matrix: A compressed sparse row matrix. Rows correspond to users, 
        columns correspond to artists, and values represent the interaction weights
        between users and artists.
    """
    # Load the data and set index using userID and artistID
    user_artists = pd.read_csv(file_path, sep='\t')
    user_artists.set_index(['userID', 'artistID'], inplace=True)
    
    # Extract data for the CSR matrix
    data = user_artists['weight']
    rows = user_artists.index.get_level_values('userID')
    cols = user_artists.index.get_level_values('artistID')
    
    # Create and return CSR Matrix
    matrix = csr_matrix((data, (rows, cols))).tocsr()
    return matrix

class ArtistRetriever:
    """Reads in a CSV file, retrieves artist information using the artistID or artist name.
        
    Attributes:
        artists (pd.DataFrame): A DataFrame containing artist information.
    """
    def __init__(self, artist_file_path):
        self.artists = pd.read_csv(artist_file_path, sep='\t')
        
    def get_artist_name(self, artist_id):
        """Retrieve the name of an artist given their ID.

        Args:
            artist_id (int): The ID of the artist.

        Returns:
            str: Name of the artist if found, otherwise "Unknown Artist".
        """
        artist_row = self.artists[self.artists['id'] == artist_id]
        return artist_row['name'].values[0] if not artist_row.empty else "Unknown Artist"
    
    def get_artist_id(self, artist_name):
        """Retrieve the ID of an artist given their name.

        Args:
            artist_name (str): Name of an artist.

        Returns:
            int: The ID of the artist if found, otherwise "Unknown Artist".
        """
        artist_row = self.artists[self.artists['name'].str.lower() == artist_name.lower()]
        return artist_row['id'].values[0] if not artist_row.empty else "Unknown Artist"
        
class Recommender:
    """Recommender system for suggesting artists based on user preferences.

    This class provides methods to fit a collaborative filtering model to a user-artist interaction matrix,
    recommend new artists to users, and find similar artists based on a given artistID.

    Attributes:
        model: The collaborative filtering model used for recommendations.
        artist_retriever (ArtistRetriever): An instance of the ArtistRetriever class for retrieving artist names.
    """
    def __init__(self, model, artist_retriever):
        self.model = model
        self.artist_retriever = artist_retriever
    
    def fit(self, user_artist_matrix):
        """Fit a collaborative filtering model to a user-artist interaction matrix.

        Args:
            user_artist_matrix (csr_matrix):  A sparse matrix representing the user-artist interactions.
        """
        # Weight the matrix to reduce impact of superfan users and weight given to popular items
        artist_user_plays = bm25_weight(user_artist_matrix, K1=100, B=0.8)
        
        # Change matrix from (item, user) to (user, item) using transpose
        self.model.fit(artist_user_plays.T.tocsr()) 
        
    def recommend_artist(self, user_id, user_artist_matrix, num_recs=10):
        """Recommend artists based on user's past listening habits.

        Args:
            user_id (int): The ID of the user.
            user_artist_matrix (csr_matrix): A sparse matrix representing the user-artist interactions.
            num_recs (int, optional): Number of recommendations to return. Defaults to 10.

        Returns:
            tuple: A tuple containing two lists:
                - artist_names (list of str): Names of the recommended artists.
                - scores (list of float): Scores corresponding to the relevance of each artist.
        """
        recs = self.model.recommend(user_id, user_artist_matrix[user_id], N= num_recs, filter_already_liked_items=True)
        artists, scores = recs
        artist_names = [self.artist_retriever.get_artist_name(aid) for aid in artists]
        return artist_names, scores
    
    def recommend_similar(self, artist_id, num_recs=10):
        """Recommend artists similar to a given artist.

        Args:
            artist_id (int): The ID of the artist to find similar artists for.
            num_recs (int, optional): Number of similar artists to return. Defaults to 10.

        Returns:
            tuple: A tuple containing two lists:
                - ids (list of int): IDs of similar artists.
                - scores (list of float): Similarity scores corresponding to each artist.
        """
        ids, scores = self.model.similar_items(artist_id, N=num_recs)
        return ids, scores
        
if __name__ == "__main__":
    # Load user-artist interaction matrix
    artists, users, artist_user_plays = get_lastfm()
    userartist_matrix = load_user_artists(Path(__file__).parent.parent / 'lastFMdataset' / 'user_artists.dat')
    
    # Initiate artist retriever
    artists_file = Path(__file__).parent.parent / 'lastFMdataset' / 'artists.dat'
    artist_retriever = ArtistRetriever(artists_file)
    
    # Instantiate and train the model
    model = AlternatingLeastSquares(factors=64, iterations=10, regularization=0.5)
    recommender = Recommender(model, artist_retriever)
    recommender.fit(userartist_matrix)

    while True:
        choice = int(input("\nEnter 1 for artist recommendations by userID, 2 for similar artist recommendations by artistID, or 3 to generate an artistID based on the artist's name. Enter 4 to exit:\n"))
        if choice == 1:
            # Get recommendations for a user
            user_id_input = int(input("Enter a userID:   "))
            recommended_artists, scores = recommender.recommend_artist(user_id_input, userartist_matrix, num_recs=10)
            for artist, score in zip(recommended_artists, scores):
                print(f"Artist: {artist}, Score: {score}")
        elif choice == 2:
            # Get recommendations for similar artists based on artistID
            artist_id_input = int(input("Enter an artistID:   "))
            print("Artist: ", artist_retriever.get_artist_name(artist_id_input))
            recommended_artists, scores = recommender.recommend_similar(artist_id_input)
            for ids, score in zip(recommended_artists, scores):
                print(f"Artist: {artists[ids]}, Score: {score}")
        elif choice == 3:
            # Generate artistID based on artist name
            artist_name_input = str(input("Enter an artist's name:   "))
            print(artist_retriever.get_artist_id(artist_name_input))
        elif choice == 4:
            print("\nThank you for using the music recommender!")
            break
        else:
            print("\nThat is not an option.\n")