# artist-recommender
A Python-based music recommendation system that suggests artists to users based on collaborative filtering techniques. It utilizes the Last.fm dataset and Python's implicit library for implementing the Alternating Least Squares (ALS) model and other collaborative filtering methods.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Features
- User-Based recommendations: Recommends artists to users based on their past listening habits
- Artist Similarity: Recommends similar artists based on a given artist ID or name
- Artist Information Retrieval: Retrieves artist names based on artist ID or vice versa

## Installation
To set up this project, follow these steps:
1. Clone the repository:
    git clone https://github.com/your-username/music-recommender.git
    cd music-recommender
2. Set up a virtual environment:
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Download the Last.fm dataset:
    Place the dataset files (user_artists.dat, artists.dat) in a directory named lastFMdataset within the project directory.

## Usage
To run the recommendation system, type in the command:
  python musiccollabfiltering/recommender.py
You will be prompted with the following options:

## Code Overview
### Main Components
load_user_artists: Loads user-artist interactions data from a file and returns it as a CSR matrix.
ArtistRetriever: Class for retrieving artist names and images using artist IDs.
Recommender: Class for fitting a collaborative filtering model and generating recommendations.

### Key Methods
fit: Fits the collaborative filtering model to the user-artist interaction matrix.
recommend_artist: Recommends artists to a user based on their listening history.
recommend_similar: Finds artists similar to a given artist.

## Technologies Used
This project is programmed in Python.
The dataset used in this project is sourced from Last.fm, including user-artist interactions and artist metadata.
Python's Implicit Library is used for the implementation of the ALS model.

## Contact
Yevgenia Minchuk

Email: minchuky@msu.edu
