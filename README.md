# Artist Recommender
A Python-based music recommendation system that suggests artists to users based on collaborative filtering techniques. It utilizes the Last.fm dataset and Python's implicit library for implementing the Alternating Least Squares (ALS) model and other collaborative filtering methods.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Code Overview](#code-overview)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Features
- User-Based recommendations: Recommends artists to users based on their past listening habits
- Artist Similarity: Recommends similar artists based on a given artist ID or name
- Artist Information Retrieval: Retrieves artist names based on artist ID or vice versa

## Installation
To set up this project through downloading the ZIP file, follow these steps:

1. Download the ZIP file from GitHub
   
2. Create the Python environment:

- Open the Command Palette (View > Command Palette or Ctrl+Shift+P)
- Select the Python: Create Environment command to create a virtual environment in your workspace. Choose venv and the Python environment you want to use
- After the virtual environment is created, run Terminal: Create New Terminal (Ctrl+Shift+ ``) from the Command Palette, which creates a terminal and automatically activates the virtual environment
- Type these commands into the command line:

    python -m venv venv
   
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
3. Download the dependencies:
   
    pip install -r requirements.txt
    
### Usage
To run the recommendation system, type in the command:

    python musiccollabfiltering/recommender.py
  
You will be prompted with the following options:

Enter 1 for artist recommendations by userID, 2 for similar artist recommendations by artistID, or 3 to generate an artistID based on the artist's name. Enter 4 to exit:

### Example:

   Enter 1 for artist recommendations by userID, 2 for similar artist recommendations by artistID, or 3 to generate an artistID based on the artist's name. Enter 4 to exit:
   
   3
   
   Enter an artist's name:   coldplay
   
   The artist ID for coldplay is 65
   
   Enter 1 for artist recommendations by userID, 2 for similar artist recommendations by artistID, or 3 to generate an artistID based on the artist's name. Enter 4 to exit:
   
   2
   
   Enter an artistID:   65
   
   Your Artist:  Coldplay
   
   Artist: #1, Score: 1.0000001192092896
   
   Artist: 1. allegro, Score: 0.48792362213134766
   
   Artist: 194_dj piligrim, Score: 0.41729864478111267
   
   Artist: 13 & god, Score: 0.4159063696861267
   
   Artist: (f-zero gx) daiki kasho, Score: 0.393725723028183
   
   Artist: 1.serija, Score: 0.375570684671402
   
   Artist: 13 nelly furtado y juanes, Score: 0.36356422305107117
   
   Artist: 100 portraits & waterdeep, Score: 0.36119163036346436
   
   Artist: 02 welcome to rapture, Score: 0.3569047749042511
   
   Artist: 08 (zero eight), Score: 0.34334686398506165

   
   Enter 1 for artist recommendations by userID, 2 for similar artist recommendations by artistID, or 3 to generate an artistID based on the artist's name. Enter 4 to exit:
   
   4
   
   Thank you for using the music recommender!

## Code Overview
### Main Components
load_user_artists: Loads user-artist interaction data from a file and returns it as a CSR matrix.

ArtistRetriever: Class for retrieving artist information using the artistID or artist name.

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
