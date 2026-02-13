# Movie Recommendation System

## Overview
A collaborative filtering recommendation system built on the MovieLens dataset. It suggests movies to a user based on the preferences of similar users.

## Dataset
- **Source:** MovieLens Small Dataset
- **Files:** `movies.csv`, `ratings.csv`

## Methodology
1. **User-Based Collaborative Filtering:**
   - Constructs a User-Item Matrix.
   - Calculates **Cosine Similarity** between users.
   - Recommends movies liked by similar users that the target user hasn't seen.
2. **Matrix Factorization (SVD) [Bonus]:**
   - Uses Singular Value Decomposition to identify latent features in the user-movie matrix.
   - Provides item-based recommendations (e.g., "Users who liked Star Wars also liked...").

## Technologies Used
- Python
- Scikit-Learn (Cosine Similarity, TruncatedSVD)
- Pandas & NumPy

## Key Features
- Generates top $N$ movie recommendations for any specific User ID.
- Handles sparse data using matrix operations.
