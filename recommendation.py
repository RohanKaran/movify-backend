import os
import time
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import dotenv_values

config = dotenv_values(".env")
movies_link = config.get("MOVIES_LINK")
crew_link = config.get("CREW_LINK")
ratings_link = config.get("RATINGS_LINK")


def recommend(movie):
    similarity_mat = pk.load(open('similarity_matrix.pkl', 'rb'))
    f = pk.load(open('final_df.pkl', 'rb'))
    movie = f[f.primaryTitle == movie]
    movie_index = movie.index[len(movie) - 1]
    distances = similarity_mat[movie_index]
    mlist = sorted(list(enumerate(distances)), reverse=True, key=lambda item: item[1])[1:21]

    mlist = dict(mlist)

    print(mlist)
    for key in mlist:
        mlist[key] *= f.iloc[key].popularity
    mlist = dict(sorted(mlist.items(), reverse=True, key=lambda item: item[1]))
    print(mlist)
    result = []
    for i in mlist:
        result.append(f.iloc[i].primaryTitle)
    return result


def preProcessing():
    start = time.time()
    movies = pd.read_csv(movies_link, sep='\t', dtype=str)
    movies['startYear'] = movies['startYear'].replace("\\N", '0')
    movies['startYear'] = movies['startYear'].astype(int)
    movies['runtimeMinutes'] = pd.to_numeric(movies.runtimeMinutes, errors="coerce")
    movies['runtimeMinutes'] = movies['runtimeMinutes'].replace(np.nan, 0)
    movies['runtimeMinutes'] = movies['runtimeMinutes'].astype(float).astype(int)
    movies_processed = movies[
        ((movies.titleType == 'movie') | (movies.titleType == 'tvSeries')) & (movies.genres != '\\N')]

    crews = pd.read_csv(crew_link, sep='\t', dtype=str)
    crews['directors'] = crews['directors'].replace("\\N", '')
    crews['writers'] = crews['writers'].replace("\\N", '')
    crews['crews'] = crews['writers'].str.split(",") + crews['directors'].str.split(",")
    crews = crews.drop(['writers', 'directors'], axis=1)
    final = pd.merge(movies_processed, crews, on='tconst')
    final['isAdult'] = final['isAdult'].replace('0', 'notadult')
    final['isAdult'] = final['isAdult'].replace('1', 'isadult')
    final['genres'] = final['genres'].str.split(',')

    ratings = pd.read_csv(ratings_link, sep='\t', dtype={'tconst': str, 'averageRating': float, 'numVotes': int})
    ratings['popularity'] = ratings['averageRating'] * ratings['numVotes']
    ratings = ratings[ratings.popularity >= ratings.popularity.mean()]
    ratings = ratings[ratings.popularity >= ratings.popularity.mean()]
    ratings_min = ratings['popularity'].min()
    ratings_max = ratings['popularity'].max()
    ratings['popularity'] = (ratings['popularity'] - ratings_min) / (ratings_max - ratings_min)
    ratings = ratings.drop(['averageRating', 'numVotes'], axis=1)
    final = pd.merge(final, ratings, on='tconst')
    final['titleType'] = final['titleType'].str.split(',')
    final['tags'] = final['titleType'] + final['titleType'] + final['isAdult'].str.split() + final['crews'] + final[
        'genres']
    print(final['tags'])
    f = final[['tconst', 'tags', 'primaryTitle', 'popularity']]
    f['tags'] = f['tags'].apply(lambda x: " ".join(x).lower())

    pk.dump(f, open('final_df.pkl', 'wb'))
    cv = CountVectorizer(token_pattern=r"\S+")
    vectors = cv.fit_transform(f['tags']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    pk.dump(similarity_matrix, open('similarity_matrix.pkl', 'wb'))
    end = time.time()
    print(end - start)
# print(f[f['primaryTitle'] == 'The Avengers'].index[2])
# print(f[f['primaryTitle'] == 'Avengers: Age of Ultron'])
