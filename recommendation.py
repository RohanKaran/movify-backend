import pickle
import bz2
from datetime import timedelta, datetime
from functools import lru_cache, wraps
from urllib.request import urlopen, urlretrieve
from kaggle import KaggleApi

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


@timed_lru_cache(24 * 3600)
def fetchDataFromKaggle():
    api = KaggleApi()
    api.authenticate()
    link = api.kernel_output(user_name='rohankaran', kernel_slug='movie-recommendation-system')
    print(link)
    f = pickle.load(urlopen(link['files'][0]['url']))
    # urlretrieve(link['files'][1]['url'], "similarity_matrix.pkl")
    # # sm = pickle.load(urlopen(link['files'][1]['url']))
    # sm = pickle.load(open('similarity_matrix.pkl', 'rb'))
    # start = time.time()
    # cv = CountVectorizer(max_features=7000)
    # vector = time.time()
    # print("Vectorized", vector - start)
    # vectors = cv.fit_transform(f['tags']).toarray()
    # sm = cosine_similarity(vectors)
    # with bz2.BZ2File('data/similarity_mat.pbz2', 'w') as file:
    #     pickle.dump(sm, file)
    sm = pickle.load(bz2.BZ2File('data/similarity_mat.pbz2', 'rb'))
    return f, sm


def recommend(movie):
    f, similarity_mat = fetchDataFromKaggle()

    movie = f[f.primaryTitle == movie]
    movie_index = movie.index[len(movie) - 1]
    distances = similarity_mat[movie_index]
    mlist = sorted(list(enumerate(distances)), reverse=True, key=lambda item: item[1])[1:21]

    mlist = dict(mlist)
    for key in mlist:
        mlist[key] *= f.iloc[key].popularity
    mlist = dict(sorted(mlist.items(), reverse=True, key=lambda item: item[1]))

    result = []
    for i in mlist:
        result.append({f.iloc[i].tconst: f.iloc[i].primaryTitle})
    return result
