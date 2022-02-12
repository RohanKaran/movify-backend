import pickle
from datetime import timedelta, datetime
from functools import lru_cache, wraps
from urllib.request import urlopen
from kaggle import KaggleApi


def timed_lru_cache(seconds: int, maxsize: int = 512):
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
    sm = pickle.load(urlopen(link['files'][1]['url']))
    # sm = pickle.load(open('data/similarity_matrix.pkl', 'rb'))
    print("hi")
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
