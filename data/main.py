from fastapi import FastAPI
from recommendation import recommend, fetchDataFromKaggle


app = FastAPI()


@app.on_event("startup")
def startup_event():
    fetchDataFromKaggle()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/recommendation")
async def recommendation(movie: str):
    return recommend(movie.strip())
