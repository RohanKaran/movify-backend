from fastapi import FastAPI
from recommendation import recommend, fetchDataFromKaggle


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/daily_update")
async def daily_update(s: str):
    if s.strip() == 'update':
        fetchDataFromKaggle()
        return "Success!"
    return "Error!"


@app.post("/recommendation")
async def recommendation(movie: str):
    return recommend(movie.strip())
