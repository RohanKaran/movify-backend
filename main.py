from fastapi import FastAPI
from recommendation import recommend, preProcessing

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/daily_update")
async def daily_update(s: str):
    if s.strip() == 'update':
        preProcessing()
        return "Success!"
    return "Error!"


@app.post("/recommendation")
async def recommendation(movie: str):
    return recommend(movie.strip())
