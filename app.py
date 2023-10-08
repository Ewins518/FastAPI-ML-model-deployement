from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import uvicorn
import pandas as pd
import pickle
from starlette.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class FuelFeatures(BaseModel):
    displ: int
    hp: int
    year: int
    origin: float

model = pickle.load(open('model.pkl', 'rb'))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@app.post('/make_predictions', response_class=HTMLResponse)
async def make_predictions(request: Request, displ: int = Form(...),
                            hp: int = Form(...),
                            year: float = Form(...),
                            origin: float = Form(...)):
    features = FuelFeatures(
        displ=displ,
        hp=hp,
        year=year,
        origin=origin,
    
    )
    features_dict = features.dict().values()
    
    features_df = pd.DataFrame([features_dict])
    
    predicted_fuel = model.predict(features_df)
    return templates.TemplateResponse('index.html',{"request": request, "prediction_text":'Fuel consumption prediction is : {} MPG'.format(round(float(predicted_fuel[0]),2))})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
