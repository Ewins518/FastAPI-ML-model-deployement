from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle

app = FastAPI()

class FuelFeatures(BaseModel):
    displ: int
    hp: int
    year: int
    origin: float

model = pickle.load(open('model.pkl', 'rb'))

@app.get("/")
def home():
    return {'ML model for Fuel consumption prediction'}

@app.post('/make_predictions')
async def make_predictions(features: FuelFeatures):
    
    predicted_fuel = model.predict([[features.displ,features.hp,features.year,features.origin]])[0]
    return({"Fuel consumption prediction": predicted_fuel})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
