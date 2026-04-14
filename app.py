from fastapi import FastAPI
from api.models.sentence import PredictRequest, PredictResponse


app = FastAPI()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    return PredictResponse(prediction="positive")
