from fastapi import FastAPI
from api.models.sentence import PredictRequest, PredictResponse
from inference import load_sentiment_model


app = FastAPI()
model = load_sentiment_model()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    result = model.predict(request.text)
    return PredictResponse(prediction=result)
