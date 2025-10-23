from pydantic import BaseModel
from typing import List

class Review(BaseModel):
    review: str

class PredictRequest(BaseModel):
    reviews: List[Review]