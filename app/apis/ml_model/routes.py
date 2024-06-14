from fastapi import APIRouter
from app.models import predict_species
from app.schemas import IrisData

router = APIRouter()

@router.post("/predict/")
async def predict_species_api(iris_data: IrisData):
	species = predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)
	return {"species": species}