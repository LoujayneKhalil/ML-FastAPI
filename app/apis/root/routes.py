from fastapi import APIRouter

router = APIRouter()

# Define a function to return a description of the app
def get_app_description():
	return (
    	"Welcome to the Iris Species Prediction API!"
    	"This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
	)

# Define the root endpoint to return the app description
@router.get("/")
def root():
	return {"message": get_app_description()}