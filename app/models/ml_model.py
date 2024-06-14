# Build a logistic regression classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define a function to predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
	features = [[sepal_length, sepal_width, petal_length, petal_width]]
	prediction = model.predict(features)
	return iris.target_names[prediction[0]]