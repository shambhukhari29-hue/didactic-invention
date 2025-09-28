# Iris Flower Classification - My First ML Project
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = load_iris()
print("Flower Features:", iris.feature_names)
print("Flower Types:", iris.target_names)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.1%}")

# Make prediction
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
print(f"Predicted flower type: {iris.target_names[prediction[0]]}")

# Additional: Feature importance
print(f"\nFeature Importance: {model.feature_importances_}")
