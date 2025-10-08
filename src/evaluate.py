import joblib
import numpy as np

def evaluate_model():
    model = joblib.load("models/model.pkl")
    test_x = np.array([[6],[7],[8]])
    preds = model.predict(test_x)
    return preds

if __name__ == "_main__":
    evaluate_model()