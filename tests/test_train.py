import os
import joblib
import src.train import train_model


def test_model_trining():
    train_model()
    assert os.path.exists("models/model.pkl"),"Model file not found"
    model = joblib.load("models/model.pkl")
    assert hasattr(model,"predict"),"Model does not found"

def test_placeholder():
    assert True
