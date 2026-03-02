import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.explain_service import ExplainService

client = TestClient(app)

@pytest.fixture
def mock_xgb_model():
    from xgboost import XGBClassifier
    # Modelo pequeno treinado
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = np.random.randint(0, 2, 100)
    
    model = XGBClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y)
    
    return {
        "model": model,
        "features": list(X.columns),
        "version": "1.0.0"
    }

@pytest.fixture
def mock_svm_model():
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = np.random.randint(0, 2, 100)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42))
    ])
    model.fit(X, y)
    
    return {
        "model": model,
        "features": list(X.columns),
        "version": "1.0.0"
    }

@pytest.fixture
def mock_logistic_model():
    from sklearn.linear_model import LogisticRegression
    
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return {
        "model": model,
        "features": list(X.columns),
        "version": "1.0.0"
    }

@patch('app.routers.explain.load_model')
@patch('app.routers.explain.load_training_data')
def test_shap_endpoint_success(mock_load_data, mock_load_model, mock_xgb_model):
    mock_load_model.return_value = mock_xgb_model
    # Retorna DataFrame X e Array Y dummy
    mock_load_data.return_value = (
        pd.DataFrame(np.random.rand(50, 5), columns=[f"feat_{i}" for i in range(5)]),
        np.zeros(50)
    )
    
    response = client.post("/explain/shap/1.0.0")
    
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "shap_summary"
    assert "image_base64" in data
    assert len(data["image_base64"]) > 100 # base64 não vazio

@patch('app.routers.explain.load_model')
@patch('app.routers.explain.load_training_data')
def test_svm_boundary_endpoint_success(mock_load_data, mock_load_model, mock_svm_model):
    mock_load_model.return_value = mock_svm_model
    
    y_mock = np.zeros(50)
    y_mock[:25] = 1
    
    mock_load_data.return_value = (
        pd.DataFrame(np.random.rand(50, 5), columns=[f"feat_{i}" for i in range(5)]),
        y_mock
    )
    
    response = client.post("/explain/boundary/1.0.0")
    
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "decision_boundary"
    assert "image_base64" in data

@patch('app.routers.explain.load_model')
def test_linear_coefficients_endpoint_success(mock_load_model, mock_logistic_model):
    mock_load_model.return_value = mock_logistic_model
    
    response = client.get("/explain/coefficients/1.0.0")
    
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "linear_coefficients"
    assert "image_base64" in data

@patch('app.routers.explain.load_model')
def test_invalid_model_type_for_shap(mock_load_model, mock_logistic_model):
    # Passando regressão logística para rota SHAP deve falhar
    mock_load_model.return_value = mock_logistic_model
    
    response = client.post("/explain/shap/1.0.0")
    assert response.status_code == 400
    assert "específico para modelos de árvore" in response.json()["detail"]
