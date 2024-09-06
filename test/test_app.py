from app import app 

def test_app():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert response.data != None

def test_predict():
    response = app.test_client().post('/predict')
    assert response.status_code == 200
    assert response.data != None