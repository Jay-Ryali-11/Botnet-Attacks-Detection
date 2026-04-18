import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app('testing')
    with app.test_client() as c:
        yield c


def test_login_page_loads(client):
    response = client.get('/login')
    assert response.status_code == 200


def test_registration_page_loads(client):
    response = client.get('/registration')
    assert response.status_code == 200


def test_login_wrong_password(client):
    response = client.post('/login', data={
        'email': 'notexist@example.com',
        'password': 'wrongpassword'
    }, follow_redirects=True)
    assert response.status_code == 200


def test_index_loads(client):
    response = client.get('/')
    assert response.status_code == 200