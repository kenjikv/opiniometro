import pytest

from api.schemas import MAX_CHARS
from src.config import LABELS


def test_health(client):
    body = client.get("/api/health").json()
    assert body["estado"] == "ok" and body["modelo_cargado"] is True
    assert body["base_de_datos"] == "no configurada"


def test_predict_returns_label_probabilities_and_steps(client):
    response = client.post("/api/predict", json={"texto": "El majadito no estaba rico 😡"})
    assert response.status_code == 200
    body = response.json()
    assert body["sentimiento"] in LABELS
    assert set(body["probabilidades"]) == set(LABELS)
    assert abs(sum(body["probabilidades"].values()) - 1) < 1e-3
    assert body["probabilidad"] == max(body["probabilidades"].values())
    steps = body["preprocesamiento"]
    assert "no" in steps["final"] and "emojinegativo" in steps["tokens"]
    assert "el" in steps["stopwords_eliminadas"]


@pytest.mark.parametrize("payload, message", [
    ({"texto": ""}, "El texto está vacío."),
    ({"texto": "     "}, "El texto está vacío."),
    ({"texto": "123 456 !!!"}, "El texto no contiene palabras."),
    ({"texto": "a" * (MAX_CHARS + 1)}, f"el máximo es {MAX_CHARS}"),
    ({"texto": 5}, "Debe ser un texto"),
    ({}, "Falta este campo"),
    ({"texto": "rico", "estrellas": 5}, "Campo no permitido"),
])
def test_invalid_input_gets_a_clear_message(client, payload, message):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert message in response.json()["error"]


def test_malformed_json(client):
    response = client.post("/api/predict", content="{texto: rico",
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "JSON válido" in response.json()["error"]


def test_batch_from_txt_skips_invalid_lines(client):
    content = "Riquísimo todo\n\nPésima atención\n12345\n".encode()
    response = client.post("/api/predict/batch",
                           files={"archivo": ("resenas.txt", content, "text/plain")})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["omitidas"] == [{"fila": 3, "motivo": "El texto no contiene palabras."}]
    assert sum(body["resumen"].values()) == 2


def test_batch_from_csv(client):
    content = "id,texto\n1,Muy rico el silpancho\n2,Nunca más vuelvo\n".encode()
    response = client.post("/api/predict/batch",
                           files={"archivo": ("resenas.csv", content, "text/csv")})
    assert response.status_code == 200
    assert [r["fila"] for r in response.json()["resultados"]] == [1, 2]


@pytest.mark.parametrize("filename, content, message", [
    ("resenas.csv", b"comentario\nrico\n", "columna llamada 'texto'"),
    ("resenas.pdf", b"rico", "Formato no soportado"),
    ("resenas.txt", "rico".encode("utf-16"), "UTF-8"),
])
def test_batch_rejects_bad_files(client, filename, content, message):
    response = client.post("/api/predict/batch", files={"archivo": (filename, content)})
    assert response.status_code == 422
    assert message in response.json()["error"]


def test_metrics(client, trained):
    body = client.get("/api/metrics").json()
    assert body["version"] == trained["version"]
    assert {m["name"] for m in body["models"]} == {"naive_bayes", "logistic_regression",
                                                   "baseline"}


def test_history_needs_a_database(client):
    response = client.get("/api/history")
    assert response.status_code == 503
    assert "base de datos" in response.json()["error"]
