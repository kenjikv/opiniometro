# One image for training and serving. Using the same image guarantees that
# the preprocessing code and library versions are identical in both places.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is cached while only the code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY params.yaml ./
COPY data ./data
COPY src ./src
COPY api ./api

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Development image: tests and notebook on top of the runtime image.
FROM runtime AS dev

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY tests ./tests
COPY notebooks ./notebooks

CMD ["pytest", "-q"]
