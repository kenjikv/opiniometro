# Shortcuts for macOS and Linux. On Windows run the docker compose commands directly.

up:            ## build and start everything (trains the first time)
	docker compose up --build -d
	@echo "Frontend: http://localhost:$${FRONTEND_PORT:-8080}   API: http://localhost:$${API_PORT:-8000}/api/docs"

train:         ## retrain with the current params.yaml and restart the API
	docker compose run --rm trainer python -m src.pipeline
	docker compose restart api

reload-data:   ## reload data/raw/resenas.csv into PostgreSQL and retrain
	docker compose run --rm trainer python -m src.pipeline --reload-data
	docker compose restart api

test:          ## run the automated tests
	docker compose --profile dev run --rm --build tests

frontend:      ## rebuild the frontend after changing frontend/
	docker compose up -d --build frontend

notebook:      ## JupyterLab on http://localhost:8888
	docker compose --profile dev up --build notebook

logs:
	docker compose logs -f api trainer

down:          ## stop the containers (keeps the database)
	docker compose down

clean:         ## stop and delete the database volume and the artifacts
	docker compose down -v
	find artifacts -type f ! -name .gitkeep -delete

.PHONY: up train reload-data test frontend notebook logs down clean
