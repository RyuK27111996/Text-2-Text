install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2

test:
	pytest -q
