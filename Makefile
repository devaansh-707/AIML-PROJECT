PYTHON ?= python

.PHONY: setup train infer tune explain test app format lint

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	pre-commit install

train:
	$(PYTHON) train_multiclass.py

infer:
	$(PYTHON) predict_multiclass.py

tune:
	$(PYTHON) -m src.tuning.optuna_search

explain:
	$(PYTHON) -m src.eval.explain --data data/test.csv --output outputs --meta outputs/meta.json

test:
	pytest

app:
	streamlit run app.py

format:
	black .
	isort .

lint:
	flake8 .

