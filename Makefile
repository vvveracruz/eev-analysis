run:
	pipenv run python src/main.py

analyse:
	pipenv run python src/analysis.py

format:
	pipenv run black src
	pipenv run isort src