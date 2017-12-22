.PHONY: clean train prediction submission

clean: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

train: clean
	python src/train.py

prediction: train
	python src/predict.py

submission: prediction
	python src/submit.py
