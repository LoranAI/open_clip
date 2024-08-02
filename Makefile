install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-training:
	python -m pip install -r requirements-training.txt

install-test: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

test: ## [Local development] Run unit tests
	python -m pytest -x -s -v tests

install-all: ## [Local development] Install all requirements
	python -m pip install -U pip
	python -m pip install -e .
	python -m pip install -r requirements-training.txt
	python -m pip install -r requirements-test.txt
	python -m pip install wandb
	python -m pip install clip
	python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

