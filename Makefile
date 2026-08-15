PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest

format:
	$(PYTHON) -m black .

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/smoke_api.py

run-api:
	$(PYTHON) app/run.py --reload

run-cli:
	$(PYTHON) -m app.cli --bedrooms 3 --bathrooms 2.0 --sqft_living 1910 --sqft_lot 7600 --floors 1.5 --sqft_above 1560 --yr_built 1975 --zipcode 98065 --lat 47.5724 --long -122.2300 --sqft_living15 1840 --sqft_lot15 7620

.PHONY: install format test smoke run-api run-cli
