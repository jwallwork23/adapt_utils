all: lint test

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,E226,E402,E731,E741,F403,F405,F999,N803,N806,W503 \
		--exclude=unsteady/test_cases/spaceship/nic*
	@echo "PASS"

test:
	@echo "Running test suite..."
	@pytest test
	@echo "PASS"

test_adapt:
	@echo "Running mesh adaptation test suite..."
	@pytest test/adapt
	@echo "PASS"

test_steady:
	@echo "Running tests for steady examples..."
	@pytest test/test_steady.py
	@echo "PASS"

test_unsteady:
	@echo "Running tests for unsteady examples..."
	@pytest test/test_unsteady.py
	@echo "PASS"

test_adjoint:
	@echo "Running tests for adjoint examples..."
	@pytest test/test_adjoint.py
	@echo "PASS"
