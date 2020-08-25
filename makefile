all: lint

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,F403,F405,E226,E402,E721,E731,E741,W503,N803,N806,F999
	@echo "PASSED"
