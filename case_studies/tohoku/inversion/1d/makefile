all: discrete continuous

.phony = discrete continuous

discrete:
	@python3 discrete.py 0
	@python3 discrete.py 1
	@python3 discrete.py 2

plot_discrete:
	@python3 plot_progress.py 0 discrete
	@python3 plot_progress.py 1 discrete
	@python3 plot_progress.py 2 discrete

continuous:
	@python3 continuous.py 0
	@python3 continuous.py 1
	@python3 continuous.py 2

plot_continuous:
	@python3 plot_progress.py 0 continuous
	@python3 plot_progress.py 1 continuous
	@python3 plot_progress.py 2 continuous

gradient:
	@python3 gradient_discrete.py 0
	@python3 gradient_continuous.py 0
	@python3 gradient_discrete.py 1
	@python3 gradient_continuous.py 1
	@python3 gradient_discrete.py 2
	@python3 gradient_continuous.py 2
