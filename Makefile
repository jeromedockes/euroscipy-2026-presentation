all: pres notebooks

pres:
	quarto render skrub-dataops.qmd

preview: notebooks
	quarto preview skrub-dataops.qmd &
	cd scripts && SKB_TABLE_REPORT_N_ROWS=4 SKB_DATA_OPS_OPEN_GRAPH_DROPDOWN=True jupyter notebook

notebooks:
	rm -f scripts/intro/optuna.db
	find scripts -name '*.py' -type f -print0 | xargs -0 jupytext --to notebook

