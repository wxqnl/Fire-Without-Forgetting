# make        -> main.pdf (paper only, for submission)
# make full   -> main_full.pdf (paper + supplementary, cross-references resolved)
# make data   -> refresh figures and tables from ../runs
PY ?= /home/user01/miniconda3/envs/fwf/bin/python

main.pdf: main.tex preamble.tex refs.bib sec/*.tex tables/*.tex fig/*
	rm -f withsupp.flag
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

full:
	touch withsupp.flag
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=main_full main.tex
	rm -f withsupp.flag

data:
	$(PY) scripts/make_figures.py
	$(PY) scripts/paper_tables.py

clean:
	latexmk -C main.tex; latexmk -C -jobname=main_full main.tex; rm -f withsupp.flag

.PHONY: full data clean
