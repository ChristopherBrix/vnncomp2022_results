
MAIN=main

all $(MAIN).pdf : $(MAIN).tex 
	pdflatex -halt-on-error $<
	-bibtex $(basename $<)
	pdflatex -halt-on-error $<
	pdflatex -halt-on-error $<

$(MAIN).fmt : prelude1.tex
	pdflatex -ini -jobname="$(MAIN)" "&pdflatex prelude1.tex\dump"

STEM=$(notdir $(PWD))

snapshot : $(MAIN).pdf
	@TODAY="`date +%Y%m%d`"; \
	N="1"; \
	while [ -f $$TODAY-$(STEM)-$$N.pdf ]; do \
	  N="`expr $$N + 1`"; done; \
	echo $(MAIN).pdf '->' $$TODAY-$(STEM)-$$N.pdf; \
	cp $(MAIN).pdf $$TODAY-$(STEM)-$$N.pdf

%.pdf : %.gpl
	gnuplot $(<)

plots : $(subst gpl,pdf,$(wildcard *.gpl))

clean :
	rm -f $(addprefix $(MAIN)., aux bbl blg fmt log out pdf)

.PHONY : plots snapshot clean

