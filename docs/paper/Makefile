PDF_VIEWER := evince
MAIN_TEX := paper

.PHONY := all pdf clean
all: pdf

pdf:
	@latexmk -pdf -pvc $(MAIN_TEX) &

clean:
	@latexmk -c -C
	@rm -f $(MAIN_TEX).run.xml

