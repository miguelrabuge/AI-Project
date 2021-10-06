PDF_VIEWER := evince
MAIN_TEX := proposal

.PHONY := pdf read clean

pdf:
	@echo "Compiling..."
	@pdflatex ${MAIN_TEX}.tex 
	@biber ${MAIN_TEX}.bcf 
	@pdflatex ${MAIN_TEX}.tex 

read: 
	${PDF_VIEWER} ${MAIN_TEX}.pdf &

all-clean: clean
	@rm -f ${MAIN_TEX}.pdf

clean:
	@echo "Cleaning..."
	@rm -f !(*.{tex,bib,pdf})
