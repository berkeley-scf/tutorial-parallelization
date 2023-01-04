parallel-R.md: parallel-R.Rmd
	Rscript -e "rmarkdown::render('parallel-R.Rmd', rmarkdown::md_document(preserve_yaml = TRUE, variant = 'gfm', pandoc_args = '--markdown-headings=atx'))"  ## atx headers ensures headers are all like #, ##, etc. Shouldn't be necessary as of pandoc >= 2.11.2
## 'gfm' ensures that the 'r' tag is put on chunks, so code coloring/highlighting will be done when html is produced.

parallel-julia.md: parallel-julia.qmd
	quarto render parallel-julia.qmd --to md
## To render Julia, need to specify Jupyter kernel. Also need to have IJulia installed for the user doing the rendering.
## Currently using a kernel that hard-codes in 4 threads (`-t 4`) to do parallel stuff.
## Rendering occasionally/non-reproducibly fails with inscrutable error messages.


clean:
	rm -f parallel-R.md 
