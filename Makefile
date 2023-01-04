parallel-R.md: parallel-R.Rmd
	Rscript -e "rmarkdown::render('parallel-R.Rmd', rmarkdown::md_document(preserve_yaml = TRUE, variant = 'gfm', pandoc_args = '--markdown-headings=atx'))"  ## atx headers ensures headers are all like #, ##, etc. Shouldn't be necessary as of pandoc >= 2.11.2
## 'gfm' ensures that the 'r' tag is put on chunks, so code coloring/highlighting will be done when html is produced.

parallel-julia.md: parallel-julia.qmd
	quarto render parallel-julia.qmd --to markdown
## To render Julia, need to specify Jupyter kernel. Also need to have IJulia installed for the user doing the rendering.
## May be able to create a kernel that hard-codes in 4 threads to do parallel stuff.

clean:
	rm -f parallel-R.md 
