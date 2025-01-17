### Instructions
1. Download and install [Latex](https://www.tug.org/mactex/mactex-download.html). 
2. Update the latex package manager *tlmgr*
    ```bash
    wget https://mirror.ctan.org/systems/texlive/tlnet/update-tlmgr-latest.sh
    chmod +x update-tlmgr-latest.sh
    ./update-tlmgr-latest.s
    ```
3. If you don't have some basic packages, it won't work. You can install these packages with the package manager
    ```bash
    tlmgr install amsmath cancel listings multirow setspace
    ```
    I also installed some additional package collections 
    ```bash
    tlmgr install collection-latexrecommended
    tlmgr install collection-mathscience
    tlmgr install collection-latexextra
    ```
    If you get errors about specific style pages e.g. *algorithm.sty*, search for it, then install it.
    ```bash
    tlmgr search --file algorithm.sty
    algorithms:
            texmf-dist/tex/latex/algorithms/algorithm.sty
    tlmgr install algorithms
    ```
4. To use a package in your `.tex` file, include the package name like an import statement at the top 
e.g.`\usepackage{algorithms,amsmath}`. One or more packages can be imported simultaneously.
5. Convert the file to pdf using `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw1.tex`.
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw2.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw3.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw4.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw5.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw6.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw7.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw8.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw9.tex`
    - `pdflatex -halt-on-error -interaction=nonstopmode -file-line-error hw10.tex`
6. Install the `vlfeat` matlab toolbox manually from https://www.vlfeat.org/install-matlab.html for `hw9`.
7. When you have a `.bib` reference file, you have to run bibtex after running pdflatex to proccess the bib file
    - `bibtex hw10`, where hw10 is the name of the tex file
    - run `pdflatex hw10.tex` again after running bibtext to create the final output file
    - If you get this type of error from pdflatex
        ```bash
        ! LaTeX3 Error: Mismatched LaTeX support files detected.
        (LaTeX3)        Loading 'expl3.sty' aborted!
        (LaTeX3)        
        (LaTeX3)        The L3 programming layer in the LaTeX format
        (LaTeX3)        is dated 2024-08-16, but in your TeX tree the files require
        (LaTeX3)        at least 2024-09-10.
        ```
        It can be fixed by updating ***tinytex*** in *R* by run running `>tinytex::tlmgr_update()` in an R terminal window