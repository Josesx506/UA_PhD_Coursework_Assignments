### Assignments for STAT 675
1. Install R and R studio <br><br>
2. Install IRKernel to enable R to run in Jupyter notebooks
    ```r
    install.packages('IRkernel')
    IRkernel::installspec()  # to register the kernel in the current R installation
    ```
    - Install *"R Extension for Visual Studio Code"* by REditorSupport
    <br><br>
3. Install supporting packages to enable Rmarkdown (rmd) knitting in vscode
    - Open Rstudio terminal and get the path to `pandoc` with `echo $RSTUDIO_PANDOC`.
    - Change vs code settings under `Settings > R > Rmarkdown > Knit:`. Change the settings block
        ```bash
        rmarkdown::render          # Old
        Sys.setenv(RSTUDIO_PANDOC="--path to pandoc"); rmarkdown::render # New
        ```
    - This allows it to convert rmd files to `html_notebooks`. 
    - To support conversion to `pdf`, install a LaTeX distribution for your platform.
        - I installed the MacTex package with *brew* that allows *"pdflatex"* command to work - `brew install --cask mactex-no-gui`.
        - Mactex worked in R but not in VSCode, so I installed tinytex instead.
        - You may install TinyTeX in R: 
            ```r
            install.packages("tinytex")
            tinytex::install_tinytex()```
    <br><br>

### Tips
- `paste(names(q1_df), collapse = "', '")` will concatenate the variable names with `', '` as the separator.
- Subsetting a dataframe
    ```R
    chickwtsLinseed <- subset(q1_df, feed == "linseed")          # or
    chickwtsLinseed <- chickwts[chickwts$feed == "linseed", ]    # The comma MUST not be ommitted
    ```

