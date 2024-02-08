## High Dimensional LDA implementation
- [intro](#intro)
- [setup](#setup)

### Intro
Prof. Hints: A direct implementation of the LDA method might not be effective to handle high dimensional data because of the following reasons. First, it might be slow and unstable to find the inverse of a large sample covariance matrix. Second, while pseudo inverse is well-defined theoretically, in practice the smallest non-zero singular value of the sample covariance might be extremely close to zero, which also leads to numerical instability. You may check how the lda() is implemented and handle the small singular values. <br>

Prof. Ning Hao has a published implementation
- [SPCALDA](https://cran.r-project.org/web/packages/SPCALDA/index.html) documentation - https://cran.r-project.org/web/packages/SPCALDA/SPCALDA.pdf
- Publication link - https://www3.stat.sinica.edu.tw/statistica/J28N1/J28N19/J28N19.html
- arXiV link - https://arxiv.org/abs/1511.00282
- Obtain high-dimensional dataset from https://github.com/ramhiser/datamicroarray

### Setup
- Install the following packages to get the professors implementation and dataset for testing the project
    ```R
    install.packages('SPCALDA')
    install.packages('devtools')
    ```
- Install the dataset library from Github for easier loading in R
    ```R
    library(devtools)
    install_github('ramhiser/datamicroarray')
    ```
- View all installed R packages
    ```R
    packINFO <- as.data.frame(installed.packages())[,c("Package", "Version")]
    rownames(packINFO) <- NULL
    ```
- Load a Dataset from the github library for testing high dimensionality
    ```R
    library(datamicroarray)
    data('alon', package = 'datamicroarray')
    ```