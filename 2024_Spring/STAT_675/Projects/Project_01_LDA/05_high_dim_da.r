# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("breastCancerMAINZ")
# BiocManager::install("genefu")
# BiocManager::install("sparseLDA")

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("01_cust_lda.r")

# Load libraries
library(breastCancerMAINZ)
library(cowplot)
library(genefu)
library(ggplot2)
library(sparseLDA)
library(MASS)
set.seed(675)

# Load breast cancer dataset
data(mainz)

# Prepare the input
feats <- t(exprs(mainz)) # gene expressions
n <- nrow(feats)
h <- diag(n) - 1 / n * matrix(1, ncol = n, nrow = n)
feats <- h %*% feats
targ <- pData(mainz)$grade


# breast.lda <- MASS::lda(x = feat, grouping = targ)
# print(dim(feats))

