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
library(multiDA)
library(plda)
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

# Convert it into a dataframe for train test split
brcnc <- data.frame(feats)
brcnc$Sp <- targ

#################### Fit one realization of large dataset ####################
# Split data into training and test sets
split_data <- train_test_split(brcnc, train_cols = ncol(feats),
                               train_size = 0.8)
# Extract the results
x_train <- split_data$x_train
y_train <- split_data$y_train
x_test <- split_data$x_test
y_test <- split_data$y_test

# Train the models
# breast.lda <- MASS::lda(x = x_train, grouping = y_train)
breast.mda <- multiDA(x_train, y_train, penalty = "EBIC",
                      equal.var = TRUE, set.options = "exhaustive")
# breast.plda <- plda(x_train, y_train, type = "poisson", prior = "prox_train, y_trainportion")
# breast.slda <- sda(x = x_train, y = y_train, lambda = 1e-6,
#                    stop = -50, maxIte = 25, trace = TRUE)


# Predict with the models
# pred.lda <- predict(breast.lda, newdata = x_test)$class
pred.mda <- predict(breast.mda, newdata=x_test)$y.pred
# pred.plda <- 
# pred.slda <- predict.sda(breast.slda, newdata = x_test)$class


# Misclassification error
mean(pred.lda != y_test)
mean(pred.mda != y_test)
mean(pred.slda != y_test)