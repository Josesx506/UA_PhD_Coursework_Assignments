# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("breastCancerMAINZ")
# BiocManager::install("genefu")
# BiocManager::install("HiDimDA")
# BiocManager::install("multiDA")
# BiocManager::install("sparseLDA")

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("01_cust_lda.r")

# # Load libraries
library(breastCancerMAINZ)
library(cowplot)
library(genefu)
library(ggplot2)
library(multiDA)
library(HiDimDA)
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


# # Train the models
breast.lda <- MASS::lda(x = x_train, grouping = y_train)
breast.mda <- multiDA(x_train, y_train, penalty = "EBIC",
                      equal.var = TRUE, set.options = "exhaustive")
breast.dlda <- Dlda(x_train, y_train)
breast.slda <- sda(x = x_train, y = y_train, lambda = 1e-6,
                   stop = -50, maxIte = 25, trace = TRUE)


# # Predict with the models
pred.lda <- predict(breast.lda, newdata = x_test)$class
pred.mda <- predict(breast.mda, newdata = x_test)$y.pred
pred.dlda <- predict(breast.dlda, x_test, grpcodes = levels(y_train))$class
pred.slda <- predict.sda(breast.slda, newdata = x_test)$class


# # Misclassification error
mserr.lda <- mean(pred.lda != y_test)
mserr.mda <- mean(pred.mda != y_test)
mserr.dlda <- mean(pred.dlda != y_test)
mserr.slda <- mean(pred.slda != y_test)

# print the misclassification error
cat(mserr.lda, "|", mserr.mda, "|", mserr.dlda, "|", mserr.slda, "\n")


######################## Simulate through IRIS data #########################
n_tests <- 1000
mda_mis_err <- numeric((n_tests))

for (i in 1:n_tests) {
  # Split data into training and test sets
  split_data <- train_test_split(brcnc, train_cols = ncol(feats),
                                 train_size = 0.8)

  # Extract the training data
  x_train <- split_data$x_train
  y_train <- split_data$y_train
  x_test <- split_data$x_test
  y_test <- split_data$y_test

  # Create and fit the MultiDA model data
  mda_model <- multiDA(x_train, y_train, penalty = "EBIC",
                       equal.var = TRUE, set.options = "exhaustive")
  pred_class <- predict(mda_model, newdata = x_test)$y.pred

  # Average misclassfication error
  mda_mis_err[i] <- mean(pred_class != y_test)
}

# Calculate the standard error of the average misclassification dist.
mda_mn <- round(mean(mda_mis_err), 4)
mda_se <- round(sd(mda_mis_err) / sqrt(n_tests), 4)

cat("\n\n\n################## Simulation Results ##################\n")
cat("The misclassification mean for the multiDA is", mda_mn, "\n")
cat("The misclassification standard error for the multiDA is", mda_se, "\n")

