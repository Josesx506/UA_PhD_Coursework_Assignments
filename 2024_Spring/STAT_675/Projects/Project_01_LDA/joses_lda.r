# Import Libraries
library(MASS)
library(magrittr)
library(Matrix)
library(mvtnorm)
library(datamicroarray)
set.seed(671)
options(scipen = 5)


#################################### IRIS Data ####################################
Iris <- data.frame(rbind(iris3[, , 1], iris3[, , 3], iris3[, , 2]),
                   Sp = rep(c("setosa", "virginica", "versicolor"), rep(50, 3)))

# Dictionary to convert  classes to numeric labels (one_hot_enc)
species_labels <- unique(Iris$Sp)
label_encoding <- as.integer(factor(Iris$Sp, levels = species_labels))

train_test_split <- function(data, train_size = 0.67) {
  # Shuffle the data
  data <- data[sample(nrow(data), replace = FALSE), ]

  # Determine the number of rows for the training set
  train_rows <- round(train_size * nrow(data))

  # Split the data into training and test sets
  train_data <- data[1:train_rows, ]
  test_data <- data[(train_rows + 1):nrow(data), ]

  # Extract features and target variables for training and test sets
  # Features (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
  x_train <- as.matrix(train_data[, 1:4])
  x_test <- as.matrix(test_data[, 1:4])
  # Encode target variable
  y_train <- factor(train_data$Sp)
  y_test <- factor(test_data$Sp)

  # Check for missing values in features. This dataset didn't have missing vals
  missing_x_train <- any(is.na(x_train))
  missing_x_test <- any(is.na(x_test))

  return(list(x_train = x_train, y_train = y_train,
              x_test = x_test, y_test = y_test))
}
#########################################################################################################





#################################### LDA Model Class ####################################
# LDA model constructor
LDA <- function(n_components) {
  model <- list(
    n_components = n_components,
    linear_discriminants = NULL, # equiv of mass_model$scaling
    cls_means = NULL,            # Class mean
    cls_covrs = NULL,            # Class covariance
    cls_mlhbd = NULL,            # Mahalanobis distance - Unused but retained
    cls_prior = NULL,            # Prior probability for each class
    grp_names = NULL
  )
  return(model)
}

within_between_scatter <- function(X, groups) {
  # I used one function for both covariance matrices
  #       to reduce loop time for large classes
  # s_w - Within class scatter matrix calculation.
  #       Should return a matrix with shape (n_feat, n_feat)
  # s_b - Between-class scatter matrix calculation function.
  #       Should return a matrix with shape (n_feat, n_feat)
  n_feat <- ncol(X)
  mean_ovr <- colMeans(X, na.rm = TRUE)          # Mean of all features
  s_w <- matrix(0, nrow = n_feat, ncol = n_feat) # within-class matrix
  s_b <- matrix(0, nrow = n_feat, ncol = n_feat) # Between class matrix
  # s_b2 <- list()

  for (c in unique(groups)) {
    class_indices <- which(groups == c)
    x_c <- X[class_indices, ]                      # X features for this class
    x_c_mn <- colMeans(x_c, na.rm = TRUE)          # X-mean for unique class

    # ----------------------------- Within Groups -----------------------------
    # Subtract each feature for unique classes from the class mean
    # Add the dot product of the transposed subtraction
    mean_sub <- x_c - x_c_mn
    dot_prod_mean_sub <- t(mean_sub) %*% mean_sub
    s_w <- s_w + dot_prod_mean_sub                 # Update s_w

    # ----------------------------- Between Groups -----------------------------
    n_c <- nrow(x_c)                               # Num samples per class
    # Subtract each class mean from the overall mean
    # Reshape class mean - overall mean
    # mean_diff <- x_c_mn - mean_ovr
    # diff_sub <- n_c * outer(mean_diff, mean_diff)
    # s_b2[[c]] <- diff_sub
    mean_diff <- matrix((x_c_mn - mean_ovr), ncol = 1, nrow = n_feat)
    diff_sub <- n_c * (mean_diff %*% t(mean_diff))
    s_b <- s_b + diff_sub                          # Update s_b
  }

  # s_b2 <- Reduce(`+`, s_b2)
  # print(s_b2)
  # print(s_b)

  return(list(s_w = s_w,
              s_b = s_b))
}

lda_fit <- function(model, X, groups) {
  # Check if target variable is a factor
  if (!is.factor(groups)) {
    stop("Target variable must be a factor.")
  }

  # Extract number of components and dimensions
  n_components <- model$n_components

  # Calculate the covariance matrices
  cv_mtx <- within_between_scatter(X, groups)
  s_w <- cv_mtx$s_w    # Extract within-class scatter matrix
  s_b <- cv_mtx$s_b    # Extract between-class scatter matrix

  # Solve the generalized eigenvalue problem -
  # np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
  eigen_decomp <- eigen(solve(s_w) %*% s_b)
  eig_vec <- eigen_decomp$vectors           # Eigenvectors
  eig_val <- eigen_decomp$values            # Eigenvalues
  # Get indices of reverse sorted abs eigenvalues
  idxs <- order(eig_val, decreasing = TRUE)
  # Sort the eigen values and vectors using the derived indices
  eig_val <- eig_val[idxs]
  eig_vec <- eig_vec[, idxs]
  # Extract the biggest eigenvectors depending on num LDA components
  discriminant_vectors <- eig_vec[1:n_components, ]

  # Calculate class means and covariance for each class in projected dimensions
  cls_means <- sapply(unique(groups), function(g) {
    x_cls <- X[groups == g, ]
    x_proj <- x_cls %*% t(discriminant_vectors)
    colMeans(x_proj)
  })
  cls_covrs <- lapply(unique(groups), function(g) {
    x_cls <- X[groups == g, ]
    x_proj <- x_cls %*% t(discriminant_vectors)
    cov(abs(x_proj))
  })
  # cls_mlhbd <- lapply(unique(groups), function(g) {
  #   x_cls <- X[groups == g, ]
  #   x_proj <- x_cls %*% t(discriminant_vectors)
  #   x_mn <- colMeans(x_proj)
  #   x_cv <- cov(x_proj)
  #   mahalanobis(x_proj, x_mn, x_cv)
  # })
  cls_prior <- lapply(unique(groups), function(g) {
    # Prior probability of each class
    x_cls <- X[groups == g, ]
    nrow(x_cls) / nrow(X)
  })

  # Update the model attributes
  model$cls_means <- cls_means
  model$cls_covrs <- cls_covrs
  # model$cls_mlhbd <- cls_mlhbd
  model$cls_prior <- cls_prior
  model$linear_discriminants <- discriminant_vectors
  model$grp_names <- unique(groups)
  # Assign column names to the class means and eigenvectors
  colnames(model$cls_means) <- unique(groups)
  colnames(model$linear_discriminants) <- colnames(X)

  return(model)
}

lda_transform <- function(model, X) {
  # Perform dot product to project test data to LDA/fit domain
  # Transpose the eigenvectors because it was transposed in transform
  x_proj <- X %*% t(model$linear_discriminants)
  return(x_proj)
}

gaussian_distribution <- function(x, u, cov) {
  scalar <- (1 / ((2 * pi) ^ (length(x) / 2))) * (1 / sqrt(det(cov)))
  x_sub_u <- x - u
  exponent <- -1 / 2 * t(x_sub_u) %*% solve(cov) %*% x_sub_u
  return(scalar * exp(exponent))
}

lda_predict <- function(model, X, proba = FALSE) {
  # Project the test data into the LDA domain
  x_proj <- X %*% t(model$linear_discriminants)

  # Initialize matrix to store probabilities for each class
  class_probs <- matrix(0, nrow = nrow(X), ncol = length(model$grp_names))

  # Loop through each row and calculate class probabilities for each sample
  for (i in seq_len(nrow(x_proj))) { #1:nrow(x_proj)
    for (c in lda_model$grp_names) {
      cl_mean <- lda_model$cls_means[, c]
      j <- as.integer(factor(c, levels = species_labels))
      cl_cov <- lda_model$cls_covrs[[j]]

      # Calculate likelihood using multivariate Gaussian with
      # the class mean and cov of projected data
      likelihood <- dmvnorm(x_proj[i, ],
                            mean = cl_mean,
                            sigma = cl_cov)
      # likelihood <- gaussian_distribution(x_proj[i, ], cl_mean, cl_cov)

      # Prior probability of class j. It is eqal for each class.
      # e.g if 3 classes, prior = 1/3 = 0.33
      # prior_prob <- length(x_proj) / length(lda_model$grp_names)
      cl_prior <- lda_model$cls_prior[[j]]

      # Posterior probability of class j
      class_probs[i, j] <- likelihood * cl_prior
    }
  }

  # Normalize the probabilities across each row
  class_probs <- class_probs / rowSums(class_probs)

  # If class probabilities are requested
  if (proba) {
    return(class_probs)
  } else {
    return(apply(class_probs, 1, which.max))
  }
}

#########################################################################################################




#################################### IRIS TEST ####################################

# Split data into training and test sets
split_data <- train_test_split(Iris, train_size = 0.67)
# Extract the results
x_train <- split_data$x_train
y_train <- split_data$y_train
x_test <- split_data$x_test
y_test <- split_data$y_test


# Create and fit LDA model
lda_model <- LDA(n_components = 2)
lda_model <- lda_fit(lda_model, x_train, y_train)
# Transform and predict with LDA Model
y_trans <- lda_transform(lda_model, x_test)
# Can return class probabilities or predicted classes
y_pred <- lda_predict(lda_model, x_test, proba = FALSE)
# Convert the encoded labels back to species names for reference (if needed)
rnmd_pred <- species_labels[y_pred]

# Plot transformed LDA dimensions
# plot(y_trans[, 1], y_trans[, 2], col = y_test, pch = 16,
#      main = "Scatter Plot with Color Array", xlab = "X", ylab = "Y")

# Create confusion matrix
conf_matrix <- table(rnmd_pred, y_test)
rownames(conf_matrix) <- species_labels
colnames(conf_matrix) <- species_labels
cat("\n######### The Manual LDA prediction confusion matrix #########\n")
print(conf_matrix)
#########################################################################################################


#################################### Compare MASS LDA ####################################
train_data <- data.frame(x_train, y_train)
test_data <- data.frame(x_test, y_test)
mass_model <- lda(y_train ~ ., data = train_data, prior = c(1, 1, 1) / 3)
pred_class <- predict(mass_model, newdata = test_data)$class
mass_result <- cbind.data.frame(pred_class, Actual_class = y_test)
cfm_mass <- table(mass_result)
rownames(cfm_mass) <- species_labels
colnames(cfm_mass) <- species_labels
cat("\n\n\n######### The MASS LDA prediction confusion matrix #########\n")
print(cfm_mass)

#########################################################################################################


#################################### Loop through IRIS data ####################################
n_tests <- 500
cust_mis_err <- numeric((n_tests))
mass_mis_err  <- numeric((n_tests))

for (i in 1:n_tests) {
  # Split data into training and test sets
  split_data <- train_test_split(Iris, train_size = 0.67)

  # Extract the training data
  x_train <- split_data$x_train
  y_train <- split_data$y_train
  x_test <- split_data$x_test
  y_test <- split_data$y_test

  x_train_scaled <- scale(x_train)
  x_test_scaled <- scale(x_test,
    center = attr(x_train_scaled, "scaled:center"),
    scale = attr(x_train_scaled, "scaled:scale"))

  # Create and fit LDA model
  lda_model <- LDA(n_components = 2)
  lda_model <- lda_fit(lda_model, x_train_scaled, y_train)
  # Can return class probabilities or predicted classes
  y_pred <- lda_predict(lda_model, x_test_scaled, proba = FALSE)
  # Convert the encoded labels back to species names for reference (if needed)
  rnmd_pred <- species_labels[y_pred]

  # Create and fit the mass model data
  train_data <- data.frame(x_train, y_train)
  test_data <- data.frame(x_test, y_test)
  mass_model <- lda(y_train ~ ., data = train_data, prior = c(1, 1, 1) / 3)
  pred_class <- predict(mass_model, newdata = test_data)$class

  # Average misclassfication error
  cust_mis_err[i] <- mean(rnmd_pred != y_test)
  mass_mis_err[i] <- mean(pred_class != y_test)
}

# Calculate the standard error of the average misclassification dist.
cust_se <- round(sd(cust_mis_err) / sqrt(n_tests), 4)
mass_se <- round(sd(mass_mis_err) / sqrt(n_tests), 4)

cat("The misclassification standard error for the custom lda is", cust_se, "\n")
cat("The misclassification standard error for the MASS lda is", mass_se, "\n")
#################################################################################################



#################################### Load large dataset ####################################
data('sorlie', package = 'datamicroarray')   # Breast Cancer Dataset
# alon_df <- as.data.frame(alon)


#################################### Large Class Test ####################################
# print(lda_model)
# print(mass_model)

#########################################################################################################