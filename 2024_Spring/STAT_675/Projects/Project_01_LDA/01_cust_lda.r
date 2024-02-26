# Import Libraries
library(MASS)
library(magrittr)
library(mvtnorm)

# Train-test-split modification
# The label column bust always be named Sp to match iris data
# train_cols - can be modified to the number of X feature cols
train_test_split <- function(data, train_cols = 4,
                             train_size = 0.75) {
  # Shuffle the data
  data <- data[sample(nrow(data), replace = FALSE), ]

  # Determine the number of rows for the training set
  train_rows <- round(train_size * nrow(data))

  # Split the data into training and test sets
  train_data <- data[1:train_rows, ]
  test_data <- data[(train_rows + 1):nrow(data), ]

  # Extract features and target variables for training and test sets
  x_train <- as.matrix(train_data[, 1:train_cols])
  x_test <- as.matrix(test_data[, 1:train_cols])
  # Encode target variable
  y_train <- factor(train_data$Sp)
  y_test <- factor(test_data$Sp)

  return(list(x_train = x_train, y_train = y_train,
              x_test = x_test, y_test = y_test))
}


####################### LDA model constructor #######################
ldajo <- function(ncomp) {
  model <- list(
    ncomp = ncomp,    # number of LDA components
    lin_disc = NULL,  # linear discriminants equiv of mass_model$scaling
    cls_means = NULL, # Class mean
    cls_covrs = NULL, # Class covariance
    cls_mlhbd = NULL, # Mahalanobis distance - Unused but retained
    cls_prior = NULL, # Prior probability for each class
    grp_names = NULL
  )
  return(model)
}

ldajo_fit <- function(model, x_inp, y_targ) {
  # Compute class means
  class_means <- t(sapply(unique(y_targ),
                          function(i) colMeans(x_inp[y_targ == i, ])))

  # Compute the overall mean
  overall_mean <- colMeans(x_inp)

  # Compute within-class scatter matrix
  within_cls_scat <- array(0, dim = c(ncol(x_inp), ncol(x_inp)))
  for (i in unique(y_targ)) {
    within_cls_scat <- within_cls_scat +
      nrow(x_inp[y_targ == i, ]) * cov(x_inp[y_targ == i, ])
  }

  # Compute between-class scatter matrix
  cls_1_diff <- class_means[1, ] - overall_mean
  btw_cls_scat <- nrow(x_inp) * (cls_1_diff) %*% t(cls_1_diff)
  for (i in 2:length(unique(y_targ))) {
    n_class <- nrow(x_inp[y_targ == levels(y_targ)[i], ])
    cls_i_diff <- class_means[i, ] - overall_mean
    btw_cls_scat <- btw_cls_scat + n_class * (cls_i_diff) %*% t(cls_i_diff)
  }

  # Solve the generalized eigenvalue problem
  if (det(within_cls_scat) == 0) {
    eigen_decomp <- eigen(ginv(within_cls_scat) %*% btw_cls_scat)
  } else {
    eigen_decomp <- eigen(solve(within_cls_scat) %*% btw_cls_scat)
  }

  # Get eigenvalues and eigenvectors
  eigen_values <- eigen_decomp$values
  eigen_vectors <- eigen_decomp$vectors

  # Get indices of reverse sorted eigenvalues
  idxs <- order(eigen_values, decreasing = TRUE)

  # Sort eigenvalues and eigenvectors
  eigen_values <- eigen_values[idxs]
  eigen_vectors <- eigen_vectors[, idxs]

  # Extract the biggest eigenvectors limit num_LDA_components
  discr_vects <- eigen_vectors[, 1:model$ncomp]

  # Update the model attributes
  model$lin_disc <- Re(discr_vects)
  model$grp_names <- unique(y_targ)

  # Calculate the mean, covariance, and prior probabilities
  model <- lda_attrs(model, x_inp, y_targ)

  return(model)
}

lda_attrs <- function(model_fit, x_inp, y_targ, mhl = FALSE) {
  # Calculate the transformed mean, covariance, and prior probability
  # It can also compute the mahalanobis distance
  trns_means <- list()
  trns_prior <- list()
  model_fit$cls_covrs <- list()

  if (mhl) {
    model_fit$cls_mlhbd <- list()
  }

  for (i in unique(y_targ)) {
    x_class <- x_inp[y_targ == i, ]
    # Project data onto the discriminant vectors
    x_proj <- as.matrix(x_class) %*% model_fit$lin_disc
    xpj_mn <- colMeans(x_proj)                    # Mean
    trns_prior[[i]] <- nrow(x_proj) / nrow(x_inp) # Prior
    xpj_cv <- cov(x_proj)                         # Covariance
    if (mhl) {
      model_fit$cls_mlhbd[[i]] <- mahalanobis(x_proj, xpj_mn, xpj_cv)
    }
    # Update the model entries
    trns_means[[i]] <- xpj_mn
    model_fit$cls_covrs[[i]] <- xpj_cv
  }

  # Convert to a dataframe and format it properly
  model_fit$cls_means <- do.call(rbind, trns_means)
  model_fit$cls_prior <- do.call(rbind, trns_prior)
  rownames(model_fit$cls_means) <- names(trns_means)
  rownames(model_fit$cls_prior) <- names(trns_prior)

  return(model_fit)
}


lda_predict <- function(model, x_test_, proba = FALSE) {
  # Project the test data into the LDA domain
  x_proj <- as.matrix(x_test_) %*% model$lin_disc

  # Initialize matrix to store probabilities for each class
  class_probs <- matrix(0, nrow = nrow(x_test_), ncol = length(model$grp_names))

  # Loop through each row and calculate class probabilities for each sample
  for (i in seq_len(nrow(x_proj))) {
    for (c in model$grp_names) {
      cl_mean <- model$cls_means[c, ]
      cl_cov <- model$cls_covrs[[c]]

      # Calculate likelihood using multivariate Gaussian with
      # the class mean and cov of projected data
      likelihood <- dmvnorm(x_proj[i, ],
                            mean = cl_mean,
                            sigma = cl_cov)

      # Prior probability of class c
      cl_prior <- model$cls_prior[c, ]

      # Convert the class label to a number for the column
      nc <- as.integer(factor(c, levels = model$grp_names))
      # Posterior probability of class j
      class_probs[i, nc] <- likelihood * cl_prior
    }
  }

  # Normalize the probabilities across each row
  class_probs <- class_probs / rowSums(class_probs)

  # If class probabilities are requested
  if (proba) {
    return(class_probs)
  } else {
    return(model$grp_names[apply(class_probs, 1, which.max)])
  }
}

#################################################################