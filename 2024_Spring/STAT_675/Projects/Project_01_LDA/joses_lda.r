# Import Libraries
library(MASS)
library(magrittr)
library(mvtnorm)
set.seed(675)


# Create plot background
plot(NULL, pch=20, xlab='Sepal Length', ylab='Sepal Width', xlim=c(4,8), ylim=c(1.5,4.5))

species <- c("setosa", "versicolor", "virginica")
species_col <- c("blue", "red", "green")
names(species_col) <- species
legend("topright", species, pch=20, col=species_col)

ellipse <- function(s,t) {
    u <- c(s,t) - center
    u %*% sigma.inv %*% u / 2
}


##################### SINGLE TEST POINT ANALYSIS #####################
x0 <- c(6.2, 4)       # Test selection point
f0 <- rep(NA, 3)      # Function likelihood
names(f0) <- species

# Plot the projected points
for (i in species) {
    X <- iris[iris$Species == i, c("Sepal.Length", "Sepal.Width")]
    S <- cov(X)                # covariance of input columns above
    mu <- apply(X, 2, mean)    # mean of input columns above
    p <- rmvnorm(1000, mean=mu, sigma=S)   # multivariate simulation or random points using real data mean and vov
    center <- apply(p, 2, mean)            # center of simulated points
    sigma <- cov(p)                        # covariance simulated points
    sigma.inv <- solve(sigma, matrix(c(1,0,0,1), 2, 2)) # sigma inverse

    f0[i] <- dmvnorm(x0, mean=mu, sigma=sigma) # density function of multivariate normal for test sel. pt.

    n <- 100
    x <- seq(3, 8, length.out=n)
    y <- seq(1.5, 6, length.out=n)
    z <- mapply(ellipse, as.vector(rep(x,n)), as.vector(outer(rep(0,n), y, `+`)))
    contour(x,y, matrix(z,n,n), levels=c(1,3), col=species_col[i], add=TRUE, lty=1)
    points(X, pch=20, col=species_col[i])
}

# Plot the test central point
points(6.2,4, pch=15, col='black', cex=3)
prior.prob <- rep(1/3,3)           # Assign equal probabilities for each class
names(prior.prob) <- species       # Add column names for each class
Bayes.prob <- f0 * prior.prob / sum(f0 * prior.prob) # Compute Bayes prob
Bayes.prob <- round(Bayes.prob, 3)                   # Round off Bayes prob per class
# The point belongs to the class with the highest probability




##################### Evaluation Data #####################
Iris <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]), 
                   Sp = rep(c("setosa", "versicolor", "virginica"), rep(50,3)))
# Train-test- split
train <- sort(sample(1:150, 100, replace= FALSE))
# test <- Iris[-train, ]
test <- setdiff(1:150, train)
table(Iris$Sp[train])
###############################################################


##################### Function to replicate LDA for multiple points #####################
# LDA <- function(n_components) {
#   model <- list(
#     n_components = n_components,
#     linear_discriminants = NULL
#   )
#   return(model)
# }

# lda_fit <- function(model, X, y) {
#   n_features <- ncol(X)
#   class_labels <- unique(y)

#   mean_overall <- colMeans(X)
#   SW <- matrix(0, n_features, n_features)
#   SB <- matrix(0, n_features, n_features)
#   for (c in class_labels) {
#     X_c <- X[y == c, ]
#     mean_c <- colMeans(X_c)
#     SW <- SW + t(X_c - mean_c) %*% (X_c - mean_c)
#     n_c <- nrow(X_c)
#     mean_diff <- matrix((mean_c - mean_overall), n_features, 1)
#     SB <- SB + n_c * (mean_diff) %*% t(mean_diff)
#   }

#   # Determine SW^-1 * SB
#   A <- solve(SW) %*% SB
#   # Get eigenvalues and eigenvectors of SW^-1 * SB
#   eigen <- eigen(A)
#   eigenvalues <- eigen$values
#   eigenvectors <- eigen$vectors
#   # sort eigenvalues high to low
#   idxs <- order(abs(eigenvalues), decreasing = TRUE)
#   eigenvalues <- eigenvalues[idxs]
#   eigenvectors <- eigenvectors[, idxs]
#   # store first n eigenvectors
#   model$linear_discriminants <- eigenvectors[, 1:model$n_components]
# }

# lda_transform <- function(model, X) {
#   # project data
#   return(X %*% model$linear_discriminants)
# }

library(Matrix)

# LDA model constructor
LDA <- function(n_components) {
  model <- list(
    n_components = n_components,
    discriminant_vectors = NULL,
    group_means = NULL,
    wccm = NULL
  )
  return(model)
}

within_scatter <- function(X, groups, class_feature_means) {
  n_features <- ncol(X)
  n_groups <- length(unique(groups))
  S_w <- matrix(0, nrow = n_features, ncol = n_features)
  
  for (c in unique(groups)) {
    class_indices <- which(groups == c)
    mean_subtracted <- t(X[class_indices, ] - class_feature_means[, as.integer(c)])
    S_w <- S_w + mean_subtracted %*% t(mean_subtracted)
  }
  
  return(S_w)
}

# Between-class scatter matrix calculation function
between_scatter <- function(X, groups, class_feature_means) {
  n_features <- ncol(X)
  n_groups <- length(unique(groups))
  
  between_class_scatter_matrix <- matrix(0, nrow = n_features, ncol = n_features)
  
  for (c in unique(groups)) {
    class_indices <- which(groups == c)
    mc <- matrix(class_feature_means[, as.integer(c)], ncol = 1)  # Reshape mc as a column vector
    m <- matrix(colMeans(X), ncol = 1)  # Reshape m as a column vector
    between_class_scatter_matrix <- between_class_scatter_matrix + nrow(X[class_indices, ]) * (mc - m) %*% t(mc - m)
  }
  
  return(between_class_scatter_matrix)
}


# LDA fitting function
lda_fit <- function(model, X, groups) {
  # Check if target variable is a factor
  if (!is.factor(groups)) {
    stop("Target variable must be a factor.")
  }
  
  # Extract number of components and dimensions
  n_components <- model$n_components
  n_features <- ncol(X)
  
  # Calculate class means
  group_means <- sapply(unique(groups), function(g) colMeans(X[groups == g, ]))
  
  # Calculate within-class covariance matrix
  S_w <- within_scatter(X, groups, group_means)
  
  # Calculate between-class scatter matrix
  S_b <- between_scatter(X, groups, group_means)
  
  # Solve the generalized eigenvalue problem
  eigen_decomp <- eigen(solve(S_w) %*% S_b)
  
  # Extract leading eigenvectors (up to max n_components)
  discriminant_vectors <- eigen_decomp$vectors[, 1:min(ncol(S_b), n_components)]

  # Calculate within-class covariance matrix for each class
  within_class_covariance_matrices <- lapply(unique(groups), function(g) {
    X_class <- X[groups == g, ]
    cov(X_class)
  })
  
  # Update model properties
  model$group_means <- group_means
  model$S_w <- S_w
  model$discriminant_vectors <- discriminant_vectors
  model$wccm <- within_class_covariance_matrices
  
  return(model)
}



# LDA transformation function
# lda_transform <- function(model, X_test) {
#   # Ensure all properties are set
#   if (is.null(model$discriminant_vectors)) {
#     stop("Model needs to be fitted before transformation.")
#   }
  
#   # Extract number of eigenvectors and test data
#   w_matrix <- model$discriminant_vectors
#   n_eigens <- model$n_components
  
#   # Project data onto LDA subspace
#   discriminant_scores <- as.matrix(X_test %*% (w_matrix))
  
#   # Calculate class probabilities
#   class_probs <- matrix(0, nrow = nrow(discriminant_scores), ncol = ncol(w_matrix) + 1)

#   for (i in 1:nrow(discriminant_scores)) {
#     for (j in 1:(ncol(w_matrix) + 1)) {
#       class_probs[i, j] <- sum(dnorm(discriminant_scores[i, ], 
#                                      mean = colMeans(discriminant_scores),
#                                      sd = apply(discriminant_scores, 2, sd))) / (ncol(w_matrix) + 1)
#     }
#   }
  
#   # Predict class labels based on maximum probability
#   predicted_classes <- apply(class_probs, 1, which.max)
#   print(class_probs)
  
#   return(predicted_classes)
# }

# lda_transform <- function(model, X_test) {
#   # Check if the model has been fitted
#   if (is.null(model$discriminant_vectors)) {
#     stop("Model needs to be fitted before transformation.")
#   }
  
#   # Extract discriminant vectors from the model
#   discriminant_vectors <- model$discriminant_vectors
  
#   # Project test data onto the discriminant subspace
#   X_lda <- X_test %*% (discriminant_vectors)
#   print(model$group_means)
  
#   # Compute class probabilities for each test point
#   class_probs <- matrix(0, nrow = nrow(X_lda), ncol = ncol(discriminant_vectors) + 1)
#   for (i in 1:nrow(X_lda)) {
#     distances <- apply(discriminant_vectors, 2, function(w) sqrt(sum((X_lda[i, ] - w)^2)))
#     print(exp(-distances) / sum(exp(-distances)))
#     print(distances %*% model$S_w  )
#     # class_probs[i, ] <- exp(-distances) / sum(exp(-distances))
#   }
  
#   # Predict class labels based on maximum probability
#   class_probs <- apply(class_probs, 1, which.max)
#   return(class_probs)
# }


lda_transform <- function(model, X_test) {
  discriminant_vectors <- model$discriminant_vectors
  group_means <- model$group_means
  
  # Transform data to discriminant space
  X_transformed <- as.matrix(X_test) %*% discriminant_vectors
  
  # Calculate class means in discriminant space
  transformed_group_means <- sapply(unique(model$groups), function(g) {
    mean(X_transformed[model$groups == g, , drop = FALSE], na.rm = TRUE)
  })
  print(transformed_group_means)
  
  # Calculate class probabilities
  class_probs <- sapply(1:nrow(X_transformed), function(i) {
    probs <- sapply(1:length(unique(model$groups)), function(j) {
      cat(X_transformed[i, ], "\n", transformed_group_means, "\n")
      d <- sqrt(sum((X_transformed[i, ] - (transformed_group_means[j, ]))^2))
      1 / (1 + d) # Using a simple inverse distance-based approach for classification
    })
    probs / sum(probs)
  })
  
  return(class_probs)
}



# Example usage:
# Define your training data X_train and labels y_train




##################### Test Model on IRIS Dataset #####################
# Convert species names to numeric labels
species_labels <- unique(Iris$Sp)
label_encoding <- as.integer(factor(Iris$Sp, levels = species_labels))

# Check for missing values in features
missing_X_train <- any(is.na(X_train))
missing_X_test <- any(is.na(X_test))

# Split the dataset into features and targets
X_train <- as.matrix(Iris[train, 1:4])   # Features (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
y_train <- label_encoding[train]         # Target variable
X_test <- as.matrix(Iris[test, 1:4])
y_test <- label_encoding[test]

# Convert the encoded labels back to species names for reference (if needed)
# species_names <- species_labels[label_encoding]


# Create and fit LDA model
lda_model <- LDA(n_components = 2)
lda_model <- lda_fit(lda_model, X_train, factor(y_train))
y_pred <- lda_transform(lda_model, X_test)

# print(dim(lda_mod$linear_discriminants))
print(y_pred)
print(y_test)








##################### Multi-point classification with LDA MASS #####################

# print(train)

# Fit lda model on 4 features. Can also work with less features but more features improves predictions
# mod <- lda(Sp ~ Sepal.L.+Sepal.W.+Petal.L.+Petal.W., Iris, prior = c(1,1,1)/3, subset=train)
# pred_class <- predict(mod, test)$class                     # Perform predictions
# cfm = cbind.data.frame(pred_class, Actual_class=test$Sp)   # Confusion matrix
# print(table(cfm))


