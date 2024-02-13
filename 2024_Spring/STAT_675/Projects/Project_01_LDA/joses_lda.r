# Import Libraries
library(MASS)
library(magrittr)
library(Matrix)
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


#################################### SINGLE TEST POINT ANALYSIS ####################################
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
#########################################################################################################



#################################### IRIS Data ####################################
Iris <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]), 
                   Sp = rep(c("setosa", "versicolor", "virginica"), rep(50,3)))
# Train-test- split indices
train <- sort(sample(1:150, 100, replace= FALSE))  # Get the indices of the training rows
test <- setdiff(1:150, train)                      # Get the indices of the test rows

# Create dictionary to convert string target classes to numeric labels (one_hot_encoding)
species_labels <- unique(Iris$Sp)
label_encoding <- as.integer(factor(Iris$Sp, levels = species_labels))

# Split the dataset into features and targets
X_train <- as.matrix(Iris[train, 1:4])   # Features (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
y_train <- label_encoding[train]         # Encode target variable
X_test <- as.matrix(Iris[test, 1:4])     #
y_test <- label_encoding[test]           # Encode target variable

# Check for missing values in features. This dataset didn't have missing values
missing_X_train <- any(is.na(X_train))
missing_X_test <- any(is.na(X_test))
#########################################################################################################


#################################### LDA Model Class ####################################
# LDA model constructor
LDA <- function(n_components) {
  model <- list(
    n_components = n_components,
    linear_discriminants = NULL,
    cls_means = NULL,            # Class mean
    cls_covrs = NULL,            # Class covariance
    cls_mlhbd = NULL             # Mahalanobis distance - I didn't really use it but I felt I should keep it
  )
  return(model)
}

within_between_scatter <- function(X, groups) {
  # I used one function for both covariance matrices to reduce loop time for large classes
  # S_w - Within class scatter matrix calculation. Should return a matrix with shape (n_feat, n_feat)
  # S_b - Between-class scatter matrix calculation function. Should return a matrix with shape (n_feat, n_feat)
  n_feat <- ncol(X)
  n_groups <- length(unique(groups))
  mean_ovr <- colMeans(X, na.rm = TRUE)          # Mean of all features
  S_w <- matrix(0, nrow = n_feat, ncol = n_feat) # within-class matrix
  S_b <- matrix(0, nrow = n_feat, ncol = n_feat) # Between class matrix
  
  for (c in unique(groups)) {
    class_indices <- which(groups == c)
    X_c <- X[class_indices, ]                      # X features for this class
    X_c_mn <- colMeans(X_c, na.rm = TRUE)          # X-mean for unique class

    # ------------------------------- Within Groups -------------------------------
    # Subtract each feature for unique classes from the class mean
    mean_sub <- t(X_c - X_c_mn) %*% (X_c - X_c_mn) # Add the dot product of the transpsed subtraction
    S_w <- S_w + mean_sub                          # Update S_w

    # ------------------------------- Between Groups -------------------------------
    n_c <- nrow(X_c)                               # Num samples per class
    # Subtract each class mean from the overall mean
    mean_diff <- matrix((X_c_mn - mean_ovr), ncol = 1, nrow = n_feat) # Reshape class mean - overall mean
    diff_sub <- mean_diff %*% t(mean_diff)
    S_b <- S_b + diff_sub                          # Update S_b
  }

  return(list(S_w=S_w,
              S_b=S_b))
}

lda_fit <- function(model, X, groups) {
  # Check if target variable is a factor
  if (!is.factor(groups)) {
    stop("Target variable must be a factor.")
  }
  
  # Extract number of components and dimensions
  n_components <- model$n_components
  n_features <- ncol(X)

  # Calculate the covariance matrices
  cv_mtx <- within_between_scatter(X, groups)
  S_w <- cv_mtx$S_w    # Extract within-class scatter matrix
  S_b <- cv_mtx$S_w    # Extract between-class scatter matrix
  
  # Solve the generalized eigenvalue problem - np.linalg.eig(np.linalg.inv(S_w).dot(S_b)) 
  eigen_decomp <- eigen(solve(S_w) %*% S_b)
  eig_vec = t(eigen_decomp$vectors)           # Eigenvectors
  eig_val = eigen_decomp$values               # Eigenvalues
  idxs <- rev(order(abs(eig_val)))            # Get indices of reverse sorted abs eigenvalues
  # Sort the eigen values and vectors using the derived indices
  eig_val <- eig_val[idxs]
  eig_vec <- eig_vec[idxs, ]
  # Extract only the biggest eigenvectors depending on the LDA components defined
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
    cov(x_proj)
  })
  cls_mlhbd <- lapply(unique(groups), function(g) {
    x_cls <- X[groups == g, ]
    x_proj <- x_cls %*% t(discriminant_vectors)
    x_mn <- colMeans(x_proj)
    x_cv <- cov(x_proj)
    mahalanobis(x_proj, x_mn, x_cv)
  })

  # Update the model attributes
  model$cls_means <- cls_means
  model$cls_covrs <- cls_covrs
  model$cls_mlhbd <- cls_mlhbd
  model$linear_discriminants <- discriminant_vectors        
  colnames(model$linear_discriminants) <- colnames(X)        # Assign column names to the eigenvectors

  return(model)
}

lda_transform <- function(model, X) {
  # Perform dot product to project test data to LDA/fit domain
  # Transpose the eigenvectors because it was transposed in transform
  X_proj <- X %*% t(model$linear_discriminants)
  return(X_proj)
}

lda_predict <- function(model, X, proba=FALSE) {
  # Project the test data into the LDA domain
  x_proj <- X %*% t(model$linear_discriminants)

  # Initialize matrix to store probabilities for each class
  class_probs <- matrix(0, nrow = nrow(X), ncol = ncol(model$cls_means))

  # Loop through each row and calculate class probabilities for each sample
  for (i in 1:nrow(x_proj)) {
    for (j in 1:ncol(model$cls_means)) {
      # Calculate likelihood using multivariate Gaussian with the class mean and cov of projected data
      likelihood <- dmvnorm(x_proj[i, ], mean = lda_model$cls_means[, j], sigma = lda_model$cls_covrs[[j]])
      # Prior probability of class j. It is eqal for each class. e.g if 3 classes, prior = 1/3 = 0.33
      prior_prob <- 1 / ncol(lda_model$cls_means)
      # Posterior probability of class j
      class_probs[i, j] <- likelihood * prior_prob
    }
  }

  # Normalize the probabilities across each row
  class_probs <- class_probs / rowSums(class_probs)
  
  # If class probabilities are requested
  if(proba) {
    return(class_probs)
  } else {
    return(apply(class_probs, 1, which.max))
  }
}
#########################################################################################################




#################################### IRIS TEST ####################################
# Create and fit LDA model
lda_model <- LDA(n_components = 2)
lda_model <- lda_fit(lda_model, X_train, factor(y_train))
# Transform and predict with LDA Model
y_trans <- lda_transform(lda_model, X_test)
y_pred <- lda_predict(lda_model, X_test)     # Can return class probabilities or predicted classes

# Convert the encoded labels back to species names for reference (if needed)
rnmd_pred <- species_labels[y_pred]

# Plot transformed LDA dimensions
plot(y_trans[, 1], y_trans[, 2], col = y_test, pch = 16, 
     main = "Scatter Plot with Color Array", xlab = "X", ylab = "Y")

# Create confusion matrix - 3 feats outperforms 4 feats
conf_matrix <- table(y_pred, y_test)
print(conf_matrix)
#########################################################################################################


#################################### Compare MASS LDA ####################################


#########################################################################################################




#################################### Large Class Test ####################################


#########################################################################################################


