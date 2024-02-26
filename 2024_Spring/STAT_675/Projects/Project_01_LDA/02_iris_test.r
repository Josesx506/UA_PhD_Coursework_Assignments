library(cowplot)
library(ggplot2)
library(MASS)
library(magrittr)
library(mvtnorm)
set.seed(671)


####################### IRIS Dataset #######################
iris <- data.frame(rbind(iris3[, , 1], iris3[, , 3], iris3[, , 2]),
                   Sp = rep(c("setosa", "virginica", "versicolor"), rep(50, 3)))

train_test_split <- function(data, train_cols = 4, train_size = 0.67) {
  # Shuffle the data
  data <- data[sample(nrow(data), replace = FALSE), ]

  # Determine the number of rows for the training set
  train_rows <- round(train_size * nrow(data))

  # Split the data into training and test sets
  train_data <- data[1:train_rows, ]
  test_data <- data[(train_rows + 1):nrow(data), ]

  # Extract features and target variables for training and test sets
  # Features (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
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
  eigen_decomp <- eigen(solve(within_cls_scat) %*% btw_cls_scat)

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

###################### Test the model on one realization #######################
# Split data into training and test sets
split_data <- train_test_split(iris, train_size = 0.67)
# Extract the results
x_train <- split_data$x_train
y_train <- split_data$y_train
x_test <- split_data$x_test
y_test <- split_data$y_test


# Create and fit LDA model
lda_model <- ldajo(ncomp = 2)
lda_model <- ldajo_fit(lda_model, x_train, y_train)
y_pred <- lda_predict(lda_model, x_test, proba = FALSE)

# Create confusion matrix
cust_result <- cbind.data.frame(y_pred, Actual_class = y_test)
conf_matrix <- table(cust_result)
rownames(conf_matrix) <- lda_model$grp_names
colnames(conf_matrix) <- lda_model$grp_names
cat("\n######### The Manual LDA prediction confusion matrix #########\n")
print(conf_matrix)


########################## Compare MASS LDA ##########################
train_data <- data.frame(x_train, y_train)
test_data <- data.frame(x_test, y_test)
mass_model <- lda(y_train ~ ., data = train_data, prior = c(1, 1, 1) / 3)
pred_class <- predict(mass_model, newdata = test_data)$class
mass_result <- cbind.data.frame(pred_class, Actual_class = y_test)
cfm_mass <- table(mass_result)
rownames(cfm_mass) <- lda_model$grp_names
colnames(cfm_mass) <- lda_model$grp_names
cat("\n\n########## The MASS LDA prediction confusion matrix ##########\n")
print(cfm_mass)


######################### Simulate through IRIS data #########################
n_tests <- 1000
cust_mis_err <- numeric((n_tests))
mass_mis_err  <- numeric((n_tests))

for (i in 1:n_tests) {
  # Split data into training and test sets
  split_data <- train_test_split(iris, train_size = 0.67)

  # Extract the training data
  x_train <- split_data$x_train
  y_train <- split_data$y_train
  x_test <- split_data$x_test
  y_test <- split_data$y_test

  x_train_scaled <- scale(x_train)
  x_test_scaled <- scale(x_test, center = attr(x_train_scaled, "scaled:center"),
                         scale = attr(x_train_scaled, "scaled:scale"))

  # Create and fit LDA model
  lda_model <- ldajo(ncomp = 2)
  lda_model <- ldajo_fit(lda_model, x_train_scaled, y_train)
  # Can return class probabilities or predicted classes
  y_pred <- lda_predict(lda_model, x_test_scaled, proba = FALSE)

  # Create and fit the mass model data
  train_data <- data.frame(x_train, y_train)
  test_data <- data.frame(x_test, y_test)
  mass_model <- lda(y_train ~ ., data = train_data, prior = c(1, 1, 1) / 3)
  pred_class <- predict(mass_model, newdata = test_data)$class

  # Average misclassfication error
  cust_mis_err[i] <- mean(y_pred != y_test)
  mass_mis_err[i] <- mean(pred_class != y_test)
}

# Calculate the standard error of the average misclassification dist.
cust_mn <- round(mean(cust_mis_err), 4)
mass_mn <- round(mean(mass_mis_err), 4)
cust_se <- round(sd(cust_mis_err) / sqrt(n_tests), 4)
mass_se <- round(sd(mass_mis_err) / sqrt(n_tests), 4)

cat("\n\n\n################## Simulation Results ##################\n")
cat("The misclassification standard error for the custom lda is", cust_se, "\n")
cat("The misclassification standard error for the MASS lda is", mass_se, "\n")



########################## Plot the results ##########################
# Create a dataframe for the custom and mass misclassification errors
sim_df <- cbind.data.frame(cstm_lda = cust_mis_err, mass_lda = mass_mis_err)

# Get the directory path where the script is located
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
# Set the working directory to the script directory
setwd(script_dir)

# Plot the results
plot <- ggplot(sim_df, aes(x = cstm_lda)) +
  geom_histogram(bins = 20, fill = "blue", alpha = 0.5) +
  ggtitle("Custom LDA") + xlab("Misclassification error")

# Add text annotation for mean and standard deviation
plot <- plot + annotation_custom(grid::textGrob(label =
    paste("Mean:", cust_mn, "\n", "SE:", cust_se),
  x = unit(0.85, "npc"), y = unit(0.85, "npc"),
))

# Create the second plot
plot2 <- ggplot(sim_df, aes(x = mass_lda)) +
  geom_histogram(bins = 20, fill = "green", alpha = 0.5) +
  ggtitle("Mass LDA") + xlab("Misclassification error")


# Add text annotation for mean and standard deviation
plot2 <- plot2 + annotation_custom(grid::textGrob(label =
    paste("Mean:", mass_mn, "\n", "SE:", mass_se),
  x = unit(0.85, "npc"), y = unit(0.85, "npc"),
))


# Combine the plots
combined_plot <- plot_grid(plot, plot2, labels = c("A", "B"), ncol = 2)


# Save the plot
ggsave("histogram_plot.png", combined_plot, width = 10, height = 6, dpi = 300)
################################################################################