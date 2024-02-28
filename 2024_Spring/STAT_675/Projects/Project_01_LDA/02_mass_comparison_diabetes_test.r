# BiocManager::install("mlbench")
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("01_cust_lda.r")

# Load libraries
library(cowplot)
library(ggplot2)
library(MASS)
library(mlbench)
set.seed(675)

# Load dataset
data("PimaIndiansDiabetes")
classic <- PimaIndiansDiabetes
# Rename the label column to Sp
names(classic)[names(classic) == "diabetes"] <- "Sp"

###################### Test the model on one realization #######################
# Split data into training and test sets
split_data <- train_test_split(classic, train_cols = 8, train_size = 0.90)
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
mass_model <- lda(y_train ~ ., data = train_data, prior = c(1, 1) / 2)
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
  split_data <- train_test_split(classic, train_cols = 8, train_size = 0.85)

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
  mass_model <- lda(y_train ~ ., data = train_data, prior = c(1, 1) / 2)
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
ggsave("hist_plt_classic.png", combined_plot, width = 10, height = 6, dpi = 300)
################################################################################