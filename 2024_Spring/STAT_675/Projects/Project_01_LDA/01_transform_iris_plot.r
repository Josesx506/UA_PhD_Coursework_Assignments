script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("01_cust_lda.r")

# Load library
library(cowplot)
library(ggplot2)

# Set seed
set.seed(675)


####################### IRIS Dataset #######################
iris <- data.frame(rbind(iris3[, , 1], iris3[, , 3], iris3[, , 2]),
                   Sp = rep(c("setosa", "virginica", "versicolor"), rep(50, 3)))


###################### Test the model on one realization #######################
# Split data into training and test sets
split_data <- train_test_split(iris, train_size = 0.50)
# Extract the results
x_train <- split_data$x_train
y_train <- split_data$y_train
x_test <- split_data$x_test
y_test <- split_data$y_test


# Create and fit LDA model
lda_model <- ldajo(ncomp = 2)
lda_model <- ldajo_fit(lda_model, x_train, y_train)
y_trans <- as.matrix(x_test) %*% lda_model$lin_disc


# Create a data frame for plotting
plot_data <- data.frame(
  LD1 = y_trans[, 1],
  LD2 = y_trans[, 2],
  Sp = y_test
)

# Define colors for species
species_col <- c("blue", "red", "green")

# Create the plot
pl <- ggplot(plot_data, aes(x = LD1, y = LD2, color = Sp)) +
  geom_point(shape = 16) +
  labs(title = "Scatter Plot of Linear Discriminants",
       x = "LD 1", y = "LD 2") +
  scale_color_manual(values = species_col) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.text = element_text(size = 10)) +
  guides(color = guide_legend(title = "Species",
                              override.aes = list(shape = 20)))

cwpl <- plot_grid(pl, labels = c("A"), ncol = 1)

ggsave("linear_disc.png", plot = cwpl, width = 8, height = 6, dpi = 300)