# Set the working directory to the script directory
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("01_cust_lda.r")

# Import Libraries
library(MASS)
library(datamicroarray)
set.seed(675)
options(scipen = 5)

# Load the golub dataset. 72 rows, 2 classes, 7130 features
data("golub", package = "datamicroarray")
glb <- data.frame(golub$x)
glb$Sp <- golub$y

#################### Fit one realization of large dataset ####################
# Split data into training and test sets
split_data <- train_test_split(glb, train_cols = ncol(golub$x),
                               train_size = 0.9)
# Extract the results
x_train <- split_data$x_train
y_train <- split_data$y_train
x_test <- split_data$x_test
y_test <- split_data$y_test


# Create and fit LDA model
# lda_model <- ldajo(ncomp = 2)
# lda_model <- ldajo_fit(lda_model, x_train, y_train)
# y_pred <- lda_predict(lda_model, x_test, proba = FALSE)
# conf_matrix <- table(y_pred, y_test)
# print(conf_matrix)

train_data <- data.frame(x_train, y_train)
test_data <- data.frame(x_test, y_test)
mass_model <- lda(y_train ~ ., data = train_data,
                  prior = rep(1, 2) / 2)
pred_class <- predict(mass_model, newdata = test_data)$class
conf_matrix <- table(pred_class, y_test)
print(conf_matrix)


#################### Data Limit tests ####################
# It can handle golub dataset with
# 72 rows, 7129 columns, 2 target labels