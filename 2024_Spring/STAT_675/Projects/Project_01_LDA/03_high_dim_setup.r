# Set the working directory to the script directory
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("01_cust_lda.r")

# Import Libraries
library(MASS)
library(datamicroarray)
set.seed(675)
options(scipen = 5)

# Load the chris dataset. 217 rows, 3 classes, 1413 features
data("christensen", package = "datamicroarray")
chris <- christensen$x
chris$Sp <- christensen$y


#################### Fit one realization of large dataset ####################
# Split data into training and test sets
split_data <- train_test_split(chris, train_cols = ncol(christensen$x))
# Extract the results
x_train <- split_data$x_train
y_train <- split_data$y_train
x_test <- split_data$x_test
y_test <- split_data$y_test


# Create and fit LDA model
lda_model <- ldajo(ncomp = 2)
lda_model <- ldajo_fit(lda_model, x_train, y_train)
y_pred <- lda_predict(lda_model, x_test, proba = FALSE)
conf_matrix <- table(y_pred, y_test)
print(conf_matrix)

train_data <- data.frame(x_train, y_train)
test_data <- data.frame(x_test, y_test)
mass_model <- lda(y_train ~ ., data = train_data,
                  prior = rep(1, 3) / 3)
pred_class <- predict(mass_model, newdata = test_data)$class
conf_matrix <- table(pred_class, y_test)
print(conf_matrix)


######################## Large Class Test ########################
# It's slow for multiple realization tests, so I ran only one rlztn.

# Alon Dataset. (Too few rows to meet project criteria)
# Warning MASS: In lda.default(x, grouping, ...) : variables are collinear

# Christensen Dataset
# Warning MASS: In lda.default(x, grouping, ...) : variables are collinear
#####################################################################