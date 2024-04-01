script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")

set.seed(675)


# Simulate data with epidemic change
simula_data <- function(n, change_point, mean_before, mean_after) {
  data <- rnorm(n, mean = mean_before, sd = 1)
  data[change_point:n] <- rnorm(n - change_point + 1, mean = mean_after, sd = 1)
  return(data)
}

# Set simulation parameters
n <- 1000            # Sample size
change_point <- 600  # Change point location (after 500th data point)
mean_before <- 0     # Mean before the change point
mean_after <- 1      # Mean after the change point (epidemic increase)

# Simulate data
simulated_data <- simula_data(n, change_point, mean_before, mean_after)

# Calculate the maximum normalized subarray statistic
statistic <- max_norm_subarray(simulated_data)

# Print results
cat("Maximum normalized subarray statistic:", statistic$statistic, "\n")