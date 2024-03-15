set.seed(675)

# Parameters
num_samples <- 100  # Number of samples in each sequence
num_sequences <- 10000  # Number of sequences to generate
significance_level <- 0.05  # Chosen significance level

# Generate sequences under the null hypothesis (standard normal noise)
sequences <- matrix(rnorm(num_samples * num_sequences), nrow=num_sequences)

# Calculate test statistics (mean of each sequence)
test_statistics <- apply(sequences, 1, mean)

# Determine the threshold based on the empirical distribution
threshold <- quantile(test_statistics, 1 - significance_level)

# Count the number of times the null hypothesis is rejected (type I error)
num_rejections <- sum(test_statistics > threshold)

# Calculate the proportion of rejections (type I error rate)
type_1_error_rate <- num_rejections / num_sequences

cat("Type I error rate:", type_1_error_rate, "\n")

# Check if the type I error rate is close to the chosen significance level
if (abs(type_I_error_rate - significance_level) < 0.01) {
  cat("Type I error is well controlled.\n")
} else {
  cat("Type I error is not well controlled.\n")
}
