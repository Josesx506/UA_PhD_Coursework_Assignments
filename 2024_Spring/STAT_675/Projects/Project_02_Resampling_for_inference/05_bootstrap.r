script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")

set.seed(675)


n_sim <- 1000  # Number of simulations
x_len <- 1000  # Length of each sample
sig_lv <- 0.05 # Significance level. 5% threshold for rejecting null

x <- rnorm(x_len)
test_statistic <- max_norm_subarray(x)$statistic

pb <- txtProgressBar(min = 0, max = n_sim, initial = 0)

boot_test_stats <- numeric(n_sim)
count_reject <- 0

for (i in 1:n_sim) {
  boot_smp <- sample(x, replace = TRUE)
  boot_test_stat <- max_norm_subarray(boot_smp)$statistic
  boot_test_stats[i] <- boot_test_stat

  # Reject null hypothesis if test statistic is greater than threshold
  if (boot_test_stat > test_statistic) {
    count_reject <- count_reject + 1
  }

  setTxtProgressBar(pb, i)
}

cat("\n")

boot_test_statistic <- mean(boot_test_stats)

# Output the results
cat("Original Test Statistic:", test_statistic, "\n")
cat("Bootstrap Test Statistic:", boot_test_statistic, "\n")


# Calculate type I error rate
type1_error_rate <- count_reject / n_sim

# Print type I error rate
cat("Type I error rate:", type1_error_rate, "\n")

# Check if the type I error rate is close to the chosen significance level
if (abs(type1_error_rate - sig_lv) < 0.01) {
  cat("Type I error is well controlled.\n")
} else {
  cat("Type I error is not well controlled.\n")
}