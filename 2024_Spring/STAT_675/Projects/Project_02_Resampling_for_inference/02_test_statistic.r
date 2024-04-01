script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")

set.seed(675)


n_sim <- 1000  # Number of simulations
x_len <- 1000  # Length of each sample
sig_lv <- 0.05 # Significance level. 5% threshold for rejecting null

n_pass_null_hyp <- 0
test_stats <- numeric(n_sim)

pb <- txtProgressBar(min = 0, max = n_sim, initial = 0)

for (i in 1:n_sim) {
  x <- rnorm(x_len)
  max_nsub <- max_norm_subarray(x)

  test_stat <- max_nsub$statistic
  test_stats[i] <- test_stat

  setTxtProgressBar(pb, i)
}

cat("\n")
p95_thresh <- quantile(test_stats, 0.95)

# Print the threshold
cat("Threshold for rejecting the null hypothesis at ",
    sig_lv * 100, "% significance level: ", p95_thresh, "\n", sep = "")