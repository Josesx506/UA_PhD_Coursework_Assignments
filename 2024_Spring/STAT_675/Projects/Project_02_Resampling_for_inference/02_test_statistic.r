script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")

set.seed(675)


n_sim <- 1000  # Number of simulations
x_len <- 1000  # Length of each sample
sig_lv <- 0.05 # Significance level. 5% threshold for rejecting null

n_pass_null_hyp <- 0
test_stats <- numeric(n_sim)

for (i in 1:n_sim) {
  x <- generate_changepoint_array(n = x_len, m = 10)
  max_nsub <- max_norm_subarray(x$timeseries)

  test_stat <- max_nsub$statistic
  test_stats[i] <- test_stat

  # Calculate threshold for rejecting null hyp using sig. level
  critical_value <- qnorm(1 - significance_level)

  # Compare with observed test statistic
  if (test_stat > critical_value) {
    n_pass_null_hyp <- n_pass_null_hyp + 1
  }
}


mn_stats <- mean(test_stats)
se_stats <- sd(test_stats) / sqrt(n_sim)
margin_of_error <- qnorm(1 - significance_level / 2) * se_stats
conf_int <- c(mn_stats - margin_of_error, mn_stats + margin_of_error)
