script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")

set.seed(675)


n_sim <- 10000  # Number of simulations
x_len <- 1000  # Length of each sample
dof <- 50      # 50 degrees of freedom
sig_lv <- 0.05 # Significance level. 5% threshold for rejecting null

n_pass_null_hyp <- 0
test_statsn <- numeric(n_sim)
test_statst <- numeric(n_sim)

cat("Starting simulation for ", n_sim, " runs ",
    "with distribution length of ", x_len, " ......\n", sep = "")


repl_thresh <- replicate(n_sim, {
  max_norm_subarray(scaled_t(x_len, dof))$statistic
})
p95_thresh <- quantile(repl_thresh, 0.95)


# Print the threshold
cat("Threshold for rejecting the null hypothesis at ",
    sig_lv * 100, "% significance level: ", p95_thresh, "\n", sep = "")


hyp_test <- replicate(n_sim, {
  as.integer(max_norm_subarray(scaled_t(x_len, dof))$statistic > p95_thresh)
})

# Calculate type I error rate
type1_error_rate <- mean(hyp_test)

# Print type I error rate
cat("Type I error rate:", type1_error_rate, "\n")

# Check if the type I error rate is close to the chosen significance level
if (abs(type1_error_rate - sig_lv) < 0.01) {
  cat("Type I error is well controlled.\n")
} else {
  cat("Type I error is not well controlled.\n")
}