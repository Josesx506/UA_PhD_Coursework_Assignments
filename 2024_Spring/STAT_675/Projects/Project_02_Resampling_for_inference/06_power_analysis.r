script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")

library(Hmisc)
library(ggplot2)
set.seed(675)


n_sim <- 10000  # Number of simulations
x_len <- 1000  # Length of each sample
sig_lv <- 0.05 # Significance level. 5% threshold for rejecting null
m_sim <- 1000  # Number of simulations for alternative hypothesis
alt_mu <- c(seq(-0.2, 0.25, 0.025)) # alternatives
power <- numeric(length(alt_mu))


cat("Starting simulation for ", n_sim, " runs ",
    "with distribution length of ", x_len, " ......\n", sep = "")


repl_thresh <- replicate(n_sim, {
  max_norm_subarray(rnorm(x_len))$statistic
})
p95_thresh <- quantile(repl_thresh, 0.95)


# Print the threshold
cat("Threshold for rejecting the null hypothesis at ",
    sig_lv * 100, "% significance level: ", p95_thresh, "\n", sep = "")

pb <- txtProgressBar(min = 0, max = length(alt_mu), initial = 0)

# Compute the power
for (i in seq_along(alt_mu)){
  mu <- alt_mu[i]
  pvalues <- replicate(m_sim, {
    as.integer(max_norm_subarray(rnorm(x_len, mu))$statistic > p95_thresh)
  })
  power[i] <- mean(pvalues)
  setTxtProgressBar(pb, i)
}
cat("\n")

# Standard error
se <- sqrt(power * (1 - power) / m_sim)

# Create a dataframe with your data
df <- data.frame(alt_mu, power, se)

# Create the plot
p <- ggplot(df, aes(x = alt_mu, y = power)) +
  geom_line(linetype = "dashed") +    # Dashed line
  geom_point() +                      # Points
  geom_errorbar(aes(ymin = power - se, ymax = power + se),
                width = 0.02) +        # Error bars
  geom_hline(yintercept = 0.05, linetype = "solid") +         # Horizontal line
  labs(x = expression(theta)) +       # X-axis label
  theme_minimal()                     # Minimal theme

detach(package:Hmisc)

ggsave(filename =  "power-plot.png", plot = p, width = 6, height = 5,
       units = "in", dpi = 300)

cat("Completed Plot \n")