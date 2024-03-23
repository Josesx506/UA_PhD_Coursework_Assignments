script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
library(ggplot2)


# Seed value means the plot is reproduced exactly everytime
set.seed(675)  # for reproducibility



generate_changepoint_array <- function(n = 1000, m = 50, n_sd = 2) {
  #' Creates a array of values with a changepoint above the mean
  #' Returns a list of multiple variables
  #' Example
  #' result <- generate_changepoint_array()
  #' print(result$timeseries)

  # Generate random normal distribution
  data <- rnorm(n, mean = 0, sd = 1)

  # Select a random start index for the modified signal
  start_index <- sample(1:(n - m + 1), 1)
  end_index <- start_index + m - 1

  # Calculate mean and standard deviation of the original distribution
  original_mean <- mean(data)
  original_sd <- sd(data)

  # Calculate mean of the signal and its boundaries
  signal_mean <- mean(data[start_index:end_index])
  signal_start <- start_index
  signal_end <- end_index

  # Increase mean of signal linearly
  for (i in start_index:(start_index + floor(m / 2) - 1)) {
    data[i] <- data[i] + ((n_sd * original_sd - signal_mean)
                          / (m / 2)) * (i - start_index + 1)
  }

  # Decrease mean of signal linearly
  for (i in (start_index + floor(m / 3)):end_index) {
    data[i] <- data[i] + ((n_sd * original_sd - signal_mean)
                          / (m / 2)) * (end_index - i + 1)
  }

  # Calculate the mean of the entire distribution and signal segment
  entire_mean <- mean(data)
  signal_mean <- mean(data[start_index:end_index])
  norm_sum <- sum(data[start_index:end_index]) / sqrt(m)

  # Return results
  return(list(m_start = signal_start,
              m_end = signal_end,
              sig_mean = signal_mean,
              org_mean = original_mean,
              mrg_mean = entire_mean,
              norm_sum = norm_sum,
              timeseries = data))
}


# Plot the timeseries function
plot_changepoint_ts <- function(result, color = "blue", lw = 1, fs = 14,
                                save_plot = FALSE, name = NULL) {
  #' Create a lineplot of the changepoint timeseries
  #' Requires the generate_changepoint_array() function
  #' Example
  #' result <- generate_changepoint_array()
  #' plot_ts(result = result, save_plot = TRUE, name = "example_ts")

  data <- result$timeseries
  start_wnd <- result$m_start
  end_wnd <- result$m_end

  # Create data frame for ggplot
  df <- data.frame(index = seq_along(data), value = data)

  # Plot the distribution
  p <- ggplot(df, aes(x = index, y = value)) +
    geom_line(color = "black", size = lw * 0.3) +
    geom_vline(xintercept = c(start_wnd, end_wnd), size = lw,
               color = "red", linetype = "dashed") +
    xlab("Index") + ylab("Amplitude") +
    scale_x_continuous(limits = c(-1, length(data) + 1),
                       expand = c(0.01, 0.01)) +
    theme_minimal() + theme(plot.margin = unit(c(0, 0.5, 0, 0), "cm"))

  # Customize line thickness and colors
  p <- p + theme(axis.text = element_text(size = fs - 2),
                 axis.title = element_text(size = fs, face = "bold"),
                 panel.border = element_rect(colour = "black",
                                             fill = NA, size = lw * 0.5))

  # Save plot if specified
  if (save_plot) {
    if (is.null(name)) {
      print("Please provide a filename to save the plot.")
    } else {
      ggsave(filename = paste0(name, ".png"),
             plot = p, width = 10, height = 3,
             units = "in", dpi = 300)
    }
  }
  print(p)
}


max_norm_subarray <- function(x) {
  #' Get the maximum normalized subarray within a sequenc
  #' Big-O complexity is (n^2)
  #' Example
  #' result <- generate_changepoint_array()
  #' max_sub <- max_norm_subarray(result$timeseries)

  n <- length(x)
  maxx <- max(cumsum(x) / sqrt(1:n))
  left <- 1
  right <- which.max(cumsum(x) / sqrt(1:n))
  for (i in 2:n) {
    current_max <- max(cumsum((x[i:n]) / sqrt(1:(n + 1 - i))))
    if (current_max > maxx) {
      maxx <- current_max
      left <- i
      right <- which.max(cumsum((x[i:n]) / sqrt(1:(n + 1 - i)))) + i - 1
    }
  }
  mu <- mean(x[left:right])
  return(list(statistic = maxx, start = left, end = right, mu = mu))
}
