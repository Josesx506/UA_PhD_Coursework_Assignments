script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")


# Example usage
result <- generate_changepoint_array()
# plot_changepoint_ts(result = result, save_plot = TRUE, name = "timeseries_1")


# Show changepoints in dataset
x <- c(rnorm(300, 0, 1),
       rnorm(40, 2, 1),
       rnorm(260, 0, 1),
       rnorm(30, 1.5, 1),
       rnorm(370, 0, 1))

df <- data.frame(idx = seq_along(x), amp = x)
lw <- 1
fs <- 14

# Plot the distribution
p <- ggplot(df, mapping = aes(x = idx, y = amp)) +
  geom_line(color = "black", size = lw * 0.3) +
#   geom_vline(xintercept = c(start_wnd, end_wnd), size = lw,
#              color = "red", linetype = "dashed") +
  xlab("Index") + ylab("Amplitude") +
  scale_x_continuous(limits = c(-1, dim(df)[1] + 1),
                     expand = c(0.01, 0.01)) +
  theme_minimal() + theme(plot.margin = unit(c(0, 0.5, 0, 0), "cm"))

# Customize line thickness and colors
p <- p + theme(axis.text = element_text(size = fs - 2),
               axis.title = element_text(size = fs, face = "bold"),
               panel.border = element_rect(colour = "black",
                                           fill = NA, linewidth = lw * 0.5))
ggsave(filename =  "multi-changepoint.png", plot = p, width = 10, height = 3,
       units = "in", dpi = 300)
