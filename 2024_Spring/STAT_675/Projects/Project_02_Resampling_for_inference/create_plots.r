script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(script_dir)
source("utils.r")


# Example usage
result <- generate_changepoint_array()
plot_changepoint_ts(result = result, save_plot = TRUE, name = "timeseries_1")