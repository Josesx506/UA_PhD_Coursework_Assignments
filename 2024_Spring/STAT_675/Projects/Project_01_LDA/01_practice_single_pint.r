# Import Libraries
library(MASS)


# Create plot background
# plot(NULL, pch=20, xlab='Sepal Length', ylab='Sepal Width', xlim=c(4,8), ylim=c(1.5,4.5))

species <- c("setosa", "versicolor", "virginica")
species_col <- c("blue", "red", "green")
names(species_col) <- species
# legend("topright", species, pch = 20, col = species_col)

ellipse <- function(s, t) {
  u <- c(s, t) - center
  u %*% sigma.inv %*% u / 2
}


#################################### SINGLE TEST POINT ANALYSIS ####################################
x0 <- c(6.2, 4)       # Test selection point
f0 <- rep(NA, 3)      # Function likelihood
names(f0) <- species

# Plot the projected points
for (i in species) {
    print(i)
  X <- iris[iris$Sp == i, c("Sepal.L.", "Sepal.W.")]
  S <- cov(X)                # covariance of input columns above
  mu <- apply(X, 2, mean)    # mean of input columns above
  # multivariate simulation or random points using real data mean and vov
  p <- rmvnorm(1000, mean = mu, sigma = S)
  center <- apply(p, 2, mean)            # center of simulated points
  sigma <- cov(p)                        # covariance simulated points
  sigma.inv <- solve(sigma, matrix(c(1, 0, 0, 1), 2, 2)) # sigma inverse

  # density function of multivariate normal for test sel. pt.
  f0[i] <- dmvnorm(x0, mean = mu, sigma = sigma)

  n <- 100
  x <- seq(3, 8, length.out = n)
  y <- seq(1.5, 6, length.out = n)
  z <- mapply(ellipse, as.vector(rep(x, n)),
              as.vector(outer(rep(0, n), y, `+`)))
  contour(x, y, matrix(z, n, n), levels = c(1, 3),
          col = species_col[i], add = TRUE, lty = 1)
  points(X, pch = 20, col = species_col[i])
}

# Plot the test central point
# points(6.2, 4, pch = 15, col = "black", cex = 3)
prior.prob <- rep(1 / 3, 3)       # Assign equal probabilities for each class
names(prior.prob) <- species      # Add column names for each class
Bayes.prob <- f0 * prior.prob / sum(f0 * prior.prob) # Compute Bayes prob
Bayes.prob <- round(Bayes.prob, 3)            # Round off Bayes prob per class
# The point belongs to the class with the highest probability
#########################################################################################################