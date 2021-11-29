library(randomForest)

looFit <- function(i, Y, X, loadLib = FALSE) {
    if(loadLib)
        library(randomForest)
    out <- randomForest(y = Y[-i], x = X[-i, ], xtest = X[i, ])
    return(out$test$predicted)
}

set.seed(1)
## training set
n <- 200
p <- 20
X <- matrix(rnorm(n*p), nrow = n, ncol = p)
colnames(X) <- paste("X", 1:p, sep="")
X <- data.frame(X)
Y <- X[, 1] + sqrt(abs(X[, 2] * X[, 3])) + X[, 2] - X[, 3] + rnorm(n)

