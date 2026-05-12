# Sithija Manage. ssm255@cornell.edu
# krtexas package
# krtexas_functions.R

# NOTE:
# In a package, dependencies should be declared in DESCRIPTION (Imports/LinkingTo)
# and referenced with :: or :::. We avoid calling library() at top-level here.

#' Internal imports
#'
#' @name krtexas_internal_imports
#' @keywords internal
#' @noRd
#' @import foreach
#' @import doParallel
NULL

#' Fit KR TEXAS model (adaptive-penalty version)
#'
#' This is the high-level user-facing function. It first fits KR TEXAS with
#' lambda = 0 to obtain initial weights, then re-fits with an adaptive penalty.
#'
#' @param X Numeric matrix of predictors (n x p).
#' @param Y Numeric response vector of length n.
#' @param A Numeric matrix defining the tree / aggregation structure (M x p).
#' @param kernel Kernel type, "gaussian" or "epanechnikov".
#' @param nfolds Number of CV folds.
#' @param lambda Optional fixed lambda. If NULL, a lambda path is considered.
#' @param min_lambda Minimum lambda for path (if lambda NULL).
#' @param max_lambda Maximum lambda for path (if lambda NULL).
#' @param nlambda Number of lambdas in path.
#' @param alpha Sigma penalty parameter.
#' @param eps Optimizer tolerance (factr for L-BFGS-B).
#' @param gamma_threshold Threshold for zeroing gammas.
#' @param use_num_grad Whether to force numerical gradient (unused in current code).
#' @param silent If TRUE, suppresses most output from optim.
#' @param parallel Whether to use parallel CV.
#' @param n_cores Number of cores for parallel CV (default: detectCores() - 1).
#' @param warm_start Whether to warm-start gammas across lambdas.
#'
#' @return A list with elements:
#'   \itemize{
#'     \item gammas_learned
#'     \item lambda.best
#'     \item sigma_opt
#'     \item training_loss
#'     \item cv_lambda_losses
#'     \item convergence
#'     \item inputs (X, Y, A, kernel)
#'   }
#' @export
krtexas_fit <- function(X,
                        Y,
                        A,
                        kernel = "gaussian",
                        nfolds = 4,
                        lambda = NULL,
                        min_lambda = 0,
                        max_lambda = NULL,
                        nlambda = 10,
                        alpha = 0,
                        eps = 1e-6,
                        gamma_threshold = 0,
                        use_num_grad = FALSE,
                        silent = TRUE,
                        parallel = TRUE,
                        n_cores = NULL,
                        warm_start = TRUE,
                        method = "LLR",
                        gamma_init_strat = "smallest",
                        distance = "L2",
                        num_restarts_stage_1 = 10,
                        num_restarts_stage_2 = 10,
                        max_attempts_stage_3 = 10,
                        developer_code = 0) {
  cat("Running \n")
  cat(" +--------------+.  ●\n")
  cat(" |.  KR TEXAS   |. / \\ \n")
  cat(" +--------------+ ●.  ●\n")
  cat("                     / \\ \n")
  cat("                    ●.  ●\n")
  cat("Packaged Version 04/22/2026\n")
  cat("With Updated Algorithm 1. \n")

  cat("Running KR TEXAS with lambda = 0 to calculate weights (Stage 1/2)...\n")
  ###
  ## Stage 1: lambda = 0 with random restarts (num_restarts_stage_1)
  best_model0     <- NULL
  best_loss0      <- Inf
  failed_restarts <- 0L
  best_init_strat_1  <- NA_character_

  for (m in seq_len(num_restarts_stage_1)) {
    cat("  Stage 1 restart", m, "of", num_restarts_stage_1, "...\n")

    # Decide gamma_init_strat for this restart
    gamma_init_strat_m <- if (m == 1) {
      gamma_init_strat
    } else if (m %% 3 == 0) {
      "large"
    } else if (m %% 3 == 1){
      "small"
    } else if (m %% 3 == 2){
      "smallest"
    }
    krfit <- tryCatch({
      krtexas_fit_internal(
        X = X,
        Y = Y,
        A = diag(ncol(X)),
        kernel = kernel,
        nfolds = nfolds,
        lambda = 0,
        min_lambda = min_lambda,
        max_lambda = max_lambda,
        nlambda = nlambda,
        alpha = alpha,
        eps = eps,
        gamma_threshold = gamma_threshold,
        use_num_grad = use_num_grad,
        silent = silent,
        parallel = parallel,
        n_cores = n_cores,
        warm_start = warm_start,
        adaptive_weights = rep(1, ncol(X)),
        gamma_init_strat = gamma_init_strat_m
      )
    }, error = function(e) {
      cat(e$message, "\n")
      failed_restarts <<- failed_restarts + 1L
      ## Return an object with the fields we rely on, but with infinite loss
      list(
        gammas_learned = rep(NA_real_, ncol(X)),
        training_loss  = Inf,
        convergence    = NA_integer_
      )
    })

    ## Keep best by training_loss, but only if convergence is reasonable
    if (is.finite(krfit$training_loss) &&
        krfit$training_loss < best_loss0 &&
        krfit$convergence %in% c(0, 1, -1234, 52)) {

      best_model0       <- krfit
      best_loss0        <- krfit$training_loss
      best_init_strat_1 <- gamma_init_strat_m

      cat("New Stage 1 best found (init =",
          gamma_init_strat_m, "\n")
    }
  }# for stage 1

  cat("Stage 1 winner initialization:",
      best_init_strat_1, "\n")


  if (is.null(best_model0)) {
    stop("All KR TEXAS lambda = 0 restarts (Stage 1) failed; failed_restarts = ",
         failed_restarts)
  }

  krtx_model_0 <- best_model0
  ###

  krtx_model_0$gammas <- (krtx_model_0$gammas)^(3/4) #undersmoothing
  m = floor(nrow(X)/10)
  int_ind_ml    <- find_interior_points(X, krtx_model_0$gammas, m)$ind #m=200

  if(method == "NW_ML"){
    if(distance == "L1")
      C_list <- get_adaptive_penalties_internal_NWML_L1(krtx_model_0, A, int_ind_ml)
    if(distance == "L2")
      C_list <- get_adaptive_penalties_internal_NWML_L2(krtx_model_0, A, int_ind_ml)
  }# if NW_ML
  if(method=="LLR"){
    n = nrow(X)
    p = ncol(X)
    grad_llr <- get_grad_llr(X = X,
                             y = Y,
                             X0 = X,
                             krtx_model_0$gammas)[,-1] #(drop intercept column)
    if(distance == "L1")
      C_list <- get_adaptive_penalties_internal_LLR_L1(grad_llr, A, X, int_ind_ml)
    if(distance == "L2")
      C_list <- get_adaptive_penalties_internal_LLR_L2(grad_llr, A, X, int_ind_ml)
  }# if LLR
  if(method=="LQR"){
    n = nrow(X)
    p = ncol(X)
    grad_lqr <- get_grad_lqr(X = X,
                             y = Y,
                             X0 = X,
                             krtx_model_0$gammas) #(drop intercept column)
    if(distance == "L1") #can just use same function as LLR.
      C_list <- get_adaptive_penalties_internal_LLR_L1(grad_lqr, A, X, int_ind_ml)
    if(distance == "L2")
      C_list <- get_adaptive_penalties_internal_LLR_L2(grad_lqr, A, X, int_ind_ml)
  }# if LQR

  #cat("Gammas Learned with lambda = 0: ", krtx_model_0$gammas_learned, "\n")

  C1 <- C_list$C1
  C2 <- C_list$C2
  C3 <- C_list$C3
  cat("C1: ", C1, "\n")
  cat("C2: ", C2, "\n")
  cat("C3: ", C3, "\n")
  #utils::View(data.frame(cbind(C1, C2, C3)))

  n <- nrow(X)
  b <- 1
  a2 <- 1/(2*(2 + ncol(X)))  # smaller than 1/(2 + ncol(X))
  w <- (1 / C2)^b + (1 / C3)^b + (n^a2 * C1)^b # length M

  #cat("w = ", w, "\n")

  # Replace Inf by large number for computation in weights.
  if (any(is.infinite(w))) {
    #cat("Infinite weight at:  ", which(is.infinite(w)), " (this is not a problem)\n")
    cap_w <- 1e50
    #cat("Capping Inf at ", paste0(cap_w), "\n")
    w[is.infinite(w)] <- cap_w
  }

  cat("Weights Calculated.\n")
  if(!is.null(lambda)){ #if lambda specified
    best_lambda2 <- lambda
  }else{

    cat("Running KR TEXAS with adaptive penalty (Stage 2/2)...\n")

    best_lambda2     <- NULL
    best_loss2      <- Inf
    failed_restarts2 <- 0L
    best_u <- NULL
    best_w <- NULL
    best_init_strat_2 <- NA_character_

    make_folds <- function(n, nfolds) {
      rep(1:nfolds, length.out = n)[sample.int(n)]
    }
    folds <- make_folds(nrow(X), nfolds)

    for (m in seq_len(num_restarts_stage_2)) {
      cat("  Stage 2 restart", m, "of", num_restarts_stage_2, "...\n")

      # Decide gamma_init_strat for this restart
      gamma_init_strat_m <- if (m == 1) {
        gamma_init_strat
      } else if (m %% 3 == 0) {
        "large"
      } else if (m %% 3 == 1){
        "small"
      } else if (m %% 3 == 2){
        "smallest"
      }
      tau = nrow(A)

      if(gamma_init_strat_m == "small"){
        init_u <- stats::rnorm(n = tau, mean = 1, sd = 1/4)
        init_w <- stats::rnorm(n = tau, mean = 1, sd = 1/4)
      } else if(gamma_init_strat_m == "large"){
        ### Updated ###
        init_u <- stats::rnorm(n = tau, mean = max(1, n^(2/(4+tau))), sd = 1)
        init_w <- stats::rnorm(n = tau, mean = max(1, n^(2/(4+tau))), sd = 1)
      } else if(gamma_init_strat_m == "smallest"){
        ### Updated ###
        starting_gamma <- abs( rnorm(n=tau, mean = 0.1, sd = 0.01) )
        cat("Using smallest gamma initialization of starting_gamma[1:20]: ", starting_gamma[1:20], "\n")
        init_u <- sqrt(starting_gamma)
        init_w <- sqrt(starting_gamma)
      }

      gamma_init <- init_u * init_w


      krfit2 <- tryCatch({
        krtexas_fit_internal2(
          X = X,
          Y = Y,
          A = A,
          kernel = kernel,
          nfolds = nfolds,
          lambda = lambda,
          min_lambda = min_lambda,
          max_lambda = max_lambda,
          nlambda = nlambda,
          alpha = alpha,
          eps = eps,
          gamma_threshold = gamma_threshold,
          use_num_grad = use_num_grad,
          silent = silent,
          parallel = parallel,
          n_cores = n_cores,
          warm_start = TRUE,
          adaptive_weights = w,
          gamma_init_strat = NULL,
          folds_input = folds,
          gamma_init = gamma_init
        )
      }, error = function(e) {
        failed_restarts2 <<- failed_restarts2 + 1L
        message("Stage 2 restart ", m, " failed: ", conditionMessage(e))

        # Return a dummy object with all expected fields
        list(
          gammas_learned = rep(NA_real_, ncol(X)),
          training_loss  = Inf,
          convergence    = NA_integer_,
          loss.best      = Inf,          # or NA_real_
          lambda.best    = NA_real_,
          init_u.best    = rep(NA_real_, nrow(A)),
          init_w.best    = rep(NA_real_, nrow(A))
        )
      })

      if (is.finite(krfit2$loss.best) &&
          krfit2$loss.best < best_loss2) {

        best_lambda2      <- krfit2$lambda.best
        best_loss2        <- krfit2$loss.best
        best_u            <- krfit2$init_u.best
        best_w            <- krfit2$init_w.best
        best_init_strat_2 <- gamma_init_strat_m

        cat("New Stage 2 best found (init =",
            gamma_init_strat_m,")\n")
      }
    }# for m stage 2

    cat("Stage 2 winner initialization:",
        best_init_strat_2, "\n")

    if (is.null(best_lambda2)) {
      stop("Best lambda was not found. It is null. All KR TEXAS Stage 2 restarts failed; failed_restarts2 = ",
           failed_restarts2)
    }
  }#else (lambda unspecified)

  #=== STAGE 3 ===#
  if(!is.null(best_u)){
    best_gamma_init <- best_u * best_w
  } else{
    best_gamma_init <- NULL
  }

  cat("Final KR TEXAS fit initial best_gamma_init: ",
      if (!is.null(best_gamma_init)) best_gamma_init else NA, "\n")

  ## Stage 3: robust final fit with retries + perturbations of best_gamma_init
  attempt <- 1L
  krfit3  <- NULL
  last_err <- NULL

  cat("Health Diagnostics:\n")
  cat("best_lambda2: ", best_lambda2, "\n")
  cat("w: ", w, "\n")

  while (attempt <= max_attempts_stage_3) {
    cat("Stage 3 attempt", attempt, "of", max_attempts_stage_3, "...\n")
    print("<")
    print(attempt)
    print(">")

    # Decide gamma_init_strat for this restart
    gamma_init_strat_3 <- if (attempt %% 3 == 0) {
      "large"
    } else if (attempt %% 3 == 1){
      "small"
    } else if (attempt %% 3 == 2){
      "smallest"
    }
    tau = nrow(A)

    if(gamma_init_strat_3 == "small"){
      init_u <- stats::rnorm(n = tau, mean = 1, sd = 1/4)
      init_w <- stats::rnorm(n = tau, mean = 1, sd = 1/4)
    } else if(gamma_init_strat_3 == "large"){
      init_u <- stats::rnorm(n = tau, mean = max(1, n^(2/(4+tau))), sd = 1)
      init_w <- stats::rnorm(n = tau, mean = max(1, n^(2/(4+tau))), sd = 1)
    } else if(gamma_init_strat_3 == "smallest"){
      starting_gamma <- abs( rnorm(n=tau, mean = 0.1, sd = 0.01) )
      cat("Using smallest gamma initialization of starting_gamma[1:20]: ", starting_gamma[1:20], "\n")
      init_u <- sqrt(starting_gamma)
      init_w <- sqrt(starting_gamma)
    }

    if(attempt!=1){
      best_gamma_init <- init_u * init_w
      print("Attempt is not 1 so setting best_gamma_init was set now. ")
    }else{
      print("Attempt is 1 so using best_gamma_init. ")
    }
    # * * *
    # Construct gamma_init_try for this attempt
    if (!is.null(best_gamma_init)) {
        gamma_init_try <- best_gamma_init
    } else {
      # If we have no best_gamma_init, fall back to letting krtexas_fixed_lambda
      # handle initialization internally.
      gamma_init_try <- NULL
    }
    # * * *

    krfit3 <- tryCatch({
      krtexas_fixed_lambda(
        X = X,
        Y = Y,
        A = A,
        kernel = "gaussian",
        lambda = best_lambda2,
        alpha = 0,
        eps = eps,
        gamma_threshold = 0,
        silent = silent,
        gamma_init = gamma_init_try,
        adaptive_weights = w,
        cap_loss = TRUE,
        gamma_init_strat = NULL
      )
    }, error = function(e) {
      last_err <<- e
      cat("  Stage 3 attempt", attempt, "failed: ", conditionMessage(e), "\n")
      NULL
    })

    # If we got a non-NULL result, treat it as success and break
    if (!is.null(krfit3)) {
      cat("Stage 3 succeeded on attempt", attempt, ".\n")
      break
    }

    # If gamma_init is NULL, there's no point in further attempts
    if (is.null(best_gamma_init)) {
      cat("Stage 3 failed with gamma_init = NULL and cannot perturb. Aborting.\n")
      break
    }

    attempt <- attempt + 1L
  } # while

  # If still NULL after attempts, throw a clean error
  if (is.null(krfit3)) {
    stop("Final KR TEXAS Stage 3 fit failed after ",
         attempt - 1L, " attempts. Last error: ",
         if (!is.null(last_err)) conditionMessage(last_err) else "unknown.")
  }

  # Build final model
  krtexas_model <- list(
    gammas_learned = as.vector(abs(krfit3$gammas)),
    lambda.best    = best_lambda2,
    sigma_opt      = 1,
    training_loss  = krfit3$training_loss,
    convergence    = krfit3$convergence,
    inputs         = list(X = X, Y = Y, A = A, kernel = kernel)
  )
  # ===




  krtexas_model$init_summary <- list(
    stage1_best = best_init_strat_1,
    stage2_best = best_init_strat_2
  )

  cat("Done.\n")
  krtexas_model
} # krtexas_fit

#' Internal KR TEXAS fitter for a given set of adaptive weights
#'
#' @keywords internal
krtexas_fit_internal2 <- function(X,
                                  Y,
                                  A,
                                  kernel = "gaussian",
                                  nfolds = 4,
                                  lambda = NULL,
                                  min_lambda = 0,
                                  max_lambda = NULL,
                                  nlambda = 10,
                                  alpha = 0,
                                  eps = 1e-2,
                                  gamma_threshold = 1e-6,
                                  use_num_grad = FALSE,
                                  silent = TRUE,
                                  parallel = TRUE,
                                  n_cores = NULL,
                                  warm_start = FALSE,
                                  adaptive_weights,
                                  gamma_init_strat = "smallest",
                                  developer_code = 0,
                                  folds_input = NULL,
                                  gamma_init = NULL) {
  # Checks and setup
  while (sink.number() > 0) {
    sink()
  }

  lambdaSpecified <- FALSE
  if (!is.null(lambda)) {
    lambdaSpecified <- TRUE
    min_lambda <- lambda
    max_lambda <- lambda
    nlambda <- 1
    nfolds <- 1
  }

  # Check if nlambda is a whole number and non-negative
  if (nlambda %% 1 != 0 || nlambda < 0) {
    stop("Error: nlambda must be a non-negative whole number.")
  }
  if (!is.matrix(A)) {
    stop("Error: The object 'A' is not a matrix.")
  }

  # Generate cross-validation splits
  folds <- folds_input

  # PARALLEL SETUP
  if (parallel && !lambdaSpecified) {
    if (is.null(n_cores)) {
      n_cores <- min(parallel::detectCores() - 1L, nfolds)  # Leave one core free, max = nfolds
    }
    cat("Setting up parallel processing with", n_cores, "cores\n")
    cl <- parallel::makeCluster(n_cores)
    doParallel::registerDoParallel(cl)

    # Export ALL necessary objects to cluster
    parallel::clusterExport(
      cl,
      c(
        "krtexas_fixed_lambda", "krtexas_predict_internal",
        "loss_fun", "grad_fun", "loss_fun_ep", "grad_fun_ep",
        "X", "Y", "A", "kernel", "eps",
        "gamma_threshold", "use_num_grad", "silent", "folds"
      ),
      envir = environment()
    )

    # Load required libraries on each worker
    parallel::clusterEvalQ(cl, {
      library(Matrix)
      library(lbfgs)
      library(numDeriv)
      library(krtexas) # package name; same behavior as original code
    })
  }

  lambda_losses <- numeric(nlambda)
  lambda_index <- 1L
  tau <- nrow(A)
  resultsMatrix <- matrix(
    nrow = nlambda * nfolds,
    ncol = 3L * tau + 3L
  )

  ### NEW: store the gamma_init *before* each lambda, for warm_start = TRUE
  if (!lambdaSpecified && warm_start) {
    lambda_gamma_inits <- matrix(NA_real_, nrow = nlambda, ncol = tau)
  } else {
    lambda_gamma_inits <- NULL
  }
  ### END NEW

  if (!lambdaSpecified) {
    # doing CV

    # If max_lambda unspecified, do while loop to get it
    if (is.null(max_lambda)) {
      cat("Finding max_lambda for CV...\n")
      gamma_init_copy <- gamma_init
      all_gammas_zero <- FALSE
      lambda_try <- 1e-10 # start with a number, then exponentiate by 2 iteratively to get to max_lambda
      while (!all_gammas_zero & lambda_try < 1e30) {
        cat("Trying lambda = ", lambda_try, ".\n")
        model_try <- krtexas_fixed_lambda(
          X = X,
          Y = Y,
          A = A,
          kernel = kernel,
          lambda = lambda_try,
          alpha = alpha,
          eps = eps,
          gamma_threshold = gamma_threshold,
          silent = silent,
          gamma_init = gamma_init_copy,
          adaptive_weights = adaptive_weights,
          cap_loss = FALSE,
          gamma_init_strat = gamma_init_strat
        )
        gammas_try <- model_try$gammas
        if (any(gammas_try != 0)) {
          # if any gammas are still nonzero
          lambda_try <- lambda_try * 2
        } else {
          all_gammas_zero <- TRUE
        }

        gamma_init_copy <- gammas_try
        gamma_init_copy[gamma_init_copy == 0] <- 1e-2
      } # while

      max_lambda <- lambda_try
      cat("max_lambda found: ", max_lambda, "\n")
    } # if max_lambda null

    if (min_lambda == 0) {
      lambda_seq <- c(
        0,
        exp(seq(log(1e-15), log(max_lambda), length.out = nlambda - 1))
      )
    } else {
      lambda_seq <- exp(seq(log(min_lambda), log(max_lambda), length.out = nlambda))
    }

    for (lambda in lambda_seq) {
      cat("============================================\n")
      cat("Processing lambda =", lambda, "...\n")

      ### NEW: record the *current* gamma_init for this lambda index
      if (warm_start && !is.null(lambda_gamma_inits)) {
        if (is.null(gamma_init)) {
          # if nothing has been set yet, leave NA for this lambda
          lambda_gamma_inits[lambda_index, ] <- NA_real_
        } else {
          lambda_gamma_inits[lambda_index, ] <- gamma_init
        }
      }
      ### END NEW

      # Export lambda to cluster for this iteration
      if (parallel) {
        parallel::clusterExport(cl, "lambda", envir = environment())
      }

      if (parallel) {
        # PARALLEL VERSION - Process all folds for this lambda in parallel
        fold_results <- foreach::foreach(
          fold = 1:nfolds,
          .combine = rbind,
          .packages = c("Matrix", "lbfgs", "numDeriv")
        ) %dopar% {

          train_indices <- which(folds != fold)
          test_indices <- which(folds == fold)

          X_train <- X[train_indices, , drop = FALSE]
          Y_train <- Y[train_indices]
          X_test <- X[test_indices, , drop = FALSE]
          Y_test <- Y[test_indices]

          if (nfolds == 1) {
            X_train <- X_test
            Y_train <- Y_test
          }

          # Fit the model
          fit <- krtexas_fixed_lambda(
            X = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            lambda = lambda,
            alpha = alpha,
            eps = eps,
            gamma_threshold = gamma_threshold,
            silent = TRUE, # Force silent in parallel
            gamma_init = gamma_init,
            adaptive_weights = adaptive_weights,
            cap_loss = TRUE,
            gamma_init_strat = gamma_init_strat
          )

          # Make predictions
          y_pred <- krtexas_predict_internal(
            gammas_test = fit$gammas,
            X_test = X_test,
            X_train = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            sigma = fit$sigma_opt
          )

          # Compute loss
          residuals <- Y_test - y_pred
          loss <- sum(residuals^2)

          list(
            fold = fold,
            loss = loss,
            avg_loss = loss / length(Y_test),
            fit_results = c(
              fit$gammas,
              fit$convergence,
              fit$training_loss,
              fit$sigma_opt,
              fit$init_u,
              fit$init_w
            )
          )
        }#dopar

        # Process parallel results
        lambda_loss <- mean(sapply(fold_results[, "loss"], as.numeric))

        # Fill results matrix
        start_row <- (lambda_index - 1L) * nfolds + 1L
        end_row <- lambda_index * nfolds
        for (i in seq_len(nfolds)) {
          resultsMatrix[start_row + i - 1L, ] <- fold_results[[i, "fit_results"]]
        }

        if (warm_start == TRUE) {
          # Processing Gammas to get new gamma_init for *next* lambda
          gamma_fits <- do.call(
            rbind,
            lapply(
              1:nfolds,
              function(i) {
                as.numeric(fold_results[[i, "fit_results"]][1:nrow(A)])
              }
            )
          )
          gamma_init <- colMeans(gamma_fits, na.rm = TRUE)  # keep na.rm = TRUE as you had
          fit_convergences <- sapply(
            1:nfolds,
            function(i) {
              as.numeric(fold_results[[i, "fit_results"]][nrow(A) + 1L])
            }
          )
          gamma_init[gamma_init == 0] <- 1e-2
          gamma_init[is.na(gamma_init)] <- stats::rnorm(n = 1, mean = 1, sd = 1)

          sigma_fits <- sapply(
            1:nfolds,
            function(i) {
              as.numeric(fold_results[[i, "fit_results"]][nrow(A) + 3L])
            }
          )
        } # warm_start
      } else {
        # SEQUENTIAL VERSION
        lambda_loss <- 0
        for (fold in 1:nfolds) {
          resultRow <- (lambda_index - 1L) * nfolds + fold
          cat("Executing fold", fold, "/", nfolds, "for lambda =", lambda, "...\n")

          train_indices <- which(folds != fold)
          test_indices <- which(folds == fold)

          X_train <- X[train_indices, , drop = FALSE]
          Y_train <- Y[train_indices]
          X_test <- X[test_indices, , drop = FALSE]
          Y_test <- Y[test_indices]

          if (nfolds == 1) {
            X_train <- X_test
            Y_train <- Y_test
          }

          # Fit the model
          fit <- krtexas_fixed_lambda(
            X = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            lambda = lambda,
            alpha = alpha,
            eps = eps,
            gamma_threshold = gamma_threshold,
            silent = silent,
            gamma_init = gamma_init,
            adaptive_weights = adaptive_weights,
            cap_loss = TRUE,
            gamma_init_strat = gamma_init_strat
          )

          resultsMatrix[resultRow, ] <- c(
            fit$gammas,
            fit$convergence,
            fit$training_loss,
            fit$sigma_opt,
            fit$init_u,
            fit$init_w
          )

          # Make predictions
          y_pred <- krtexas_predict_internal(
            gammas_test = fit$gammas,
            X_test = X_test,
            X_train = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            sigma = fit$sigma_opt
          )

          # Compute loss
          residuals <- Y_test - y_pred
          loss <- sum(residuals^2)
          lambda_loss <- lambda_loss + loss / nfolds
        }
      }

      lambda_losses[lambda_index] <- lambda_loss
      lambda_index <- lambda_index + 1L
    } # for lambda

    # Clean up parallel cluster
    if (parallel && !lambdaSpecified) {
      parallel::stopCluster(cl)
      foreach::registerDoSEQ()
    }

    # Find optimal lambda
    finite <- is.finite(lambda_losses)
    if (!any(finite)) {
      stop("All CV losses were non-finite (NA/NaN/Inf). Cannot choose lambda. Try parallel = FALSE once and/or cap your lambda range.")
    }
    best_idx <- which(finite)[ which.min(lambda_losses[finite]) ]
    lambda_opt <- lambda_seq[best_idx]

    loss_opt <- lambda_losses[best_idx]

    ### CHANGED: instead of taking init_u / init_w from one fold in resultsMatrix,
    ### use the gamma_init that existed *before* running best_idx's lambda,
    ### then split via sqrt into init_u.best and init_w.best.
    if (warm_start && !is.null(lambda_gamma_inits)) {
      best_gamma_init <- lambda_gamma_inits[best_idx, ]

      # Handle NA entries conservatively
      if (all(is.na(best_gamma_init))) {
        init_u_best <- NULL
        init_w_best <- NULL
      } else {
        # replace NAs by small positive number or by mean of non-NA entries if you prefer
        if (any(is.na(best_gamma_init))) {
          # simple choice: set NA entries to the minimum positive non-NA, or a tiny value
          pos_vals <- best_gamma_init[!is.na(best_gamma_init) & best_gamma_init > 0]
          fill_val <- if (length(pos_vals) > 0L) min(pos_vals) else 1e-8
          best_gamma_init[is.na(best_gamma_init)] <- fill_val
        }

        init_u_best <- sqrt(best_gamma_init)
        init_w_best <- sqrt(best_gamma_init)
      }
    } else {
      init_u_best <- NULL
      init_w_best <- NULL
    }
    ### END CHANGED

  } else {
    # Single lambda case (no CV)
    lambda_opt <- lambda
    loss_opt <- NA
    if (!is.null(gamma_init)) {
      gi <- gamma_init
      gi[is.na(gi)] <- 1e-8
      gi[gi <= 0] <- 1e-8
      init_u_best <- sqrt(gi)
      init_w_best <- sqrt(gi)
    } else {
      init_u_best <- NULL
      init_w_best <- NULL
    }
  }

  list(
    lambda.best = lambda_opt,
    loss.best   = loss_opt,
    init_u.best = init_u_best,
    init_w.best = init_w_best
  )
} # krtexas_fit_internal2

#' Internal KR TEXAS fitter for a given set of adaptive weights (with final fit)
#'
#' @keywords internal
krtexas_fit_internal <- function(X,
                                 Y,
                                 A,
                                 kernel = "gaussian",
                                 nfolds = 4,
                                 lambda = NULL,
                                 min_lambda = 0,
                                 max_lambda = NULL,
                                 nlambda = 10,
                                 alpha = 0,
                                 eps = 1e-2,
                                 gamma_threshold = 1e-6,
                                 use_num_grad = FALSE,
                                 silent = TRUE,
                                 parallel = TRUE,
                                 n_cores = NULL,
                                 warm_start = FALSE,
                                 adaptive_weights,
                                 gamma_init_strat = "smallest",
                                 developer_code = 0) {
  # Checks and setup
  while (sink.number() > 0) {
    sink()
  }

  lambdaSpecified <- FALSE
  if (!is.null(lambda)) {
    lambdaSpecified <- TRUE
    min_lambda <- lambda
    max_lambda <- lambda
    nlambda <- 1
    nfolds <- 1
  }

  # Check if nlambda is a whole number and non-negative
  if (nlambda %% 1 != 0 || nlambda < 0) {
    stop("Error: nlambda must be a non-negative whole number.")
  }
  if (!is.matrix(A)) {
    stop("Error: The object 'A' is not a matrix.")
  }

  # Generate cross-validation splits
  folds <- sample.int(nfolds, size = nrow(X), replace = TRUE)

  # PARALLEL SETUP
  if (parallel && !lambdaSpecified) {
    if (is.null(n_cores)) {
      n_cores <- min(parallel::detectCores() - 1L, nfolds)  # Leave one core free, max = nfolds
    }
    cat("Setting up parallel processing with", n_cores, "cores\n")
    cl <- parallel::makeCluster(n_cores)
    doParallel::registerDoParallel(cl)

    # Export ALL necessary objects to cluster
    parallel::clusterExport(
      cl,
      c(
        "krtexas_fixed_lambda", "krtexas_predict_internal",
        "loss_fun", "grad_fun", "loss_fun_ep", "grad_fun_ep",
        "X", "Y", "A", "kernel", "eps",
        "gamma_threshold", "use_num_grad", "silent", "folds"
      ),
      envir = environment()
    )

    # Load required libraries on each worker
    parallel::clusterEvalQ(cl, {
      library(Matrix)
      library(lbfgs)
      library(numDeriv)
      library(krtexas) # package name; same behavior as original code
    })
  }

  lambda_losses <- numeric(nlambda)
  lambda_index <- 1L
  resultsMatrix <- matrix(nrow = nlambda * nfolds, ncol = nrow(A) + 3L)

  if (warm_start == TRUE) {
    gamma_init <- abs(stats::rnorm(n = nrow(A), mean = 1, sd = 1))
  } else {
    gamma_init <- NULL
  }

  if (!lambdaSpecified) {
    # doing CV

    # If max_lambda unspecified, do while loop to get it
    if (is.null(max_lambda)) {
      cat("Finding max_lambda for CV...\n")
      gamma_init_copy <- gamma_init
      all_gammas_zero <- FALSE
      lambda_try <- 1e-10 # start with a number, then exponentiate by 2 iteratively to get to max_lambda
      while (!all_gammas_zero & lambda_try < 1e30) {
        cat("Trying lambda = ", lambda_try, ".\n")
        model_try <- krtexas_fixed_lambda(
          X = X,
          Y = Y,
          A = A,
          kernel = kernel,
          lambda = lambda_try,
          alpha = alpha,
          eps = eps,
          gamma_threshold = gamma_threshold,
          silent = silent,
          gamma_init = gamma_init_copy,
          adaptive_weights = adaptive_weights,
          cap_loss = FALSE,
          gamma_init_strat = gamma_init_strat
        )
        gammas_try <- model_try$gammas
        if (any(gammas_try != 0)) {
          # if any gammas are still nonzero
          lambda_try <- lambda_try * 2
        } else {
          all_gammas_zero <- TRUE
        }

        gamma_init_copy <- gammas_try
        gamma_init_copy[gamma_init_copy == 0] <- 1e-2
      } # while

      max_lambda <- lambda_try
      cat("max_lambda found: ", max_lambda, "\n")
    } # if max_lambda null

    if (min_lambda == 0) {
      lambda_seq <- c(
        0,
        exp(seq(log(1e-15), log(max_lambda), length.out = nlambda - 1))
      )
    } else {
      lambda_seq <- exp(seq(log(min_lambda), log(max_lambda), length.out = nlambda))
    }

    for (lambda in lambda_seq) {
      cat("============================================\n")
      cat("Processing lambda =", lambda, "...\n")

      # Export lambda to cluster for this iteration
      if (parallel) {
        parallel::clusterExport(cl, "lambda", envir = environment())
      }

      if (parallel) {
        # PARALLEL VERSION - Process all folds for this lambda in parallel
        fold_results <- foreach::foreach(
          fold = 1:nfolds,
          .combine = rbind,
          .packages = c("Matrix", "lbfgs", "numDeriv")
        ) %dopar% {

          train_indices <- which(folds != fold)
          test_indices <- which(folds == fold)

          X_train <- X[train_indices, , drop = FALSE]
          Y_train <- Y[train_indices]
          X_test <- X[test_indices, , drop = FALSE]
          Y_test <- Y[test_indices]

          if (nfolds == 1) {
            X_train <- X_test
            Y_train <- Y_test
          }

          # Fit the model
          fit <- krtexas_fixed_lambda(
            X = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            lambda = lambda,
            alpha = alpha,
            eps = eps,
            gamma_threshold = gamma_threshold,
            silent = TRUE, # Force silent in parallel
            gamma_init = gamma_init,
            adaptive_weights = adaptive_weights,
            cap_loss = TRUE,
            gamma_init_strat = gamma_init_strat
          )

          # Make predictions
          y_pred <- krtexas_predict_internal(
            gammas_test = fit$gammas,
            X_test = X_test,
            X_train = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            sigma = fit$sigma_opt
          )

          # Compute loss
          residuals <- Y_test - y_pred
          loss <- sum(residuals^2)

          list(
            fold = fold,
            loss = loss,
            avg_loss = loss / length(Y_test),
            fit_results = c(
              fit$gammas,
              fit$convergence,
              fit$training_loss,
              fit$sigma_opt
            )
          )
        }#dopar

        # Process parallel results
        lambda_loss <- mean(sapply(fold_results[, "loss"], as.numeric))

        # Fill results matrix
        start_row <- (lambda_index - 1L) * nfolds + 1L
        end_row <- lambda_index * nfolds
        for (i in seq_len(nfolds)) {
          resultsMatrix[start_row + i - 1L, ] <- fold_results[[i, "fit_results"]]
        }

        #cat("Average CV loss for lambda =", lambda, ":", lambda_loss, "\n")

        if (warm_start == TRUE) {
          # Processing Gammas to get new gamma_init
          gamma_fits <- do.call(
            rbind,
            lapply(
              1:nfolds,
              function(i) {
                as.numeric(fold_results[[i, "fit_results"]][1:nrow(A)])
              }
            )
          )
          gamma_init <- colMeans(gamma_fits, na.rm = TRUE)
          fit_convergences <- sapply(
            1:nfolds,
            function(i) {
              as.numeric(fold_results[[i, "fit_results"]][nrow(A) + 1L])
            }
          )
          #cat("convergences on the n folds :", fit_convergences, "\n")
          #cat("gamma results (averaged): <", gamma_init, ">\n")
          gamma_init[gamma_init == 0] <- 1e-2
          gamma_init[is.na(gamma_init)] <- stats::rnorm(n = 1, mean = 1, sd = 1)

          sigma_fits <- sapply(
            1:nfolds,
            function(i) {
              as.numeric(fold_results[[i, "fit_results"]][nrow(A) + 3L])
            }
          )
        } # warm_start
      } else {
        # SEQUENTIAL VERSION
        lambda_loss <- 0
        for (fold in 1:nfolds) {
          resultRow <- (lambda_index - 1L) * nfolds + fold
          cat("Executing fold", fold, "/", nfolds, "for lambda =", lambda, "...\n")

          train_indices <- which(folds != fold)
          test_indices <- which(folds == fold)

          X_train <- X[train_indices, , drop = FALSE]
          Y_train <- Y[train_indices]
          X_test <- X[test_indices, , drop = FALSE]
          Y_test <- Y[test_indices]

          if (nfolds == 1) {
            X_train <- X_test
            Y_train <- Y_test
          }

          # Fit the model
          fit <- krtexas_fixed_lambda(
            X = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            lambda = lambda,
            alpha = alpha,
            eps = eps,
            gamma_threshold = gamma_threshold,
            silent = silent,
            adaptive_weights = adaptive_weights,
            cap_loss = TRUE,
            gamma_init_strat = gamma_init_strat
          )

          resultsMatrix[resultRow, ] <- c(
            fit$gammas,
            fit$convergence,
            fit$training_loss,
            fit$sigma_opt
          )

          # Make predictions
          y_pred <- krtexas_predict_internal(
            gammas_test = fit$gammas,
            X_test = X_test,
            X_train = X_train,
            Y = Y_train,
            A = A,
            kernel = kernel,
            sigma = fit$sigma_opt
          )

          # Compute loss
          residuals <- Y_test - y_pred
          loss <- sum(residuals^2)
          lambda_loss <- lambda_loss + loss / nfolds
        }
        #cat("Average CV loss for lambda =", lambda, ":", lambda_loss, "\n")
      }

      lambda_losses[lambda_index] <- lambda_loss
      lambda_index <- lambda_index + 1L
    } # for lambda

    # Clean up parallel cluster
    if (parallel && !lambdaSpecified) {
      parallel::stopCluster(cl)
      foreach::registerDoSEQ()
    }

    # Find optimal lambda
    #lambda_opt <- lambda_seq[which.min(lambda_losses)]
    finite <- is.finite(lambda_losses)
    if (!any(finite)) {
      stop("All CV losses were non-finite (NA/NaN/Inf). Cannot choose lambda. Try parallel = FALSE once and/or cap your lambda range.")
    }
    best_idx <- which(finite)[ which.min(lambda_losses[finite]) ]
    lambda_opt <- lambda_seq[best_idx]

    loss_opt <- lambda_losses[which.min(lambda_losses)]
  } else {
    lambda_opt <- lambda
  }

  # Train final model with optimal lambda
  cat("Training model with lambda =", lambda_opt, "...\n")
  #gamma_init <- rep(1e-3, nrow(A))
  gamma_init <- NULL

  lambda_opt_fit <- krtexas_fixed_lambda(
    X = X,
    Y = Y,
    A = A,
    kernel = kernel,
    lambda = lambda_opt,
    alpha = alpha,
    eps = eps,
    gamma_threshold = gamma_threshold,
    silent = silent,
    gamma_init = gamma_init,
    adaptive_weights = adaptive_weights,
    cap_loss = TRUE,
    gamma_init_strat = gamma_init_strat
  )

  list(
    gammas_learned = as.vector(abs(lambda_opt_fit$gammas)),
    lambda.best = lambda_opt,
    sigma_opt = lambda_opt_fit$sigma_opt,
    training_loss = lambda_opt_fit$training_loss,
    cv_lambda_losses = lambda_losses,
    convergence = lambda_opt_fit$convergence,
    inputs = list(X = X, Y = Y, A = A, kernel = kernel)
  )
} # krtexas_fit_internal




#' Internal solver for fixed lambda
#'
#' @keywords internal
krtexas_fixed_lambda <- function(X = NULL,
                                 Y = NULL,
                                 A = NULL,
                                 kernel = "gaussian",
                                 lambda = 0,
                                 alpha = NULL,
                                 eps = 1e-2,
                                 gamma_threshold,
                                 use_num_grad = FALSE,
                                 silent = TRUE,
                                 gamma_init = NULL,
                                 adaptive_weights = NULL,
                                 cap_loss = TRUE,
                                 gamma_init_strat = "smallest",
                                 developer_code = 0) {
  ## ---------------------------
  ## Basic checks
  ## ---------------------------
  if (!is.matrix(A)) {
    stop("Error: The object 'A' is not a matrix.")
  }
  if (!kernel %in% c("gaussian", "epanechnikov")) {
    stop("Error: 'kernel' must be either 'gaussian' or 'epanechnikov'.")
  }
  if (is.null(X) || is.null(Y)) {
    stop("Error: 'X' and 'Y' must be provided.")
  }
  if (lambda < 0) {
    cat("lambda = ", lambda, "\n.")
    stop("Error: 'lambda' must be nonnegative.")
  }

  tau   <- nrow(A)
  kappa <- lambda / 2
  sigma <- 1  # FIXED
  n = nrow(X)

  cat("*")
  if (is.null(gamma_init)) {
    if(gamma_init_strat == "small"){
      init_u <- stats::rnorm(n = tau, mean = 1, sd = 1/4)
      init_w <- stats::rnorm(n = tau, mean = 1, sd = 1/4)
    } else if(gamma_init_strat == "large"){
      ### Updated ###
      init_u <- stats::rnorm(n = tau, mean = max(1, n^(2/(4+tau))), sd = 1)
      init_w <- stats::rnorm(n = tau, mean = max(1, n^(2/(4+tau))), sd = 1)
    } else if(gamma_init_strat == "smallest"){
      ### Updated ###
      starting_gamma <- abs( rnorm(n=tau, mean = 0.1, sd = 0.01) )
      cat("In krtexas_fixed_lambda - Using smallest gamma initalization of starting_gamma[1:20]: ", starting_gamma[1:20], "\n")
      init_u <- sqrt(starting_gamma)
      init_w <- sqrt(starting_gamma)
    }

    theta_init <- abs(c(init_u, init_w))
  } else {
    init_u    <- rep(1, tau)
    init_w    <- gamma_init
    theta_init <- abs(c(init_u, init_w))
  }

  if (ncol(A) != nrow(t(X))) {
    stop("A Matrix and X Matrix Mismatch. ncol(A) should equal ncol(X).")
  } else {
    A_sparse <- Matrix::Matrix(A, sparse = TRUE)
    Ax <- as.matrix(A_sparse %*% t(X))
  }
  cat("**")

  ## ---------------------------
  ## Knock out gamma indices with impossibly high lambda*w's
  ## ---------------------------
  B_x  <- max(apply(X, 2, function(col) max(col) - min(col)))
  B_y  <- max(Y) - min(Y)
  B_xy <- max(B_x, B_y)
  lambda_w <- lambda * adaptive_weights
  gamma_zero_idx <- which(lambda_w > 32 * B_xy^4)
  gamma_zero_idx <- c(gamma_zero_idx, gamma_zero_idx + tau)

  #cat("32*B_xy^4: ", 32 * B_xy^4, "\n")
  #cat("lambda_w: ", lambda_w, "\n")
  #cat("gamma_zero_idx: ", gamma_zero_idx, "\n")

  upper_bound <- rep(Inf, length(theta_init))
  upper_bound[gamma_zero_idx] <- 0

  ## ---------------------------
  ## Loss / gradient wrappers (fixed sigma)
  ## ---------------------------
  if (kernel == "gaussian") {

    if (silent) {
      tmpfile <- tempfile()
      did_sink <- FALSE
      try({
        sink(tmpfile)
        did_sink <- TRUE
      }, silent = TRUE)
    }

    # Single function that returns list(value, gradient) for splitfngr
    fngr_wrapper <- function(theta) {
      u <- theta[1:tau]
      w <- theta[(tau + 1):(2 * tau)]

      # Function value
      loss_grad_obj <- loss_grad_fun(
        u, w, Ax, X, Y,
        sigma = sigma,
        kappa = kappa,
        alpha = alpha,
        adaptive_weights = adaptive_weights
      )

      loss_val <- loss_grad_obj$loss
      grad_val <- loss_grad_obj$grad

      # If value is bad, give a large finite penalty + zero gradient
      if (!is.finite(loss_val) || any(!is.finite(grad_val))) {#if (is.na(loss_grad_loss) || !is.finite(loss_grad_loss)) {
        cat("NaN or Inf Loss. Returning 1e20 with 0 gradient. \n")
        cat("theta: ", theta, "\n")
        cat("loss_val: ", loss_val, ". grad_val: ", grad_val,".\n")

        gamma <- u * w

        # Deterministic “probe point”: nearest neighbor in Ax-space for quick weight check
        # (No RNG use inside the objective.)
        deltas   <- sweep(Ax, 1, Ax[, 1L], FUN = "-")  # (M x n)
        dists_sq  <- colSums((deltas * sqrt(pmax(gamma, 0)))^2)
        j_nn      <- which.min(dists_sq)
        deltas_j <- sweep(Ax, 1, Ax[, j_nn], FUN = "-")
        d2_j      <- colSums((deltas_j * sqrt(pmax(gamma, 0)))^2)
        d_min     <- min(d2_j)
        weights_stab <- exp(-(d2_j - d_min) / (2 * sigma^2))
        w_sum_stab   <- sum(weights_stab)

        #Build compact snapshot (avoid huge vectors)
        snap <- list(
          when            = Sys.time(),
          pid             = Sys.getpid(),
          lambda          = lambda,
          tau             = tau,
          n               = nrow(X),
          theta_range     = range(theta, finite = TRUE),
          u_head          = head(u),
          w_head          = head(w),
          gamma_min       = suppressWarnings(min(gamma)),
          gamma_max       = suppressWarnings(max(gamma)),
          gamma_any_na    = any(is.na(gamma)),
          X_range         = range(X, finite = TRUE),
          Y_range         = range(Y, finite = TRUE),
          Ax_is_finite    = all(is.finite(Ax)),
          loss_val        = loss_val,
          grad_any_nonfin = any(!is.finite(grad_val)),
          d_min           = d_min,
          w_sum_stab      = w_sum_stab,
          dists_sq_head   = head(dists_sq)
        )#snap

        # Write only first few snapshots to disk to avoid flooding
        # Use a per-process counter to keep files small and unique.
        ctr_name <- ".krtexas_debug_counter"
        if (!exists(ctr_name, envir = .GlobalEnv, inherits = FALSE)) {
          assign(ctr_name, 0L, envir = .GlobalEnv)
        }
        cnt <- get(ctr_name, envir = .GlobalEnv, inherits = FALSE)
        if (cnt < 5L) {
          f <- tempfile(sprintf("krtexas_bad_eval_%03d_", cnt + 1L), fileext = ".rds")
          try(saveRDS(snap, f), silent = TRUE)
          assign(ctr_name, cnt + 1L, envir = .GlobalEnv)
        }

        # Penalize: large loss + zero gradient to steer optimizer away
        if (!is.finite(loss_val)) {
          cat("NaN or Inf loss detected. Penalizing with 1e20 and zero gradient.\n")
        } else if (any(!is.finite(grad_val))) {
          cat("Analytical gradient has NaNs/Infs. Penalizing with 1e20 and zero gradient.\n")
        }

        loss_val <- 1e20
        grad_val <- rep(0, length(theta))


      }# if loss NA or not finite (debugging chunk, only active when code is failing)

      list(loss_val, grad_val)
    }# fngr_wrapper

    result <- splitfngr::optim_share(
      par  = theta_init,
      fngr = fngr_wrapper,
      method  = "L-BFGS-B",
      lower   = rep(0, length(theta_init)),
      upper   = upper_bound,
      control = list(factr = eps / .Machine$double.eps)
    )

    if (silent && exists("did_sink", inherits = FALSE) && did_sink && sink.number() > 0) {
      sink()
      unlink(tmpfile)
    }

  } else if (kernel == "epanechnikov") {
    stop("Epanechnikov kernel training not implemented in this minimal version.")
  }

  ## ---------------------------
  ## Extract solution and post-process
  ## ---------------------------
  u_opt <- matrix(result$par[1:tau],             nrow = 1)
  w_opt <- matrix(result$par[(tau + 1):(2 * tau)], nrow = 1)

  gammas_learned <- u_opt * w_opt
  gammas_learned[which(abs(gammas_learned) < gamma_threshold * sigma)] <- 0

  convergence <- result$convergence

  #cat("gammas_learned: ", gammas_learned, "\n")
  #cat("Message: ", result$message, "\n")
  #cat("Value: ", result$value, "\n")

  if (result$value > 1e20 && cap_loss) {
    stop("Value > 1e20. Returning Inf.")
  }

  if (convergence != 0) {
    warning(paste0(
      "Solution is not guaranteed to be optimal. At lambda = ",lambda,
      ". L-BFGS convergence code: ", convergence, "."
    ))
  }

  list(
    gammas        = gammas_learned,
    training_loss = result$value,
    convergence   = convergence,
    sigma_opt     = sigma,  # always 1,
    init_u = init_u,
    init_w = init_w
  )
} # krtexas_fixed_lambda


#' Internal prediction function used by CV / fixed-lambda
#'
#' @keywords internal
krtexas_predict_internal <- function(gammas_test,
                                     X_test,
                                     X_train,
                                     Y,
                                     A,
                                     kernel,
                                     sigma) {
  gammas_test <- as.vector(gammas_test)

  if (kernel == "gaussian") {
    n_train <- nrow(X_train)
    n_test  <- nrow(X_test)

    Ax_train <- A %*% t(X_train)
    Ax_test  <- A %*% t(X_test)

    yhat_test <- numeric(n_test)

    for (i in 1:n_test) {
      deltas   <- Ax_train - Ax_test[, i]
      dists_sq <- colSums((deltas * sqrt(gammas_test))^2)

      ## Stabilized weights: subtract the minimum distance (exactly equivalent)
      d_min   <- min(dists_sq)
      weights <- exp(-(dists_sq - d_min) / (2 * sigma^2))

      w_sum <- sum(weights)
      yhat_test[i] <- sum(weights * Y) / w_sum
    }

    return(yhat_test)
  }

  stop("Unknown kernel in krtexas_predict_internal.")
} # krtexas_predict_internal


#' Predict from a KR TEXAS model
#'
#' @param krtexas_model Model object returned by \code{krtexas_fit()}.
#' @param newx New data matrix with same number of columns as training X.
#'
#' @return Numeric vector of predictions.
#' @export
krtexas_predict <- function(krtexas_model, newx) {
  gammas_test <- krtexas_model$gammas_learned
  sigma <- 1

  X_train <- krtexas_model$inputs$X
  Y_train <- krtexas_model$inputs$Y
  A <- krtexas_model$inputs$A
  kernel <- krtexas_model$inputs$kernel
  X_test <- newx

  if (ncol(X_test) != ncol(X_train)) {
    stop("Error: The number of columns in 'newx' must match the number of columns in training 'X'.")
  }

  gammas_test <- as.vector(gammas_test)

  if (kernel == "gaussian") {
    n_train <- nrow(X_train)
    n_test <- nrow(X_test)

    Ax_train <- A %*% t(X_train)
    Ax_test  <- A %*% t(X_test)

    yhat_test <- numeric(n_test)

    for (i in 1:n_test) {
      deltas   <- Ax_train - Ax_test[, i]
      dists_sq <- colSums((deltas * sqrt(gammas_test))^2)

      ## ---- Stabilized weights: subtract min distance (exactly equivalent) ----
      d_min   <- min(dists_sq)
      weights <- exp(-(dists_sq - d_min) / (2 * sigma^2))

      w_sum <- sum(weights)
      if (w_sum == 0 || is.nan(w_sum)) {
        cat(sprintf("Test %d: sum(weights)=0! NaN will result.\n", i))
        cat("sigma: ", sigma, "\n")
        cat("d_min: ", d_min, "\n")
        cat("head(weights):", head(weights), "\n")
        cat("head(dists_sq):", head(dists_sq), "\n")
        cat("Y_train[1:5]:", head(Y_train), "\n")
        cat("gammas_test[1:5]:", head(gammas_test), "\n")
        cat("RowSums of A (first 10):", head(rowSums(abs(A))), "\n")
      }

      yhat_test[i] <- sum(weights * Y_train) / w_sum
    }

    return(yhat_test)
  }

  stop("Unknown kernel in krtexas_predict.")
}# krtexas_predict

### Gaussian Kernel ###

#' Gaussian kernel loss and gradient function together (internal)
#'
#' @keywords internal
#' Gaussian kernel loss and gradient function (internal, C++ wrapper)
#'
#' @keywords internal
loss_grad_fun <- function(u,
                          w,
                          Ax,
                          X,
                          Y,
                          sigma,
                          kappa,
                          alpha,
                          adaptive_weights) {
  loss_grad_rcpp(u, w, Ax, X, Y, sigma, kappa, alpha, adaptive_weights)
}# loss grad fun


#' Gaussian kernel loss function (internal)
#'
#' @keywords internal
#' Gaussian kernel loss function (internal, C++ wrapper)
#'
#' @keywords internal
loss_fun <- function(u,
                     w,
                     Ax,
                     X,
                     Y,
                     sigma,
                     kappa,
                     alpha,
                     adaptive_weights) {
  loss_fun_fast_cpp(u, w, Ax, X, Y, sigma, kappa, alpha, adaptive_weights)
}# loss fun


#' Gaussian kernel gradient (internal, C++ wrapper)
#'
#' @keywords internal
grad_fun <- function(u,
                     w,
                     Ax,
                     X,
                     Y,
                     sigma,
                     kappa,
                     adaptive_weights) {
  grad_fun_fast_cpp(u, w, Ax, X, Y, sigma, kappa, adaptive_weights)
} #grad_fun


#' Epanechnikov kernel loss (internal)
#'
#' @keywords internal
loss_fun_ep <- function(u,
                        w,
                        Ax,
                        X,
                        Y,
                        sigma,
                        kappa,
                        alpha) {
  gamma <- u * w

  n <- nrow(X)
  D <- matrix(0, n, n)

  for (i in 1:(n - 2L)) {
    D[i, (i + 1L):n] <- colSums(((Ax[, (i + 1L):n] - Ax[, i]) * sqrt(gamma))^2)
  }

  D[n - 1L, n] <- sum(((Ax[, n - 1L] - Ax[, n]) * sqrt(gamma))^2)
  D <- D + t(D)

  nY <- length(Y)
  yhat <- numeric(nY)

  dydg <- matrix(0, nY, length(gamma))
  dy2dg <- matrix(0, nY, length(gamma))

  for (i in 1:nY) {
    epDi <- pmax(1 - (D[i, -i] / (sigma^2)), 0)

    if (length(epDi) - length(Y[-i]) != 0) {
      cat("length(epDi) - length(Y[-i]): ", length(epDi) - length(Y[-i]), "\n")
    }

    sumEp_d_Y <- sum(epDi * Y[-i])
    sumEp_d <- sum(epDi)

    d_epD_d_gamma_i <- -1 * t(t((Ax[, -i] - Ax[, i])^2)) / sigma^2
    d_epD_d_gamma_i[, which(epDi == 0)] <- 0

    if (sumEp_d == 0) {
      cat("sumEp_d==0. Try increasing sigma.\n")
      cat(summary(epDi))
      stop("Increase Sigma.")
    }

    yhat[i] <- sumEp_d_Y / sumEp_d
  }

  loss <- sum((Y - yhat)^2)

  if (!is.finite(loss)) {
    cat("Yhat:", yhat, "\n")
  }

  spred_penalty <- kappa * (sum(u^2) + sum(w^2))
  loss / nY + spred_penalty + (alpha / sigma^2)
} # loss_fun_ep


#' Epanechnikov kernel gradient (internal)
#'
#' @keywords internal
grad_fun_ep <- function(u,
                        w,
                        Ax,
                        X,
                        Y,
                        sigma,
                        kappa) {
  gamma <- u * w

  n <- nrow(X)
  D <- matrix(0, n, n)

  for (i in 1:(n - 2L)) {
    D[i, (i + 1L):n] <- colSums(((Ax[, (i + 1L):n] - Ax[, i]) * sqrt(gamma))^2)
  }
  D[n - 1L, n] <- sum(((Ax[, n - 1L] - Ax[, n]) * sqrt(gamma))^2)
  D <- D + t(D)

  nY <- length(Y)
  yhat <- numeric(nY)

  dydg <- matrix(0, nY, length(gamma))
  dy2dg <- matrix(0, nY, length(gamma))

  for (i in 1:nY) {
    epDi <- pmax(1 - (D[i, -i] / (sigma^2)), 0)

    sumEp_d_Y <- sum(epDi * Y[-i])
    sumEp_d <- sum(epDi)

    d_epD_d_gamma_i <- -1 * t(t((Ax[, -i] - Ax[, i])^2)) / sigma^2
    d_epD_d_gamma_i[, which(epDi == 0)] <- 0

    yhat[i] <- sumEp_d_Y / sumEp_d

    dydg[i, ] <- (d_epD_d_gamma_i %*% Y[-i] * sumEp_d -
                    rowSums(d_epD_d_gamma_i) * sumEp_d_Y) / (sumEp_d^2)

    dy2dg[i, ] <- 2 * yhat[i] * dydg[i, ]
  }

  gradient_vector <- colSums(dy2dg - 2 * dydg * Y)

  dthetadu <- gradient_vector * w
  dthetadw <- gradient_vector * u

  nY <- length(Y)
  c(dthetadu, dthetadw) / nY + kappa * 2 * c(u, w)
} # grad_fun_ep


#' Internal: gradient wrt leaves for a single newx
#'
#' @keywords internal
get_gradients_wrtleaves_internal <- function(krtexas_model, newx) {
  newx <- t(newx)
  gammas_test <- as.numeric(krtexas_model$gammas_learned)
  sigma <- krtexas_model$sigma_opt

  X_train <- krtexas_model$inputs$X
  Y_train <- krtexas_model$inputs$Y
  A <- krtexas_model$inputs$A
  kernel <- krtexas_model$inputs$kernel

  X_test <- as.matrix(newx)
  if (!is.matrix(X_test) || nrow(X_test) != 1L) {
    cat("str(as.matrix(newx)):")
    str(as.matrix(newx))
    stop("Error: 'newx' must be a matrix with a single row (1 x p).")
  }
  if (ncol(X_test) != ncol(X_train)) {
    stop("Error: number of columns in 'newx' must match training X (p).")
  }

  p <- ncol(X_train)
  n_train <- nrow(X_train)

  Ax_train <- A %*% t(X_train)
  Ax_test <- A %*% t(X_test)

  if (nrow(Ax_train) != nrow(Ax_test)) {
    stop("Internal shape error: Ax_train and Ax_test have different numbers of rows.")
  }

  if (kernel == "gaussian") {
    deltas <- Ax_train - Ax_test[, rep(1, n_train), drop = FALSE]

    if (length(gammas_test) != nrow(deltas)) {
      stop("Length of gammas_test does not match number of features after A multiplication.")
    }

    scale_vec <- sqrt(gammas_test)
    scale_mat <- matrix(scale_vec, nrow = length(scale_vec), ncol = ncol(deltas))
    dists_sq <- colSums((deltas * scale_mat)^2)

    #weights <- exp(-dists_sq / (2 * sigma^2))
    #w_sum <- sum(weights)
    d_min <- min(dists_sq)                       # <= key line
    weights <- exp(-(dists_sq - d_min) / (2 * sigma^2))
    w_sum <- sum(weights)


    if (w_sum == 0 || is.nan(w_sum)) {
      stop("Sum of kernel weights is zero or NaN for the provided newx. Returning NA gradient.")
    }

    yhat <- sum(weights * Y_train) / w_sum

    dfdx <- numeric(p)
    for (t in seq_len(p)) {
      dfdxpart1 <- -gammas_test[t] / (sigma^2 * w_sum)
      dfdxpart2 <- 0
      for (j in seq_len(n_train)) {
        dfdxpart2 <- dfdxpart2 +
          weights[j] *
          (X_test[1, t] - X_train[j, t]) *
          (Y_train[j] - yhat)
      }
      dfdx[t] <- dfdxpart1 * dfdxpart2
    }

    return(dfdx)
  } else if (kernel == "epanechnikov") {
    stop("Epanechnikov implementation not developed yet.")
  } else {
    stop("Unknown kernel specified in krtexas_model$inputs$kernel.")
  }
} # get_gradients_wrtleaves_internal


#' Internal: get siblings of a row in A
#'
#' @keywords internal
get_siblings_m_internal <- function(A, m, depths) {
  M <- nrow(A)
  p <- ncol(A)

  row_m <- A[m, ]
  depth_m <- depths[m]

  all_ones <- rep(1, p)

  other_rows <- setdiff(which(depths == depth_m), m)

  if (length(other_rows) == 0) {
    #cat("No other rows at depth", depth_m, "\n")
    return(integer(0))
  }

  valid_sets <- list()

  is_subset <- function(set1, set2) {
    all(set1 %in% set2)
  }

  is_superset_of_valid <- function(selected, valid_sets) {
    for (valid_set in valid_sets) {
      if (is_subset(valid_set, selected) && length(valid_set) < length(selected)) {
        return(TRUE)
      }
    }
    FALSE
  }

  check_condition <- function(row_indices) {
    if (length(row_indices) == 0) {
      result <- row_m
    } else {
      result <- row_m
      for (idx in row_indices) {
        result <- (result + A[idx, ]) %% 2
      }
    }

    if (all(result == all_ones)) {
      return(TRUE)
    }

    for (i in 1:M) {
      if (i == m) next
      if (all(result == A[i, ]) && all(row_m <= A[i, ])) {
        return(TRUE)
      }
    }

    FALSE
  }

  n <- length(other_rows)
  limit <- 2^n - 1L

  i = 0
  while(i <= (limit - 1)){
    i = i + 1
    if(i %% 1000L == 0)
      cat("\ri = ", i, " limit = ", limit, " || ", (i/limit)*100, "%")
    binary <- as.integer(intToBits(i)[1:n])
    selected <- other_rows[which(binary == 1L)]

    if (is_superset_of_valid(selected, valid_sets)) {
      next
    }

    if (check_condition(selected)) {
      valid_sets[[length(valid_sets) + 1L]] <- selected
    }
  }# while

  if (length(valid_sets) > 0) {
    siblings_m <- valid_sets[[which.min(lengths(valid_sets))]]
    return(siblings_m)
  }

  integer(0)
} # get_siblings_m_internal


#' Internal: compute depths of rows of A from root
#'
#' @keywords internal
get_row_depths_from_root_internal <- function(A) {
  M <- nrow(A)
  p <- ncol(A)

  depths <- rep(-1L, M)

  is_root <- sapply(
    1:M,
    function(i) {
      !any(sapply(
        1:M,
        function(j) {
          if (i == j) return(FALSE)
          all(A[j, ] >= A[i, ]) && !all(A[j, ] == A[i, ])
        }
      ))
    }
  )

  depths[is_root] <- 0L

  while (any(depths == -1L)) {
    unassigned <- which(depths == -1L)
    any_assigned_this_round <- FALSE

    for (i in unassigned) {
      all_parents <- which(sapply(
        1:M,
        function(j) {
          if (i == j) return(FALSE)
          all(A[j, ] >= A[i, ]) && !all(A[j, ] == A[i, ])
        }
      ))

      if (length(all_parents) == 0) next

      immediate_parent <- all_parents[which.min(rowSums(A[all_parents, , drop = FALSE]))]

      if (depths[immediate_parent] != -1L) {
        depths[i] <- depths[immediate_parent] + 1L
        any_assigned_this_round <- TRUE
      }
    }

    if (!any_assigned_this_round) {
      warning("Could not assign all depths - possible issue with tree structure")
      break
    }
  }

  depths
} # get_row_depths_from_root_internal


#' Internal: compute adaptive penalties C1, C2, C3
#'
#' @keywords internal
get_adaptive_penalties_internal_NWML_L2 <- function(krtexas_model, A, int_ind_ml) {
  X_train <- krtexas_model$inputs$X
  n <- nrow(X_train)
  p <- ncol(X_train)
  M <- nrow(A)

  grad_mat <- matrix(NA_real_, nrow = n, ncol = p)

  for (i in seq_len(n)) {
    grad_mat[i, ] <- get_gradients_wrtleaves_internal(
      krtexas_model,
      t(as.matrix(X_train[i, , drop = FALSE]))
    )
  }

  grad_mat <- grad_mat[int_ind_ml, , drop = FALSE]

  #utils::View(grad_mat)

  C1 <- numeric(M)
  C2 <- numeric(M)
  C3 <- numeric(M)

  cat("Generating sibling matrix...")
  sibling_mat <- matrix(0, nrow = M, ncol = M)

  # Find parent for each node
  parent <- rep(NA_integer_, M)

  for (i in seq_len(M)) {
    # Find all nodes that are ancestors of i (i.e., A[j,] >= A[i,])
    potential_parents <- which(sapply(
      seq_len(M),
      function(j) {
        if (i == j) return(FALSE)
        all(A[j, ] >= A[i, ]) && !all(A[j, ] == A[i, ])
      }
    ))

    if (length(potential_parents) > 0) {
      # Parent is the ancestor with minimum row sum (closest ancestor)
      parent[i] <- potential_parents[which.min(rowSums(A[potential_parents, , drop = FALSE]))]
    }
  }

  # Group children by parent (including NA for root's children)
  parent_factor <- factor(parent, levels = c(NA, seq_len(M)), exclude = NULL)
  children_by_parent <- split(seq_len(M), parent_factor)

  # Build sibling matrix
  for (parent_group in children_by_parent) {
    if (length(parent_group) > 1) {
      # All children of same parent are siblings of each other
      for (i in seq_along(parent_group)) {
        siblings <- parent_group[-i]  # All children except current one
        sibling_mat[parent_group[i], siblings] <- 1
      }
    }
  }

  cat(" Sibling matrix generated. \n")
  for (i in seq_len(nrow(sibling_mat))) {
    siblings <- which(sibling_mat[i, ] == 1)
    #cat(sprintf("Node %3d has siblings: %s\n",
    #            i,
    #            if(length(siblings) > 0) paste(siblings, collapse = ", ") else "none"))
  }

  for (m in seq_len(M)) {
    leaves_m <- which(A[m, ] != 0)
    if (length(leaves_m) == 0) {
      C1[m] <- 0
      C2[m] <- 0
      C3[m] <- 0
      next
    }
    grad_mat_sub_m <- grad_mat[, leaves_m, drop = FALSE]

    C2[m] <- sum(colMeans(grad_mat_sub_m^2)) / length(leaves_m) #sum((1 / n) * colSums(grad_mat_sub_m^2))

    cols <- ncol(grad_mat_sub_m)
    if (cols < 2L) {
      C1[m] <- 0
    } else {
      pairs <- combn(cols, 2, simplify = FALSE)
      s <- 0
      for (pair in pairs) {
        diff <- grad_mat_sub_m[, pair[1]] - grad_mat_sub_m[, pair[2]]
        s <- s + mean(diff^2)
      }
      C1[m] <- s / length(pairs)
    }

    siblings_m <- which(sibling_mat[m, ] == 1)
    if (length(siblings_m) == 0) {
      saveRDS(sibling_mat, "sibling_mat")
      C3[m] <- 0
      #utils::View(A)
      stop("C3 length 0.")
    } else {
      sib_mats <- lapply(
        siblings_m,
        function(k) {
          leaves_k <- which(A[k, ] != 0)
          grad_mat[, leaves_k, drop = FALSE]
        }
      )

      p_m <- ncol(grad_mat_sub_m)
      num_pairs_C3 <- sum(sapply(sib_mats, ncol) * p_m)

      if (num_pairs_C3 == 0) {
        C3[m] <- 0
      } else {
        s3 <- 0
        for (j in seq_len(p_m)) {
          col_m <- grad_mat_sub_m[, j]
          for (mat_k in sib_mats) {
            if (ncol(mat_k) == 0) next
            for (l in seq_len(ncol(mat_k))) {
              col_k <- mat_k[, l]
              s3 <- s3 + mean((col_m - col_k)^2)
            }
          }
        }
        C3[m] <- s3 / num_pairs_C3
      }
    }
  }

  list(C1 = C1, C2 = C2, C3 = C3)
} # get_adaptive_penalties_internal_NWML_L2


#' Internal: compute adaptive penalties C1, C2, C3
#'
#' @keywords internal
get_adaptive_penalties_internal_LLR_L2 <- function(grad_llr, A, X, int_ind_ml) {
  X_train <- X
  n <- nrow(X_train)
  p <- ncol(X_train)
  M <- nrow(A)

  grad_mat <- grad_llr
  grad_mat <- grad_mat[int_ind_ml, , drop = FALSE] # subsetting to interior #NEW

  #utils::View(grad_mat)

  C1 <- numeric(M)
  C2 <- numeric(M)
  C3 <- numeric(M)

  cat("Generating sibling matrix...")
  sibling_mat <- matrix(0, nrow = M, ncol = M)

  # Find parent for each node
  parent <- rep(NA_integer_, M)

  for (i in seq_len(M)) {
    # Find all nodes that are ancestors of i (i.e., A[j,] >= A[i,])
    potential_parents <- which(sapply(
      seq_len(M),
      function(j) {
        if (i == j) return(FALSE)
        all(A[j, ] >= A[i, ]) && !all(A[j, ] == A[i, ])
      }
    ))

    if (length(potential_parents) > 0) {
      # Parent is the ancestor with minimum row sum (closest ancestor)
      parent[i] <- potential_parents[which.min(rowSums(A[potential_parents, , drop = FALSE]))]
    }
  }

  # Group children by parent (including NA for root's children)
  parent_factor <- factor(parent, levels = c(NA, seq_len(M)), exclude = NULL)
  children_by_parent <- split(seq_len(M), parent_factor)

  # Build sibling matrix
  for (parent_group in children_by_parent) {
    if (length(parent_group) > 1) {
      # All children of same parent are siblings of each other
      for (i in seq_along(parent_group)) {
        siblings <- parent_group[-i]  # All children except current one
        sibling_mat[parent_group[i], siblings] <- 1
      }
    }
  }

  cat(" Sibling matrix generated. \n")
  for (i in seq_len(nrow(sibling_mat))) {
    siblings <- which(sibling_mat[i, ] == 1)
    #cat(sprintf("Node %3d has siblings: %s\n",
    #            i,
    #            if(length(siblings) > 0) paste(siblings, collapse = ", ") else "none"))
  }

  for (m in seq_len(M)) {
    leaves_m <- which(A[m, ] != 0)
    if (length(leaves_m) == 0) {
      C1[m] <- 0
      C2[m] <- 0
      C3[m] <- 0
      next
    }

    grad_mat_sub_m <- grad_mat[, leaves_m, drop = FALSE]

    C2[m] <- sum(colMeans(grad_mat_sub_m^2)) / length(leaves_m)

    cols <- ncol(grad_mat_sub_m)
    if (cols < 2L) {
      C1[m] <- 0
    } else {
      pairs <- combn(cols, 2, simplify = FALSE)
      s <- 0
      for (pair in pairs) {
        diff <- grad_mat_sub_m[, pair[1]] - grad_mat_sub_m[, pair[2]]
        s <- s + mean(diff^2)
      }
      C1[m] <- s / length(pairs)
    }

    siblings_m <- which(sibling_mat[m, ] == 1)
    if (length(siblings_m) == 0) {
      C3[m] <- 0
      #utils::View(A)
      stop("C3 length 0.")
    } else {
      sib_mats <- lapply(
        siblings_m,
        function(k) {
          leaves_k <- which(A[k, ] != 0)
          grad_mat[, leaves_k, drop = FALSE]
        }
      )

      p_m <- ncol(grad_mat_sub_m)
      num_pairs_C3 <- sum(sapply(sib_mats, ncol) * p_m)

      if (num_pairs_C3 == 0) {
        C3[m] <- 0
      } else {
        s3 <- 0
        for (j in seq_len(p_m)) {
          col_m <- grad_mat_sub_m[, j]
          for (mat_k in sib_mats) {
            if (ncol(mat_k) == 0) next
            for (l in seq_len(ncol(mat_k))) {
              col_k <- mat_k[, l]
              s3 <- s3 + mean((col_m - col_k)^2)
            }
          }
        }
        C3[m] <- s3 / num_pairs_C3
      }
    }
  }

  list(C1 = C1, C2 = C2, C3 = C3)
} # get_adaptive_penalties_internal_LLR_L2

#' Internal: compute adaptive penalties C1, C2, C3
#'
#' @keywords internal
get_adaptive_penalties_internal_NWML_L1 <- function(krtexas_model, A, int_ind_ml) {
  X_train <- krtexas_model$inputs$X
  n <- nrow(X_train)
  p <- ncol(X_train)
  M <- nrow(A)

  grad_mat <- matrix(NA_real_, nrow = n, ncol = p)

  for (i in seq_len(n)) {
    grad_mat[i, ] <- get_gradients_wrtleaves_internal(
      krtexas_model,
      t(as.matrix(X_train[i, , drop = FALSE]))
    )
  }

  grad_mat <- grad_mat[int_ind_ml, , drop = FALSE]

  #utils::View(grad_mat)

  C1 <- numeric(M)
  C2 <- numeric(M)
  C3 <- numeric(M)

  cat("Generating sibling matrix...")
  sibling_mat <- matrix(0, nrow = M, ncol = M)

  # Find parent for each node
  parent <- rep(NA_integer_, M)

  for (i in seq_len(M)) {
    # Find all nodes that are ancestors of i (i.e., A[j,] >= A[i,])
    potential_parents <- which(sapply(
      seq_len(M),
      function(j) {
        if (i == j) return(FALSE)
        all(A[j, ] >= A[i, ]) && !all(A[j, ] == A[i, ])
      }
    ))

    if (length(potential_parents) > 0) {
      # Parent is the ancestor with minimum row sum (closest ancestor)
      parent[i] <- potential_parents[which.min(rowSums(A[potential_parents, , drop = FALSE]))]
    }
  }

  # Group children by parent (including NA for root's children)
  parent_factor <- factor(parent, levels = c(NA, seq_len(M)), exclude = NULL)
  children_by_parent <- split(seq_len(M), parent_factor)

  # Build sibling matrix
  for (parent_group in children_by_parent) {
    if (length(parent_group) > 1) {
      # All children of same parent are siblings of each other
      for (i in seq_along(parent_group)) {
        siblings <- parent_group[-i]  # All children except current one
        sibling_mat[parent_group[i], siblings] <- 1
      }
    }
  }

  cat(" Sibling matrix generated. \n")
  for (i in seq_len(nrow(sibling_mat))) {
    siblings <- which(sibling_mat[i, ] == 1)
    #cat(sprintf("Node %3d has siblings: %s\n",
    #            i,
    #            if(length(siblings) > 0) paste(siblings, collapse = ", ") else "none"))
  }

  for (m in seq_len(M)) {
    leaves_m <- which(A[m, ] != 0)
    if (length(leaves_m) == 0) {
      C1[m] <- 0
      C2[m] <- 0
      C3[m] <- 0
      next
    }
    grad_mat_sub_m <- grad_mat[, leaves_m, drop = FALSE]

    C2[m] <- sum(colMeans(abs(grad_mat_sub_m))) / length(leaves_m) #sum((1 / n) * colSums(grad_mat_sub_m^2))

    cols <- ncol(grad_mat_sub_m)
    if (cols < 2L) {
      C1[m] <- 0
    } else {
      pairs <- combn(cols, 2, simplify = FALSE)
      s <- 0
      for (pair in pairs) {
        diff <- grad_mat_sub_m[, pair[1]] - grad_mat_sub_m[, pair[2]]
        s <- s + mean(abs(diff))
      }
      C1[m] <- s / length(pairs)
    }

    siblings_m <- which(sibling_mat[m, ] == 1)
    if (length(siblings_m) == 0) {
      saveRDS(sibling_mat, "sibling_mat")
      C3[m] <- 0
      #utils::View(A)
      stop("C3 length 0.")
    } else {
      sib_mats <- lapply(
        siblings_m,
        function(k) {
          leaves_k <- which(A[k, ] != 0)
          grad_mat[, leaves_k, drop = FALSE]
        }
      )

      p_m <- ncol(grad_mat_sub_m)
      num_pairs_C3 <- sum(sapply(sib_mats, ncol) * p_m)

      if (num_pairs_C3 == 0) {
        C3[m] <- 0
      } else {
        s3 <- 0
        for (j in seq_len(p_m)) {
          col_m <- grad_mat_sub_m[, j]
          for (mat_k in sib_mats) {
            if (ncol(mat_k) == 0) next
            for (l in seq_len(ncol(mat_k))) {
              col_k <- mat_k[, l]
              s3 <- s3 + mean(abs(col_m - col_k))
            }
          }
        }
        C3[m] <- s3 / num_pairs_C3
      }
    }
  }

  list(C1 = C1, C2 = C2, C3 = C3)
} # get_adaptive_penalties_internal_NWML_L1


#' Internal: compute adaptive penalties C1, C2, C3
#'
#' @keywords internal
get_adaptive_penalties_internal_LLR_L1 <- function(grad_llr, A, X, int_ind_ml) {
  X_train <- X
  n <- nrow(X_train)
  p <- ncol(X_train)
  M <- nrow(A)

  grad_mat <- grad_llr
  grad_mat <- grad_mat[int_ind_ml, , drop = FALSE] # subsetting to interior #NEW

  #utils::View(grad_mat)

  C1 <- numeric(M)
  C2 <- numeric(M)
  C3 <- numeric(M)

  cat("Generating sibling matrix...")
  sibling_mat <- matrix(0, nrow = M, ncol = M)

  # Find parent for each node
  parent <- rep(NA_integer_, M)

  for (i in seq_len(M)) {
    # Find all nodes that are ancestors of i (i.e., A[j,] >= A[i,])
    potential_parents <- which(sapply(
      seq_len(M),
      function(j) {
        if (i == j) return(FALSE)
        all(A[j, ] >= A[i, ]) && !all(A[j, ] == A[i, ])
      }
    ))

    if (length(potential_parents) > 0) {
      # Parent is the ancestor with minimum row sum (closest ancestor)
      parent[i] <- potential_parents[which.min(rowSums(A[potential_parents, , drop = FALSE]))]
    }
  }

  # Group children by parent (including NA for root's children)
  parent_factor <- factor(parent, levels = c(NA, seq_len(M)), exclude = NULL)
  children_by_parent <- split(seq_len(M), parent_factor)

  # Build sibling matrix
  for (parent_group in children_by_parent) {
    if (length(parent_group) > 1) {
      # All children of same parent are siblings of each other
      for (i in seq_along(parent_group)) {
        siblings <- parent_group[-i]  # All children except current one
        sibling_mat[parent_group[i], siblings] <- 1
      }
    }
  }

  cat(" Sibling matrix generated. \n")
  for (i in seq_len(nrow(sibling_mat))) {
    siblings <- which(sibling_mat[i, ] == 1)
    #cat(sprintf("Node %3d has siblings: %s\n",
    #            i,
    #            if(length(siblings) > 0) paste(siblings, collapse = ", ") else "none"))
  }

  for (m in seq_len(M)) {
    leaves_m <- which(A[m, ] != 0)
    if (length(leaves_m) == 0) {
      C1[m] <- 0
      C2[m] <- 0
      C3[m] <- 0
      next
    }

    grad_mat_sub_m <- grad_mat[, leaves_m, drop = FALSE]

    C2[m] <- sum(colMeans(abs(grad_mat_sub_m))) / length(leaves_m)

    cols <- ncol(grad_mat_sub_m)
    if (cols < 2L) {
      C1[m] <- 0
    } else {
      pairs <- combn(cols, 2, simplify = FALSE)
      s <- 0
      for (pair in pairs) {
        diff <- grad_mat_sub_m[, pair[1]] - grad_mat_sub_m[, pair[2]]
        s <- s + mean(abs(diff))
      }
      C1[m] <- s / length(pairs)
    }

    siblings_m <- which(sibling_mat[m, ] == 1)
    if (length(siblings_m) == 0) {
      C3[m] <- 0
      #utils::View(A)
      stop("C3 length 0.")
    } else {
      sib_mats <- lapply(
        siblings_m,
        function(k) {
          leaves_k <- which(A[k, ] != 0)
          grad_mat[, leaves_k, drop = FALSE]
        }
      )

      p_m <- ncol(grad_mat_sub_m)
      num_pairs_C3 <- sum(sapply(sib_mats, ncol) * p_m)

      if (num_pairs_C3 == 0) {
        C3[m] <- 0
      } else {
        s3 <- 0
        for (j in seq_len(p_m)) {
          col_m <- grad_mat_sub_m[, j]
          for (mat_k in sib_mats) {
            if (ncol(mat_k) == 0) next
            for (l in seq_len(ncol(mat_k))) {
              col_k <- mat_k[, l]
              s3 <- s3 + mean(abs(col_m - col_k))
            }
          }
        }
        C3[m] <- s3 / num_pairs_C3
      }
    }
  }

  list(C1 = C1, C2 = C2, C3 = C3)
} # get_adaptive_penalties_internal_LLR_L1

post_process <- function(gamma_vec, A, direction = "up"){
  if(direction == "up")
    return(aggregate_up(gamma_vec, A))
  else
    return(deaggregate_down(gamma_vec, A))
}# post_process

#' Internal: get descendants
#'
#' @keywords internal
get_descendants <- function(node_index, A) {
  # row for the given node
  parent <- A[node_index, ]

  # for each row, check if its 1's are a subset of parent's 1's
  is_desc <- apply(A, 1, function(r) {
    # all leaves of this row are within parent's leaves
    subset_ok   <- all(r <= parent)
    # and not exactly the same node (proper subset)
    proper_sub  <- any(r != parent)
    subset_ok && proper_sub
  })

  which(is_desc)
}# get_descendants

#' Internal: aggregate up
#'
#' @keywords internal
aggregate_up <- function(gamma_vec, A) {
  gamma_vec[which(gamma_vec!=0)] <- 1
  # indices of currently selected nodes
  active <- which(gamma_vec == 1)

  # collect all active nodes that are descendants of some other active node
  to_zero <- integer(0)
  for (i in active) {
    desc_i <- get_descendants(i, A)
    # only care about those descendants that are also active
    to_zero <- union(to_zero, intersect(desc_i, active))
  }

  # set them to zero
  gamma_vec[to_zero] <- 0
  gamma_vec
}# aggregate_up

#' Internal: deaggregate down
#'
#' @keywords internal
deaggregate_down <- function(gamma_vec, A) {
  gamma_vec[which(gamma_vec!=0)] <- 1
  repeat {
    active <- which(gamma_vec == 1)
    changed <- FALSE

    # process nodes in decreasing "size" (larger supports first)
    sizes <- rowSums(A[active, , drop = FALSE])
    active_ordered <- active[order(sizes, decreasing = TRUE)]

    for (v in active_ordered) {
      # descendants of v that are currently selected
      desc_v <- get_descendants(v, A)
      selected_desc <- intersect(desc_v, which(gamma_vec == 1))

      if (length(selected_desc) > 0) {
        # de-aggregate v
        D_v <- build_cover_for_node(v, gamma_vec, A)

        # remove v, add D_v
        gamma_vec[v] <- 0
        gamma_vec[D_v] <- 1

        changed <- TRUE
      }
    }

    if (!changed) break
  }

  gamma_vec
}# deaggregate_down


