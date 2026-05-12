// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <limits>

/*
 * Local QUADRATIC regression (no interaction terms) with exponential product kernel.
 *
 * Kernel weights at x0:
 *   w_i = exp( - sum_{j=1}^p gamma_j * (X_ij - x0_j)^2 )
 *
 * Local quadratic design around x0 (centered basis), WITHOUT cross terms:
 *   Z = [ 1,
 *         (X - x0)_1, ..., (X - x0)_p,          // linear terms
 *         (X - x0)_1^2, ..., (X - x0)_p^2 ]     // quadratic diagonal only
 *
 * We solve (Z' W Z) beta = Z' W y using sqrt-weights for stability.
 *
 * RETURN: m x p matrix with ONLY the linear coefficients (local slopes) at each x0.
 *
 * Arguments:
 *   X     : n x p matrix of predictors (training)
 *   y     : length-n response vector
 *   X0    : m x p matrix of query points
 *   gamma : length-p nonnegative scales
 */

// [[Rcpp::export]]
arma::mat get_grad_lqr(const arma::mat& X,
                               const arma::vec& y,
                               const arma::mat& X0,
                               const arma::vec& gamma) {
  
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  const arma::uword m = X0.n_rows;
  
  if (y.n_elem != n) Rcpp::stop("length(y) must equal nrow(X).");
  if (X0.n_cols != p) Rcpp::stop("ncol(X0) must equal ncol(X).");
  if (gamma.n_elem != p) Rcpp::stop("gamma must have length p (= ncol(X)).");
  if (arma::any(gamma < 0.0)) Rcpp::stop("All gamma entries must be nonnegative.");
  
  // Treat gamma as rowvec for broadcasting
  const arma::rowvec g = gamma.t();
  
  // Number of columns in local quadratic design without interactions:
  // 1 (intercept) + p (linear) + p (quadratic diag) = 1 + 2p
  const arma::uword q = 1 + 2 * p;
  
  // Output: ONLY linear coefficients (m x p)
  arma::mat G(m, p, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));
  
  for (arma::uword k = 0; k < m; ++k) {
    const arma::rowvec x0 = X0.row(k);
    
    // Centered differences D = X - x0 (n x p)
    arma::mat D = X.each_row() - x0;
    
    // Weights: w_i = exp( - sum_j gamma_j * D_ij^2 )
    arma::mat D2 = arma::square(D);       // n x p
    arma::vec expo = - (D2 * g.t());      // n x 1
    arma::vec w = arma::exp(expo);        // n x 1
    
    // Need at least q effective points with positive weight
    const arma::uword m_eff = arma::accu(w > 0.0);
    if (m_eff > q && w.max() > 0) {
      
      // sqrt-weighted design
      arma::vec sw = arma::sqrt(w);
      arma::mat Zw(n, q, arma::fill::zeros);
      
      // Intercept
      Zw.col(0) = sw;
      
      // Linear terms
      Zw.cols(1, p) = D.each_col() % sw;  // n x p
      
      // Quadratic diagonal terms (no interactions)
      arma::uword col_idx = 1 + p;
      for (arma::uword j = 0; j < p; ++j) {
        Zw.col(col_idx++) = (D.col(j) % D.col(j)) % sw; // (X_j - x0_j)^2
      }
      
      // Weighted response
      arma::vec yw = y % sw;
      
      // Normal equations (weighted)
      arma::mat XtWX = Zw.t() * Zw;  // q x q
      arma::vec Xty  = Zw.t() * yw;  // q
      
      // Solve
      arma::vec beta_vec(q);
      bool ok = arma::solve(beta_vec, XtWX, Xty, arma::solve_opts::likely_sympd);
      if (!ok) ok = arma::solve(beta_vec, XtWX, Xty);
      if (!ok || !beta_vec.is_finite()) {
        arma::mat XtWX_pinv = arma::pinv(XtWX);
        beta_vec = XtWX_pinv * Xty;
      }
      
      if (beta_vec.is_finite()) {
        // Extract ONLY the linear terms (positions 1..p)
        G.row(k) = beta_vec.subvec(1, p).t();
      }
    }
  }
  
  return G;
}