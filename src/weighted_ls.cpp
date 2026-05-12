// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <limits>

// [[Rcpp::export]]
arma::mat get_grad_llr(const arma::mat& X,
                       const arma::vec& y,
                       const arma::mat& X0,
                       const arma::vec& gamma) {
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;   // number of predictors
  const arma::uword m = X0.n_rows;  // number of query points

  if (y.n_elem != n)          Rcpp::stop("length(y) must equal nrow(X).");
  if (X0.n_cols != p)         Rcpp::stop("ncol(X0) must equal ncol(X).");
  if (gamma.n_elem != p)      Rcpp::stop("gamma must have length p (= ncol(X)).");
  if (arma::any(gamma < 0.0)) Rcpp::stop("All gamma entries must be nonnegative.");

  // Treat gamma as rowvec for broadcasting
  const arma::rowvec g = gamma.t();

  // Result matrix: m x (p+1); row k corresponds to X0.row(k)
  arma::mat B(m, p + 1);
  const double NA = std::numeric_limits<double>::quiet_NaN();
  const arma::uword q = p + 1;  // parameters in local linear model

  for (arma::uword k = 0; k < m; ++k) {
    const arma::rowvec x0 = X0.row(k);

    // Centered differences D = X - x0  (n x p)
    arma::mat D  = X.each_row() - x0;
    arma::mat D2 = arma::square(D);        // n x p

    // Exponents: expo_i = - sum_j gamma_j * D_ij^2   (n x 1)
    arma::vec expo = - (D2 * g.t());

    // ---------------- d_min stabilization (exponent shifting) ----------------
    // Shift exponents by their maximum so the largest becomes 0:
    // weights_i = exp(expo_i - max(expo))  == exp(-(d_i - d_min))
    // This prevents underflow without changing the WLS solution (common factor cancels).
    double expo_max = expo.max();                   // <= key line
    arma::vec w     = arma::exp(expo - expo_max);   // stabilized weights in (0, 1]
    // ------------------------------------------------------------------------

    // Default row result is NA if we fail to solve or insufficient info
    arma::rowvec beta_k(q);
    beta_k.fill(NA);

    // Count strictly positive weights (may be smaller than n due to underflow in tails)
    const arma::uword m_eff = arma::accu(w > 0.0);

    // Need at least q observations with positive weight, and a nonzero max
    if (m_eff > q && w.max() > 0) {
      // Build sqrt-weighted design Z_w = [ sqrt(w), D .* sqrt(w) ]  (n x q)
      arma::vec sw = arma::sqrt(w);
      arma::mat Zw(n, q);
      Zw.col(0)     = sw;
      Zw.cols(1, p) = D.each_col() % sw;

      // Cross-products
      arma::mat XtWX = Zw.t() * Zw;           // q x q
      arma::vec Xty  = Zw.t() * (y % sw);     // q

      // Solve for beta; prefer SPD; fallback to general, then pinv
      arma::vec beta_vec(q);
      bool ok = arma::solve(beta_vec, XtWX, Xty, arma::solve_opts::likely_sympd);
      if (!ok) ok = arma::solve(beta_vec, XtWX, Xty);
      if (!ok || !beta_vec.is_finite()) {
        arma::mat XtWX_pinv = arma::pinv(XtWX);
        beta_vec = XtWX_pinv * Xty;
      }
      if (beta_vec.is_finite()) {
        beta_k = beta_vec.t();  // store as row
      }
    }

    B.row(k) = beta_k;
  }

  return B;
}
