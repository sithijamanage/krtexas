// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

//' Fast Gaussian kernel loss (internal)
 //'
 //' C++ version of the R loss_fun. It computes:
 //'   mean( (Y - yhat)^2 ) +
 //'   sum( kappa * adaptive_weights * (u^2 + w^2) ) +
 //'   alpha / sigma^2
 //'
 //' where yhat_i is the leave‑one‑out kernel regression estimate using
 //' Gaussian kernel with bandwidth sigma and anisotropic weights gamma = u*w.
 //'
 //' @param u Numeric vector (length m)
 //' @param w Numeric vector (length m)
 //' @param Ax Numeric matrix (m x n)
 //' @param X Kept for signature compatibility; not used
 //' @param Y Numeric vector (length n)
 //' @param sigma Positive scalar
 //' @param kappa Scalar
 //' @param alpha Scalar
 //' @param adaptive_weights Numeric vector (length m)
 //' @return Scalar loss value
 // [[Rcpp::export]]
 double loss_fun_fast_cpp(const arma::vec& u,
                          const arma::vec& w,
                          const arma::mat& Ax,
                          const arma::mat& X,              // unused
                          const arma::vec& Y,
                          const double sigma,
                          const double kappa,
                          const double alpha,
                          const arma::vec& adaptive_weights) {
   
   (void)X; // suppress unused parameter warning
   
   const uword m = u.n_elem;
   const uword n = Y.n_elem;
   
   if (w.n_elem != m) stop("w must have same length as u");
   if (adaptive_weights.n_elem != m) stop("adaptive_weights must have same length as u");
   if (Ax.n_rows != m) stop("nrow(Ax) must equal length(u)");
   if (Ax.n_cols != n) stop("ncol(Ax) must equal length(Y)");
   if (!(sigma > 0.0)) stop("sigma must be positive");
   
   arma::vec gamma = u % w;
   if (gamma.min() < 0.0) {
     Rcpp::warning("gamma = u*w has negative entries; continuing anyway.");
   }
   
   // Precompute weighted Gram: G = Ax^T * (diag(gamma) * Ax)  (n x n)
   arma::mat WAx = Ax.each_col() % gamma; // m x n
   arma::mat G   = Ax.t() * WAx;          // n x n
   arma::vec diagG = G.diag();            // n
   
   const double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
   const double tiny = 1e-15;
   
   arma::vec yhat(n, fill::zeros);
   bool bad = false;
   
   // Temporary row for D_i,*
   arma::rowvec D_row(n);
   
   for (uword i = 0; i < n; ++i) {
     // D_row(j) = ||Ax_col_i - Ax_col_j||^2_gamma
     //          = G_ii + G_jj - 2 G_ij
     D_row = diagG.t();
     D_row += diagG(i);
     D_row -= 2.0 * G.row(i);
     
     arma::vec e(n, fill::zeros);
     
     for (uword j = 0; j < n; ++j) {
       if (j == i) {
         e(j) = 0.0; // leave-one-out
         continue;
       }
       double x = -D_row(j) * inv_two_sigma2;
       e(j) = std::exp(x);
     }
     
     const double sumExp_d   = arma::accu(e);
     const double sumExp_d_Y = arma::dot(e, Y);
     
     if (sumExp_d <= tiny || !std::isfinite(sumExp_d)) {
       bad = true;
       break; // will return NA, letting R wrapper handle it
     }
     
     yhat(i) = sumExp_d_Y / sumExp_d;
   }
   
   if (bad) {
     return NA_REAL; // R side checks finiteness and penalizes
   }
   
   // Squared error loss
   arma::vec diff = Y - yhat;
   double loss = arma::dot(diff, diff);
   
   // Structured penalty
   arma::vec kappa_vec = kappa * adaptive_weights;
   double spred_penalty =
     arma::dot(kappa_vec, u % u) +
     arma::dot(kappa_vec, w % w);
   
   double nY = static_cast<double>(n);
   
   return loss / nY + spred_penalty + (alpha / (sigma * sigma));
 }