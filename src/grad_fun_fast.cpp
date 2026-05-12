// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// Optional: enable OpenMP if you plan to parallelize the outer i-loop
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

//' Fast gradient function using weighted Gram + matvec identities
 //'
 //' This is a numerically equivalent but faster version of the original grad_fun.
 //' It uses a weighted Gram matrix to get all D_{i,*} in O(n) per row after one
 //' O(m n^2) GEMM, and replaces per-feature loops with two matvecs per i.
 //'
 //' @param u Numeric vector (length m)
 //' @param w Numeric vector (length m)
 //' @param Ax Numeric matrix (m x n)
 //' @param X Kept for signature compatibility; not used
 //' @param Y Numeric vector (length n)
 //' @param sigma Positive scalar
 //' @param kappa Scalar
 //' @param adaptive_weights Numeric vector (length m)
 //' @return Numeric vector length 2m: c(dthetadu, dthetadw)/n + 2*c(kappa_vec*u, kappa_vec*w)
 // [[Rcpp::export]]
 arma::vec grad_fun_fast_cpp(const arma::vec& u,
                             const arma::vec& w,
                             const arma::mat& Ax,
                             const arma::mat& X, // unused
                             const arma::vec& Y,
                             const double sigma,
                             const double kappa,
                             const arma::vec& adaptive_weights) {

   (void)X; // suppress unused parameter warning

   const uword m = u.n_elem;
   const uword n = Y.n_elem;

   if (w.n_elem != m) stop("w must have same length as u");
   if (adaptive_weights.n_elem != m) stop("adaptive_weights must have same length as u");
   if (Ax.n_rows != m) stop("nrow(Ax) must equal length(u)");
   if (Ax.n_cols != n) stop("ncol(Ax) must equal length(Y)");
   if (!(sigma > 0.0)) stop("sigma must be positive");

   // gamma and weighted-diagonal W = diag(gamma)
   arma::vec gamma = u % w;
   if (gamma.min() < 0.0) {
     Rcpp::warning("gamma = u*w has negative entries; the original formulation "
                     "uses sqrt(gamma)^2. Continuing with W = diag(gamma).");
   }

   // Precompute weighted Gram: G = X^T * (W X)  (n x n)
   // Scale each row of Ax by gamma (WAx = diag(gamma) * Ax)
   arma::mat WAx = Ax.each_col() % gamma;      // m x n
   arma::mat G   = Ax.t() * WAx;               // n x n  (BLAS GEMM)
   arma::vec diagG = G.diag();                 // n

   // Precompute Ax.^2 once for all i (saves recomputation)
   arma::mat Ax_sq = Ax % Ax;                  // m x n

   const double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
   const double tiny = 1e-15;

   arma::vec gradient_vector(m, fill::zeros);

   // Reusable temporaries
   arma::rowvec D_row(n, fill::zeros);
   arma::vec e(n, fill::zeros);
   arma::vec v(n, fill::zeros), vY(n, fill::zeros);
   arma::vec s2(m, fill::zeros), s1(m, fill::zeros);
   arma::vec s2Y(m, fill::zeros), s1Y(m, fill::zeros);

   for (uword i = 0; i < n; ++i) {
     // D_row = diagG.t() + diagG[i] - 2 * G.row(i)
     D_row = diagG.t();
     D_row += diagG(i);
     D_row -= 2.0 * G.row(i);

     // e = exp(-D_row / (2*sigma^2)), but zero out self term
     // (avoid overflow on large positive/negative values gracefully)
     for (uword j = 0; j < n; ++j) {
       if (j == i) { e(j) = 0.0; continue; }
       double x = -D_row(j) * inv_two_sigma2;
       e(j) = std::exp(x);
     }

     const double sumExp_d   = arma::accu(e);
     if (sumExp_d <= tiny || !std::isfinite(sumExp_d)) {
       // No effective neighbors; dydg_row is zero
       continue;
     }

     const double sumExp_d_Y = arma::dot(e, Y);
     const double yhat_i     = sumExp_d_Y / sumExp_d;

     // v = -e/(2*sigma^2), vY = v % Y
     v  = -e * inv_two_sigma2;
     vY = v % Y;

     // Shared matvecs across all features k
     // s1  = (Ax.^2) * v         ; s2  = Ax * v
     // s1Y = (Ax.^2) * vY        ; s2Y = Ax * vY
     s1  = Ax_sq * v;            // m x 1
     s2  = Ax    * v;            // m x 1
     s1Y = Ax_sq * vY;           // m x 1
     s2Y = Ax    * vY;           // m x 1

     // a_i = Ax.col(i)
     const arma::vec a_i = Ax.col(i);
     const arma::vec a_i_sq = a_i % a_i;

     // For all k:
     // d_sumExp_d_dgamma[k]   = s1[k]  - 2*a_i[k]*s2[k]  + a_i[k]^2 * sum(v)
     // d_sumExp_d_Y_dgamma[k] = s1Y[k] - 2*a_i[k]*s2Y[k] + a_i[k]^2 * sum(vY)
     const double s0  = arma::accu(v);
     const double s0Y = arma::accu(vY);

     arma::vec d_sumExp_d_dgamma   = s1  - 2.0 * (a_i % s2)  + a_i_sq * s0;
     arma::vec d_sumExp_d_Y_dgamma = s1Y - 2.0 * (a_i % s2Y) + a_i_sq * s0Y;

     // dydg_row = (d(sY)*s - d(s)*sY) / s^2
     arma::vec dydg_row = (d_sumExp_d_Y_dgamma * sumExp_d - d_sumExp_d_dgamma * sumExp_d_Y) /
       (sumExp_d * sumExp_d);

     gradient_vector += 2.0 * (yhat_i - Y(i)) * dydg_row;
   }

   // Map to u and w, add regularization
   arma::vec dthetadu = gradient_vector % w;
   arma::vec dthetadw = gradient_vector % u;
   arma::vec kappa_vec = kappa * adaptive_weights;

   arma::vec result(2 * m);
   result.subvec(0,     m - 1) = dthetadu / static_cast<double>(n) + 2.0 * (kappa_vec % u);
   result.subvec(m, 2 * m - 1) = dthetadw / static_cast<double>(n) + 2.0 * (kappa_vec % w);
   return result;
 }
