// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
Rcpp::List loss_grad_rcpp(const arma::vec& u,
                          const arma::vec& w,
                          const arma::mat& Ax,   // p x n
                          const arma::mat& X,    // unused; kept for signature parity
                          const arma::vec& Y,    // length n
                          const double sigma,
                          const double kappa,
                          const double alpha,
                          const arma::vec& adaptive_weights,
                          const double eps = 1e-12) {
  const int p = u.n_elem;
  const int n = Y.n_elem;
  const double s2 = sigma * sigma;

  // ----- gamma and regularization -----
  arma::vec gamma = u % w;                         // p
  arma::vec kappa_vec = kappa * adaptive_weights;  // p

  // ----- B = diag(sqrt(gamma)) %*% Ax  (row-wise scale) -----
  arma::vec sqrt_gamma = arma::sqrt(arma::clamp(gamma, 0.0, datum::inf));
  arma::mat B = Ax.each_col() % sqrt_gamma;        // p x n

  // ----- M = t(B) %*% B (n x n), symmetric -----
  arma::mat M = B.t() * B;                         // O(n n p)

  // ----- Build stabilized W -----
  // A_ij = (2 M_ij - d_i - d_j) / (2*s2)  so that W_ij = exp(A_ij) = exp(-D_ij)
  arma::vec d = M.diag();                          // n
  arma::mat A = 2.0 * M;                           // n x n
  A.each_col() -= d;                               // subtract d_i from every column
  A.each_row() -= d.t();                           // subtract d_j from every row
  A /= (2.0 * s2);                                 // now A = (2M - d - d^T)/(2*s2)

  // Numerical trick: subtract max in each row before exponentiating.
  // This is equivalent to subtracting min_j D_{ij} in the distance space
  // and does NOT change the ratios used for yhat.
  arma::vec row_max = arma::max(A, 1);             // n x 1, row-wise maxima of A
  A.each_row() -= row_max.t();                     // A_ij <- A_ij - max_j A_ij

  // Clamp exponents to avoid overflow/NaN in exp
  A = arma::clamp(A, -700.0, 0.0);


  arma::mat W = arma::exp(A);
  W.diag().zeros();                                // exclude self

  // ----- Predictions -----
  arma::vec S = arma::sum(W, 1);
  S = arma::clamp(S, eps, datum::inf);
  arma::vec WY = W * Y;                            // n
  arma::vec yhat = WY / S;                         // n
  arma::vec resid = yhat - Y;                      // n

  // ----- Loss -----
  const double data_loss = arma::mean(arma::square(resid));
  const double reg_loss  = arma::dot(kappa_vec, arma::square(u) + arma::square(w));
  const double loss = data_loss + reg_loss + (alpha / s2);

  // ----- Gradient w.r.t gamma (fully vectorized) -----
  // WR = W % (Y_j - yhat_i)  computed without forming R explicitly:
  //   WR = (W each_row % Y^T) - (W each_col % yhat)
  arma::mat WR = W.each_row() % Y.t();             // n x n
  WR -= W.each_col() % yhat;                       // n x n

  // V = Ax^T (n x p);  V2 = V % V
  arma::mat V  = Ax.t();                           // n x p
  arma::mat V2 = V % V;                            // n x p

  // B1 = WR * V,  B2 = WR * V2     (two GEMMs)
  arma::mat B1 = WR * V;                           // n x p
  arma::mat B2 = WR * V2;                          // n x p

  // T = B2 - 2 * (V % B1)          (n x p)
  arma::mat T = B2 - 2.0 * (V % B1);

  // cvec = -(yhat - Y) / (s2 * S)
  arma::vec cvec = -resid / (s2 * S);              // n

  // grad_gamma_k = sum_i c_i * T_{i,k}  ==>  grad_gamma = T^T * cvec
  arma::vec grad_gamma = T.t() * cvec;             // p

  // ----- Chain rule to u and w; divide by n to match your original scaling -----
  arma::vec grad_u = (grad_gamma % w) / double(n) + 2.0 * (kappa_vec % u);
  arma::vec grad_w = (grad_gamma % u) / double(n) + 2.0 * (kappa_vec % w);

  arma::vec grad(2 * p);
  grad.subvec(0,     p - 1) = grad_u;
  grad.subvec(p, 2 * p - 1) = grad_w;

  return Rcpp::List::create(Rcpp::Named("loss") = loss,
                            Rcpp::Named("grad") = grad);
}
