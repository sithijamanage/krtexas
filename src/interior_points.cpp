// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <limits>

//' Compute interiorness scores using the exp kernel
//' 
//' For each sample point X[i,], compute
//'   score[i] = (1/n) * sum_{k=1}^n exp( - sum_{j=1}^p gamma_j * (X[i,j] - X[k,j])^2 )
//' Optionally exclude self-similarity which only shifts all scores by a constant 1.
//' 
//' @param X Numeric matrix (n x p): sample points.
//' @param gamma Numeric vector (length p): nonnegative kernel scales per dimension.
//' @param exclude_self logical: if TRUE, set K(X_i, X_i) = 0 in the sum.
//' @param normalize logical: if TRUE, divide by n (number of points).
//' @return Numeric vector (length n): interiorness scores (larger = more interior).
//' @examples
//' \dontrun{
//' scores <- interior_scores_exp_kernel_cpp(X, gamma)
//' order <- order(scores, decreasing = TRUE)
//' }
// [[Rcpp::export]]
 arma::vec interior_scores_exp_kernel_cpp(const arma::mat& X,
                                          const arma::vec& gamma,
                                          bool exclude_self = true,
                                          bool normalize = true) {
   const arma::uword n = X.n_rows;
   const arma::uword p = X.n_cols;
   if (gamma.n_elem != p) Rcpp::stop("gamma must have length p (= ncol(X)).");
   if (arma::any(gamma < 0.0)) Rcpp::stop("All gamma entries must be nonnegative.");
   const arma::rowvec g = gamma.t();
   
   arma::vec scores(n, arma::fill::zeros);
   for (arma::uword i = 0; i < n; ++i) {
     const arma::rowvec xi = X.row(i);
     arma::mat D = X.each_row() - xi;                // n x p
     arma::vec expo = - (arma::square(D) * g.t());   // n
     arma::vec w = arma::exp(expo);
     if (exclude_self && n > 0) w(i) = 0.0;          // drop self-similarity
     double s = arma::sum(w);
     scores(i) = normalize ? (s / static_cast<double>(n)) : s;
   }
   return scores;
 }
 
//' Find the top-m most interior sample points under the exp kernel
//' 
//' @param X Numeric matrix (n x p): sample points.
//' @param gamma Numeric vector (length p): nonnegative kernel scales per dimension.
//' @param m integer: number of most interior points to return.
//' @param exclude_self logical: if TRUE, set K(X_i, X_i) = 0 when scoring.
//' @param normalize logical: if TRUE, divide scores by n.
//' @return A list with elements:
//'   - indices: IntegerVector (length m), 1-based row indices into X (descending by score)
//'   - scores:  NumericVector (length m), the corresponding scores
//'   - X_top:   NumericMatrix (m x p), the top-m points
//' 
//' @examples
//' \dontrun{
//' res <- top_interior_points_exp_kernel_cpp(X, gamma, m = 10)
//' res$indices; res$scores; res$X_top
//' }
// [[Rcpp::export]]
 Rcpp::List find_interior_points(const arma::mat& X,
                                               const arma::vec& gamma,
                                               int m,
                                               bool exclude_self = true,
                                               bool normalize = true) {
   const arma::uword n = X.n_rows;
   if (m <= 0) Rcpp::stop("m must be positive.");
   if (static_cast<arma::uword>(m) > n) Rcpp::stop("m cannot exceed number of rows in X.");
   
   arma::vec scores = interior_scores_exp_kernel_cpp(X, gamma, exclude_self, normalize);
   
   // Order indices by descending score
   arma::uvec ord = arma::sort_index(scores, "descend");
   ord = ord.head(static_cast<arma::uword>(m));
   
   // Build outputs
   Rcpp::IntegerVector idx(m);
   arma::vec top_scores(m);
   arma::mat X_top(m, X.n_cols);
   for (int k = 0; k < m; ++k) {
     arma::uword i = ord[k];
     idx[k] = static_cast<int>(i) + 1; // 1-based for R
     top_scores[k] = scores[i];
     X_top.row(k) = X.row(i);
   }
   
   return Rcpp::List::create(Rcpp::Named("indices") = idx,
                             Rcpp::Named("scores")  = Rcpp::NumericVector(top_scores.begin(), top_scores.end()),
                             Rcpp::Named("X_top")   = Rcpp::NumericMatrix(Rcpp::wrap(X_top)));
 }
 