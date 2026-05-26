# 📦 krtexas

**Kernel Regression with Tree-EXploring AggregationS (KR-TEXAS)**  
An R package for **nonparametric regression via tree-guided feature aggregation**.

***

## ✨ Overview

The **krtexas** package implements the KR-TEXAS method for regression problems where predictors are organized in a **known hierarchical tree structure** (e.g. taxonomic tree, brain region hierarchy, geographical hierarchy).

KR-TEXAS simultaneously:

* ✅ Performs **nonparametric regression**
* ✅ Learns the **correct level of feature aggregation** (and selects relevant variables)

The method is based on a **penalized Nadaraya–Watson estimator with adaptive weights**, enabling joint **model selection and aggregation** in nonlinear settings.
***

## ⚡️ Motivation

In many scientific applications (e.g., microbiome, genomics, stock market data), predictors naturally form a **tree structure**.

Rather than choosing a fixed resolution (e.g., species vs. genus) as regression inputs, KR-TEXAS **learns the optimal level directly from data**, improving:

* interpretability
* statistical efficiency
* predictive performance
  
***

## ⚙️ Installation

Install from GitHub:

```r
# install.packages("devtools")
devtools::install_github(
  "sithijamanage/krtexas",
  build_vignettes = TRUE
)
```

***
**Read the vignette for a full, illustrative example:**  
`browseVignettes("krtexas")`

***

## 🚀 Quick Start

```r
library(krtexas)

# X: n x p matrix of predictors
# Y: response vector
# A: tree structure matrix (T x p)

fit <- krtexas_fit(
  X = X,
  Y = Y,
  A = A
)

# Predictions
pred <- krtexas_predict(fit, newx = X_test)
```

***

## 📘 Key Function `krtexas_fit()`

Main entry point for fitting the model.

### 📥 Inputs

#### Core Inputs

* `X`: predictor matrix of size *(n × p)*
* `Y`: response vector of length *n*
* `A`: tree / aggregation matrix of size *(T × p)* defining the hierarchical structure

***

All of the following parameters have default values given in parentheses.
#### Cross-Validation and Regularization

* `nfolds`: number of folds used in cross-validation (4)
* `lambda`: optional fixed regularization parameter
* `min_lambda`, `max_lambda`: lower and upper bounds of the λ search range (used if `lambda` is `NULL`) (0, NULL)
* `nlambda`: number of candidate λ values evaluated within the specified range (10)

If `lambda` is not provided, the algorithm performs cross-validation over the specified range to select the optimal value.

***

#### Optimization Control

* `eps`: convergence tolerance for the L-BFGS optimizer (1e-6)
  * Smaller values ⇒ stricter convergence
* `silent`: suppress optimizer output if `TRUE` (TRUE)
* `parallel`: run cross-validation folds in parallel if `TRUE` (TRUE)
* `n_cores`: number of CPU cores used for parallel computation (detectCores() - 1)
* `warm_start`: whether to initialize each λ optimization using results from the previous λ value (TRUE)

***

#### Pilot Estimation and Initialization

* `method`: pilot estimator used for adaptive weights ("LLR")
  * `"NW_ML"`: Nadaraya–Watson with metric learning
  * `"LLR"`: local linear regression
  * `"LQR"`: local quadratic regression

* `gamma_init_strat`: initialization strategy for optimization parameters ("smallest")
  * `"smallest"`: draws from $$N(0.1, 0.01)$$ 
  * `"small"`: draws from $$N(1, 1/4)$$
  * `"large"`: draws from $$N(\max(1, n^{2/(4+T)}), 1)$$

***

#### Adaptive Penalty and Restarts

* `distance`: distance metric used in adaptive weights ("L2")
  * `"L2"`: squared difference
  * `"L1"`: absolute difference

* `num_restarts_stage_1`: number of random restarts for the **initial (λ = 0) fit** (10)

* `num_restarts_stage_2`: number of random restarts for the **adaptive penalty fit** (10)

For both stages:

* One third of the restarts use `"smallest"` initialization
* One third use `"small"` initialization
* One third use `"large"` initialization
* The best solution (lowest loss) is kept

***

#### Final Optimization Robustness

* `max_attempts_stage_3`:  
  Number of retry attempts for the final model fit if optimization fails to converge (10)

***

#### Notes

* Adaptive weights use internally fixed tuning values:
  $$a_2 = \frac{1}{2(2 + p)}, \quad b = 1$$
* Multiple restarts are **strongly recommended** due to non-convex optimization
* Parallelization is most useful when `nfolds > 1`

***

### 📤 Output

A list containing:

* `gammas_learned` – selected features/aggregations
* `lambda.best` – optimal penalty
* `training_loss`
* `convergence`
* `inputs` – original data

***

### 📈 Predicting on New Data

```r
# Predictions
pred <- krtexas_predict(saved_krtexas_model, newx = X_test)
```


## 🔍 Method Summary

KR-TEXAS solves:

$$
\min_{\gamma} \; L_n(\gamma) + \lambda \sum_v w_v \gamma_v
$$

where:

* $$L_n(\gamma)$$ is a leave-one-out kernel regression loss
* $$w_v$$ are **adaptive weights based on estimated gradients**

This allows the method to:

* shrink irrelevant nodes to zero
* retain only meaningful **aggregations or leaves**

The procedure is:

### Stage 1

Fit model with **λ = 0** → obtain pilot estimates

### Stage 2

Construct **adaptive weights** from gradients

### Stage 3

Perform **penalized optimization with cross-validation**

***

## 📖 Reference

If you use this package, please cite:

> Manage, S., Wang, Y. S., & Wells, M. T. (2026).  
> *Nonparametric Regression via Tree-Guided Feature Aggregation*.  
> arXiv preprint.

***

## 📎 Repository Structure

```
krtexas/
├── R/             # Core R functions
├── man/           # Documentation
├── src/           # C++ backend (fast computations)
├── vignettes/     # Examples and tutorials
├── DESCRIPTION
├── NAMESPACE
```

***

## 👤 Authors

**Sithija Manage**  
Cornell University  
📧 <ssm255@cornell.edu>

**Y. Samuel Wang**  
Cornell University 

**Martin T. Wells**  
Cornell University 

***

## 📜 License

MIT License 
***
