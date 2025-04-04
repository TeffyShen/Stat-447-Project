data {
  int<lower=1> N;             // Number of observations
  array[N] real x;            // Time index
  vector[N] y;                // Observed returns (standardized)
  int<lower=1> N_pred;        // Number of prediction points
  array[N_pred] real x_pred;  // Time index for predictions
}

transformed data {
  vector[N] mu = rep_vector(0, N);
  vector[N_pred] mu_pred = rep_vector(0, N_pred);
  real delta = 1e-8;          // Small stabilizing constant
}

parameters {
  real<lower=0> alpha;        // Amplitude parameter
  real<lower=0> rho;          // Length-scale parameter
  real<lower=0> sigma;        // Noise level
}

model {
  matrix[N, N] K = cov_exp_quad(x, alpha, rho);
  matrix[N, N] L_K;
  
  // Add small value to diagonal for numerical stability
  for (n in 1:N)
    K[n, n] = K[n, n] + delta;
  
  L_K = cholesky_decompose(K);
  
  // Tighter priors more appropriate for returns
  alpha ~ normal(0, 0.5);
  rho ~ inv_gamma(5, 5);
  sigma ~ normal(0, 0.1);
  
  y ~ multi_normal_cholesky(mu, L_K);
}

generated quantities {
  vector[N_pred] f_pred;
  {
    matrix[N, N] K = cov_exp_quad(x, alpha, rho) 
                   + diag_matrix(rep_vector(delta, N));
    matrix[N, N_pred] K_pred = cov_exp_quad(x, x_pred, alpha, rho);
    matrix[N_pred, N_pred] K_pred_pred = cov_exp_quad(x_pred, x_pred, alpha, rho)
                                       + diag_matrix(rep_vector(delta, N_pred));
    matrix[N, N] L = cholesky_decompose(K);
    matrix[N, N_pred] L_div_K_pred = mdivide_left_tri_low(L, K_pred);
    vector[N_pred] f_pred_mu = L_div_K_pred' * mdivide_right_tri_low(y', L)';
    matrix[N_pred, N_pred] cov_f_pred = K_pred_pred - L_div_K_pred' * L_div_K_pred;
    
    f_pred = multi_normal_rng(f_pred_mu, cov_f_pred);
  }
}

