functions {
  matrix cov_exp_quad(vector x, real alpha, real rho) {
    int N = rows(x);
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = alpha^2 + 1e-6;
      for (j in (i+1):N) {
        K[i, j] = alpha^2 * exp(-0.5 * square((x[i] - x[j])/rho));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = alpha^2 + 1e-6;
    return K;
  }
}

data {
  int<lower=1> N;
  vector[N] y;
  vector[N] x;
}

parameters {
  real mu;
  real<lower=0> alpha;
  real<lower=0> rho;
  real<lower=0> sigma;
  vector[N] f;
}

model {
  matrix[N, N] K = cov_exp_quad(x, alpha, rho);
  matrix[N, N] L_K = cholesky_decompose(K);
  
  // Priors
  mu ~ normal(0, 1);
  alpha ~ exponential(1);
  rho ~ exponential(1);
  sigma ~ exponential(1);
  
  f ~ multi_normal_cholesky(rep_vector(mu, N), L_K);
  y ~ normal(0, exp(f) * sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] volatility;
  
  volatility = exp(f) * sigma;
  for (n in 1:N) {
    y_rep[n] = normal_rng(0, volatility[n]);
  }
}
