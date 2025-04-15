data {
  int<lower=0> N;
  matrix[N, 2] X;
  vector[N] y;
}
parameters {
  real alpha;
  real beta1;
  real beta2;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 100);
  beta1 ~ normal(0, 10);
  beta2 ~ normal(0, 10);
  sigma ~ normal(0, 50);

  y ~ normal(alpha + X[,1] * beta1 + X[,2] * beta2, sigma);
}
