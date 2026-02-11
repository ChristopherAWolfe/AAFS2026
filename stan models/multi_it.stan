data {
  int<lower=1> N;                   // Number of observations
  int<lower=1> G;                   // Number of groups
  int<lower=1> D;                   // Dimensionality of Y
  matrix[N, D] y;                   // Multivariate response
  array[N] int<lower=1, upper=G> group;   // Group indicator
}

parameters {
  array[G] vector<lower=0>[D] mu;                        // Group-specific means
  cholesky_factor_corr[D] L_corr;     // Group-specific correlation matrices
  vector<lower=0>[D] sigma;           // Group-specific scales
}

transformed parameters {
  matrix[D, D] L;                     // Group-specific Cholesky of covariance
  L = diag_pre_multiply(sigma, L_corr);
  
}
model {
  // Priors
  for (g in 1:G) {
    
      mu[g,1] ~ normal(150,10);               // Prior on group means
      mu[g,2] ~ normal(140,10);
      mu[g,3] ~ normal(130,10);
      mu[g,4] ~ normal(130,10);
      mu[g,5] ~ normal(100,10);
      mu[g,6] ~ normal(60,10);
      mu[g,7] ~ normal(120,10);
      mu[g,8] ~ normal(70,10);
      mu[g,9] ~ normal(100,10);
      mu[g,10] ~ normal(110,10);
      mu[g,11] ~ normal(100,10);
      mu[g,12] ~ normal(50,10);
      mu[g,13] ~ normal(30,10);
      mu[g,14] ~ normal(40,10);
      mu[g,15] ~ normal(30,10);
      mu[g,16] ~ normal(100,10);
      mu[g,17] ~ normal(20,10);
      mu[g,18] ~ normal(110,10);
      mu[g,19] ~ normal(110,10);
      mu[g,20] ~ normal(100,10);
      mu[g,21] ~ normal(40,10);
      mu[g,22] ~ normal(30,10);

  }
  
  sigma ~ normal(0,2.5);
  L_corr ~ lkj_corr_cholesky(4);   // Prior on correlation structure

  // Likelihood
  for (n in 1:N) {
    y[n] ~ multi_normal_cholesky(mu[group[n]], L);
  }
}
