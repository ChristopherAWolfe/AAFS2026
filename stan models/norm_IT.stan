data {
  int<lower=1> N;                    // number of observations
  int<lower=1> G;                    // number of groups
  array [N] int<lower=1, upper=G> group;   // known group assignments
  vector[N] y;                      // continuous outcome
}

parameters {
  vector[G] mu_group;                 // group means
  vector<lower=0>[G] sigma_group;    // group std deviations
}

model {
  // Priors
  mu_group ~ normal(0, 20);
  sigma_group ~ exponential(1);

  // Likelihood
  for (n in 1:N)
    y[n] ~ normal(mu_group[group[n]], sigma_group[group[n]]);
}

generated quantities {
  vector[G] group_probs;
  real H_cond = 0;
  real H_marg = 0;
  real I_Y_G;
  real H_group = 0;  // Entropy of G = H(X)


  // 1. Compute empirical group proportions
  for (g in 1:G) {
    int count_g = 0;
    for (n in 1:N)
      if (group[n] == g)
        count_g += 1;
    group_probs[g] = count_g / (1.0 * N);
  }

  // 2. Conditional entropy: sum over groups of p(g)*H(N(mu_g, sigma_g))
  for (g in 1:G) {
    H_cond += group_probs[g] * (0.5 * log(2 * pi() * exp(1) * square(sigma_group[g])));
  }

  // 3. Marginal entropy H(Y) from mixture log likelihood
  for (n in 1:N) {
    vector[G] log_components;
    for (g in 1:G)
      log_components[g] = log(group_probs[g]) + normal_lpdf(y[n] | mu_group[g], sigma_group[g]);
    H_marg += log_sum_exp(log_components);
  }
  H_marg = -H_marg / N; // average negative log likelihood = entropy estimate

  // 4. Mutual information
  I_Y_G = H_marg - H_cond;
  
  for (g in 1:G) {
    if (group_probs[g] > 0)
      H_group -= group_probs[g] * log(group_probs[g]);
  }
}
