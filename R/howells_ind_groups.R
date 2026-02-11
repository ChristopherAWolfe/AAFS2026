# Script 1 - Howells: Individual Groups x Sex

set.seed(1991)

## Packages
library(tidyverse)
library(cmdstanr)
library(brms)
library(magrittr)


## Data Prep
dat <- read.csv("data/howells.csv")

male <- dat %>% filter(Sex == "M") %>% 
  select(PopNum, Population, GOL, XCB, ZYB, BBH, BNL, MAB, AUB, NPH, 
         FMB, ASB, ZMB, NLH, NLB, OBB, OBH, EKB, DKB, FRC, PAC, OCC, FOL, 
         MDH)
female <- dat %>% filter(Sex == "F") %>% 
  select(PopNum, Population, GOL, XCB, ZYB, BBH, BNL, MAB, AUB, NPH, 
         FMB, ASB, ZMB, NLH, NLB, OBB, OBH, EKB, DKB, FRC, PAC, OCC, FOL, 
         MDH)

## Stan

### Model Code
mod_uni <- cmdstan_model("code/norm_IT.stan")
mod_multi <- cmdstan_model("code/multi_IT.stan")

### Male
male_mat <- data.frame(variable = character(),
                       cond_entropy = numeric(),
                       marg_entropy = numeric(),
                       mi = numeric())

male_vars <- male[,3:24]

for(i in 1:ncol(male_vars)){
  
  standat <- list(N = nrow(male),
                  G = 30,
                  group = male$PopNum,
                  y = male_vars[,i])
  
  fit <- mod_uni$sample(data = standat, refresh = 500, parallel_chains = 4)
  summ <- fit$summary(c("H_cond", "H_marg", "I_Y_G"))
  male_mat[i,1:4] <- c(colnames(male_vars)[i], summ[1,2], summ[2,2], summ[3,2])
  fit$save_object(file = paste0("results/male_uni_howells/",colnames(male_vars)[i], ".RDS"))
  rm(fit)
  gc()
  
}

male_mat[,2:4] <- male_mat[,2:4] / log(2) ## scale to bits
write.csv(male_mat, "results/male_uni_howells/summary_male_howells.csv")

### Female
female_mat <- data.frame(variable = character(),
                       cond_entropy = numeric(),
                       marg_entropy = numeric(),
                       mi = numeric())

female_vars <- female[,3:24]

for(i in 1:ncol(female_vars)){
  
  standat <- list(N = nrow(female),
                  G = 30,
                  group = female$PopNum,
                  y = female_vars[,i])
  
  fit <- mod_uni$sample(data = standat, refresh = 500, parallel_chains = 4)
  summ <- fit$summary(c("H_cond", "H_marg", "I_Y_G"))
  female_mat[i,1:4] <- c(colnames(female_vars)[i], summ[1,2], summ[2,2], summ[3,2])
  fit$save_object(file = paste0("results/female_uni_howells/",colnames(female_vars)[i], ".RDS"))
  rm(fit)
  gc()
  
}

female_mat[,2:4] <- female_mat[,2:4] / log(2) ## scale to bits
write.csv(female_mat, "results/female_uni_howells/summary_female_howells.csv")

### Comparison between M / F
male_mat$group <- "male"
female_mat$group <- "female"

it_metrics <- rbind(male_mat, female_mat)
it_metrics  %>% ggplot(aes(group, marg_entropy, fill = group)) + geom_col() + facet_wrap(vars(variable))

### Sex Combined
all_mat <- data.frame(variable = character(),
                      cond_entropy = numeric(),
                      marg_entropy = numeric(),
                      mi = numeric())

dat2 <- dat %>% select(PopNum, Population, GOL, XCB, ZYB, BBH, BNL, MAB, AUB, NPH, 
                       FMB, ASB, ZMB, NLH, NLB, OBB, OBH, EKB, DKB, FRC, PAC, OCC, FOL, 
                       MDH)

all_vars <- dat2[,3:24]

for(i in 1:ncol(all_vars)){
  
  standat <- list(N = nrow(dat2),
                  G = 30,
                  group = dat2$PopNum,
                  y = all_vars[,i])
  
  fit <- mod_uni$sample(data = standat, refresh = 500, parallel_chains = 4)
  summ <- fit$summary(c("H_cond", "H_marg", "I_Y_G"))
  all_mat[i,1:4] <- c(colnames(all_vars)[i], summ[1,2], summ[2,2], summ[3,2])
  fit$save_object(file = paste0("results/combined_uni_howells/",colnames(all_vars)[i], ".RDS"))
  rm(fit)
  gc()
  
}

all_mat[,2:4] <- all_mat[,2:4] / log(2) ## scale to bits
write.csv(all_mat, "results/combined_uni_howells/summary_combined_howells.csv")

### Multivariate

bform <- bf(mvbind(GOL, XCB, ZYB, BBH, BNL, MAB, AUB, NPH, 
                   FMB, ASB, ZMB, NLH, NLB, OBB, OBH, EKB, DKB, FRC, PAC, OCC, FOL, 
                   MDH) ~ 1 + (1 | gr(Population))) + set_rescor(FALSE)

fit_multi <- brm(bform, data = dat2, cores = 4, chains = 4, iter = 4000, warmup = 1000, backend = "cmdstanr")
saveRDS(fit_multi, "results/multi_howells/multi_howells.RDS")

fit_multi <- readRDS("results/multi_howells/multi_howells.RDS")

post <- as_draws_df(fit_multi)
D <- length(fit_multi$ranef$resp)
G <- length(unique(fit_multi$data$Population))
draws <- nrow(post)

outcome_names <- fit_multi$ranef$resp

#### Get group names from posterior names
r_vars <- grep("^r_Population__", names(post), value = TRUE)
group_names <- unique(str_match(r_vars, "\\[(.*?),Intercept\\]")[, 2])

#### Get posterior means for each response per group
mu_array <- array(NA, dim = c(draws, D, G))
for (d in seq_len(D)) {
  outcome <- outcome_names[d]
  bname <- paste0("b_", outcome, "_Intercept")
  
  for (g in seq_len(G)) {
    gname <- group_names[g]
    rname <- paste0("r_Population__", outcome, "[", gname, ",Intercept]")
    mu_array[, d, g] <- post[[bname]] + post[[rname]]
  }
}

#### Compute per-group means
mu_group_means <- apply(mu_array, c(2, 3), mean)  # D x G matrix

#### Get group probabilities (important for weighted calculations)
pop_counts <- table(fit_multi$data$Population)
pop_probs <- as.numeric(pop_counts / sum(pop_counts))

#### Compute marginal mean across groups (WEIGHTED by group size)
mu_marginal <- mu_group_means %*% pop_probs  # D x 1 vector

#### Compute between-group covariance (CORRECTED - weighted)
Sigma_between <- matrix(0, D, D)
for (g in seq_len(G)) {
  deviation <- mu_group_means[, g] - mu_marginal
  Sigma_between <- Sigma_between + pop_probs[g] * (deviation %*% t(deviation))
}

#### Compute within-group (residual) variance per response
sigma_vars <- grep("^sigma_", names(post), value = TRUE)
within_var <- sapply(sigma_vars, function(v) mean(post[[v]]^2))
Sigma_within <- diag(within_var)

#### Total marginal covariance
Sigma_marginal <- Sigma_between + Sigma_within

#### --- H(Y): Marginal Entropy ---
det_marginal <- det(Sigma_marginal)
if (det_marginal <= 0) stop("Marginal covariance matrix not positive definite")
H_Y <- 0.5 * log2((2 * pi * exp(1))^D * det_marginal)

#### --- H(Y|G): Conditional Entropy ---
det_within <- det(Sigma_within)
if (det_within <= 0) stop("Within-group covariance matrix not positive definite")
H_Y_given_G <- 0.5 * log2((2 * pi * exp(1))^D * det_within)

#### --- Mutual Information ---
I_Y_G <- H_Y - H_Y_given_G

#### --- Entropy of G ---
H_G <- -sum(pop_probs * log2(pop_probs))

#### --- Report ---
cat("Marginal entropy H(Y):     ", round(H_Y, 3), "bits\n")
cat("Conditional entropy H(Y|G):", round(H_Y_given_G, 3), "bits\n")
cat("Mutual information I(Y;G): ", round(I_Y_G, 3), "bits\n")
cat("Entropy of G:              ", round(H_G, 3), "bits\n")
cat("\nConstraint check: I(Y;G) ≤ H(G)? ", I_Y_G <= H_G, "\n")
cat("Difference (should be ≥ 0):", round(H_G - I_Y_G, 4), "\n")

multi_mat <- matrix(nrow = 4, ncol = 2)
multi_mat[1:4, 1] <- c("Marginal entropy H(Y)", "Conditional entropy H(Y|G)", 
                       "Mutual information I(Y;G)", "Entropy (G)")
multi_mat[1:4, 2] <- c(round(H_Y, 3), round(H_Y_given_G, 3), round(I_Y_G, 3),
                       round(H_G, 3))
multi_mat <- as.data.frame(multi_mat)
write.csv(multi_mat, "results/multi_howells/summary_multi_howells.csv")

## Classification

library(caret)

#### Univariate ALl
for(i in 1:ncol(all_vars)){
  
  dat3 <- as.data.frame(cbind(dat2$Population, all_vars[,i]))
  dat3$V2 <- as.numeric(dat3$V2)
  
  # Define training control for LOOCV
  train.control <- trainControl(method = "LOOCV")
  # Train the model using rpart
  model <- train(V1 ~ V2, data = dat3, method = "lda", preProcess = c("center", "scale"), trControl = train.control)
  predictions <- predict(model, dat3)
  cmat <- confusionMatrix(predictions, as.factor(dat3$V1))
  all_mat[i,5] <- cmat$overall[1]
  
}

colnames(all_mat)[5] <- "Accuracy"

write.csv(all_mat, "results/combined_uni_howells/summary_combined_howells.csv")

all_mat %<>% mutate(region = ifelse(variable %in% c("NLH", "NPH", "NLB", "OBH", "OBB", "DKB", 
                                                    "EKB", "FMB", "ZYB", "JUB"), "SP", 
                                    ifelse(variable %in% c("GOL", "NOL", "MDH", "XCB", "WFB",
                                                           "FRC", "PAC", "OCC", "AUB"), "NEU",
                                           ifelse(variable %in% c("ASB", "FOL", "FOB"), "BAS", "Cross"))))

all_mat %>% ggplot(aes(mi, Accuracy, color = region)) + geom_point() + theme_classic()

#### Multivariate All

train.control <- trainControl(method = "LOOCV")
# Train the model using rpart
model <- train(Population ~ GOL + XCB + ZYB + BBH + BNL + MAB + AUB + NPH + FMB +
                 ASB + ZMB + NLH + NLB + OBB + OBH + EKB + DKB + FRC + PAC + OCC +
                 FOL + MDH, data = dat2, method = "lda", preProcess = c("center", "scale"),trControl = train.control)
predictions <- predict(model, dat2)
cmat <- confusionMatrix(predictions, as.factor(dat2$Population))
cmat

multi_mat <- as.data.frame(multi_mat)
multi_mat[5,1] <- "Accuracy"
multi_mat[5,2] <- cmat$overall[1]
write.csv(multi_mat, "results/multi_howells/summary_multi_howells.csv")
