# Load required packages
library(tidymodels)
library(dplyr)

# Load data
data(concrete)

# Split data into training and testing sets
set.seed(123)
concrete_split <- initial_split(concrete, prop = 0.7)
concrete_train <- training(concrete_split)
concrete_test <- testing(concrete_split)

ctrl_grid <- control_stack_grid()

# Define recipe
rec <- recipe(compressive_strength ~ ., data = concrete_train) %>% 
  step_normalize(all_predictors())

# Define KNN model
knn_mod <- nearest_neighbor( neighbors = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

# Define SVM model with RBF kernel
svm_mod <- svm_rbf(rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

# Define linear regression model
lm_mod <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

rf_spec <-
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 500
  ) %>%
  set_mode("regression") %>%
  set_engine("ranger")

# Define stacking model

cement_st <- stacks()

# Define tuning grid
tune_grid <- grid_regular(
  levels = 10,
  penalty = 10^seq(-4, 4, by = 0.5),
  rbf_sigma = 2^seq(-4, 4, by = 0.5),
  neighbors = seq(1, 10, by = 1)
)
cement_st %>%
  # add each of the models
  add_candidates(rf_res) %>%
  add_candidates(nnet_res) %>%
  add_candidates(cubist_res) %>%
  blend_predictions() %>% # evaluate candidate models
  fit_members() # fit non zero stacking coefficients

# Define workflow
wf <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(stack_mod)

# Tune hyperparameters
tune_res <- tune_grid %>% 
  tune_grid(
    wf,
    resamples = vfold_cv(concrete_train, v = 5),
    grid = tune_grid,
    metrics = metric_set(rmse),
    control = control_grid(verbose = TRUE)
  )

# Fit model with optimal hyperparameters
final_mod <- finalize_model(stack_mod, select_best(tune_res, "rmse"))

# Predict on test set
preds <- predict(final_mod, concrete_test) %>% 
  bind_cols(concrete_test)

# Evaluate performance on test set
metrics(preds, truth = strength, estimate = .pred) 











