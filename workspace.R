

# globals ----
{
  library(tidyverse)
  library(rsample)
  
  genData <- function(nrow = 1000, ncol = 1000, .min = 0, .max = 1){
    # Generating test data. 
    # To prevent the need of normalizing the data use the defaults for min and max 
    library(foreach)
    library(iterators)
    data <-  foreach( c=1:ncol, 
                      .init = tibble( y = runif(nrow, min = .min, max = .max) ), 
                      .combine = cbind) %do% {
                        col.name  <- paste0('x.', c)
                        tmp <- tibble( x = runif(nrow, min = .min, max = .max) )
                        names(tmp) <- col.name
                        return(tmp)
                      }
    return (data)
  }
  
  testData   <- genData()
  data_split <- testData %>% rsample::initial_time_split(prop = 0.9)
  train_tbl <- training(data_split)
  test_tbl  <- testing(data_split)
}

# h2o ----
{
  h2o_automl_test <- function(data, .max_runtime_secs = 60 * 10, test = test_tbl){
    library(h2o)
    library(lime)
    h2o.init()
    
    aml <- h2o.automl( x = grep( pattern = 'x.', x = names(data)), #indices of features
                       y = grep( pattern = 'y' , x = names(data)), #indices of target (will be always 1)
                       training_frame = as.h2o(data),
                       nfolds = 5,
                       max_runtime_secs = .max_runtime_secs )
    model <- aml@leaderboard %>% 
      as_tibble() %>% 
      slice(1) %>% 
      pull(model_id) %>% 
      h2o.getModel()
    h2o.performance(model = model, xval = TRUE)
    dir.create( path = model_filepath <- paste('models', 'h2o', 'automl', sep = '/'), showWarnings = F, recursive = T)
    h2o.saveModel(model, model_filepath, force = TRUE)
    
    #explainer   <- lime (data, model)
    #explanation <- explain(test, explanation, n_features = 5, feature_select = "highest_weights")
    #p <- plot_explanations(explanation) # not working :(
    #ggplot2::ggsave(filename = paste(model_filepath, lime.plot.png, sep = '/'), plot = p)
    h2o.shutdown(prompt = F)  
  }
}
# autokeras ----
{ 
  setUpAutokeras <- function(){
    library(reticulate)
    if( !('r-reticulate' %in% reticulate::conda_list(conda = '/opt/conda/bin/conda')$name) ){          
      reticulate::conda_create(envname = 'r-reticulate', packages = 'python=3.6', conda = '/opt/conda/bin/conda')
    }
    reticulate::use_condaenv(condaenv = 'r-reticulate', conda = '/opt/conda/bin/conda')
    
    library(autokeras)
    library(keras)
    autokeras::install_autokeras( method = 'conda',                                        
                                  conda = '/opt/conda/bin/conda',                          
                                  tensorflow = '2.1.0-gpu',                                
                                  version = 'default' )
  }
  
  autokeras_test <- function(data, .max_trials = 10, .epochs = 10, test = test_tbl){
    library(autokeras)
    library(keras)
    library(reticulate)
    library(ggplot2)
    reticulate::use_condaenv(condaenv = 'r-reticulate', conda = '/opt/conda/bin/conda')
    
    reg <- model_structured_data_regressor(
      column_names = grep(pattern = 'x.', x = names(data), value = T),
      loss = "mean_squared_error",
      max_trials = .max_trials,
      objective = "val_loss",
      overwrite = TRUE,
      seed = runif(1, 0, 1e+06) ) 
    
    tensorboard("models/logs/run_autokeras")
    model <- 
      fit( object = reg, 
           x = as_tibble(data[ , grep( pattern = 'x.', x = names(data))]), #tibble of features
           y = as_tibble(data[ , grep( pattern = 'y' , x = names(data))]), # target values
           epochs = .epochs,
           callbacks = list (
             keras::callback_tensorboard("models/logs/run_autokeras"),
             #           keras::callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.01),
             keras::callback_early_stopping(min_delta = 0.0001, restore_best_weights = TRUE, verbose = T)
           ),
           validation_split = 0.2
      )
    
    # Predict with the best model
    predicted <- tibble(idx      = seq(1:nrow(test)), 
                        value    = predict(model, test[ , grep( pattern = 'x.', x = names(data))]), 
                        variable = 'predicted' )
    result <- rbind( tibble(idx      = seq(1:nrow(test)), 
                            value    = test$y, 
                            variable = 'value' ),
                     predicted ) %>% 
      arrange(idx) %>%
      select(idx, variable, value)
    
    p <- result %>% ggplot(aes(idx, value, colour = variable)) + geom_line()
    # Evaluate the best model with testing data
    model %>% evaluate(
      x = as_tibble(test_tbl[ , grep( pattern = 'x.', x = names(data))]), #tibble of features
      y = as_tibble(test_tbl[ , grep( pattern = 'y' , x = names(data))])  # target values
    )
    
    # save the model
    dir.create( path = dirname( model_filepath <- paste('models', 'autokeras', 'autokeras.model', sep = '/') ), showWarnings = F, recursive = T) 
    autokeras::save_model(autokeras_model = model, filename = model_filepath)
    #nvidia-smi pmon -c 1 --select m | grep rsession
  }
}


# keras & tensorflow ----
{
  # generators ----
  {
    # data preparation
    # comming from https://blogs.rstudio.com/tensorflow/posts/2017-12-20-time-series-forecasting-with-recurrent-neural-networks/
    
    generator <- function(data, lookback, delay, min_index, max_index,
                          shuffle = FALSE, batch_size = 128, step = 1) {
      if (is.null(max_index))
        max_index <- nrow(data) - delay - 1
      i <- min_index + lookback
      function() {
        if (shuffle) {
          rows <- sample(c((min_index+lookback):max_index), size = batch_size)
        } else {
          if (i + batch_size >= max_index)
            i <<- min_index + lookback
          rows <- c(i:min(i+batch_size-1, max_index))
          i <<- i + length(rows)
        }
        
        samples <- array(0, dim = c(length(rows),
                                    lookback / step,
                                    dim(data)[[-1]]))
        targets <- array(0, dim = c(length(rows)))
        
        for (j in 1:length(rows)) {
          indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                         length.out = dim(samples)[[2]])
          samples[j,,] <- data[indices,]
          targets[[j]] <- data[rows[[j]] + delay, 1] # target variable must always be the first column !!!!
        }           
        list(samples, targets)
      }
    }
    
    lookback   =  5 # Observations will go back 10 days.
    step       =  1 # Observations will be sampled at one data point per day.
    delay      =  0 # uninteresting for the tests
    batch_size = 30 # 
  }
  
  basicTFtest <- function(data = testData){
    library(reticulate)
    use_condaenv(condaenv = 'r-reticulate', conda = '/opt/conda/bin/conda')
    train_gen <- generator(
      as.matrix(data),
      lookback = lookback,
      delay = delay,
      min_index = 1,
      max_index = floor(nrow(data)*(8/10)),
      shuffle = FALSE,
      step = step,
      batch_size = batch_size
    )
    
    
    val_gen <- generator(
      as.matrix(data),
      lookback = lookback,
      delay = delay,
      min_index = floor(nrow(data)*(8/10)) + 1,
      max_index = floor(nrow(data)*(9/10)),
      step = step,
      batch_size = batch_size
    )
    
    
    test_gen <- generator(
      as.matrix(data),
      lookback = lookback,
      delay = delay,
      min_index = floor(nrow(data)*(9/10)) + 1,
      max_index = nrow(data),
      step = step,
      batch_size = batch_size
    )
    
    
    # # How many steps to draw from val_gen in order to see the entire validation set
    val_steps <- (floor(nrow(data)*(9/10)) - floor(nrow(data)*(8/10)) + 1 - lookback) / batch_size
    # 
    # # How many steps to draw from test_gen in order to see the entire test set
    test_steps <- (nrow(data) - floor(nrow(data)*(9/10)) + 1 - lookback) / batch_size
    
    library(keras)
    model <- keras_model_sequential() %>% 
      layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
      layer_dense(units = 32, activation = "relu") %>% 
      layer_dense(units = 1)
    # at this point rsession needs 10GB more of GPU memory
    
    model %>% compile(
      optimizer = optimizer_rmsprop(),
      loss = "mae"
    )
    
    tensorboard("models/logs/run_basicTF")
    history <- model %>% fit_generator(
      train_gen,
      steps_per_epoch = 500,
      epochs = 20,
      validation_data = val_gen,
      validation_steps = val_steps,
      callbacks = callback_tensorboard("models/logs/run_basicTF")
    )
  }
}




# run tests ----
#  train_tbl %>% h2o_automl_test()

# setUpAutokeras() # run this once to install the right tensorflow version 
# train_tbl %>% autokeras_test()

# basicTFtest()