path2 <- "C:/Users/dms32073/OneDrive - University of Georgia/Documents/1. Research/01. Preliminary Data/09. Survival Analysis III//"
path3 <- "C:/Users/dms32073/OneDrive - University of Georgia/Documents/1. Research/03. Data Analysis/07. Survival analysis III//"
path4 <- "C:/Users/dms32073/OneDrive - University of Georgia/Documents/1. Research/04. Results/07. Survival Analysis III//"


# Sys.setenv(PATH = paste("C:/Users/your_username/Anaconda3/Scripts", Sys.getenv("PATH"), sep = ";"))
# system("conda --version")
# use_condaenv("ml", required = TRUE)

# reticulate::py_install("numpy==2.0.0") #if this works will have to put to top
# .rs.restartR()

# List all Conda environments
# conda_envs <- conda_list()
# print(conda_envs)
# reticulate::py_install("numpy==2.0.0", envname = "mlr3env")
# reticulate::py_install("randomForestSRC", envname = "mlr3env")

# .rs.restartR()

options(repos=c(
  mlrorg = 'https://mlr-org.r-universe.dev',
  raphaels1 = 'https://raphaels1.r-universe.dev',
  CRAN = 'https://cloud.r-project.org'
))
# install.packages(c("ggplot2", "mlr3benchmark", "mlr3pipelines", "mlr3proba", "mlr3tuning",
                   # "survivalmodels", "mlr3extralearners"))
# install.packages("Anaconda")
# library(Anaconda)
# use_condaenv("mlr3env", required = TRUE)


#using python in R
library(survivalmodels)
library(reticulate)


install
install_pycox(pip = TRUE, install_torch = TRUE)
install_keras(pip = TRUE, install_tensorflow = TRUE)

#just to set reproductability
set_seed(9524)

#To run these models once they’re installed, we’re going to use a different interface. {survivalmodels} has limited functionality, which is okay for basic model fitting/predicting, but neural networks typically require data pre-processing and model tuning, so instead we’re going to use {mlr3proba}, which is part of the {mlr3}¹⁸ family of packages and includes functionality for probabilistic supervised learning, of which survival analysis is a part of. {mlr3} packages use the R6¹⁹ interface for object-oriented machine learning in R. Full tutorials for mlr3 can be found in the {mlr3book} and there is also a chapter for survival analysis with {mlr3proba}²⁰.
library(mlr3)
library(mlr3proba)

## get the `whas` task from mlr3proba
whas <- tsk("whas")

## create our own task from the rats dataset
rats_data <- survival::rats
my_data <- read.csv(paste0(path2, "SA3Dat3C.3.csv"))

## convert characters to factors
rats_data$sex <- factor(rats_data$sex, levels = c("f", "m"))
rats <- TaskSurv$new("rats", rats_data, time = "time", event = "status")

my_data$MAN <- factor(my_data$MAN)
my_data$SPEC <- factor(my_data$SPEC) 
my_data$DAM <- factor(my_data$DAM) 
my_data$CRIFF <- factor(my_data$CRIFF) 
my_data$thin.Y.N <- factor(my_data$thin.Y.N) 
trees <- TaskSurv$new("trees", my_data, time = "t", event = "status")


## combine in list
tasks <- list(whas, rats)
my_tasks <- list(whas, trees)


################################################################################################
###########
###########             Now we look into all but DNNSurv ( - good start though)
###########
################################################################################################

# We are not going to specify a custom architecture for the models but will instead use the defaults, if you are familiar with PyTorch then you have the option to create your own architecture if you prefer by passing this to the custom_net parameter in the models.

library(paradox)

search_space <- ps(
  ## p_dbl for numeric valued parameters
  dropout = p_dbl(lower = 0, upper = 1),
  weight_decay = p_dbl(lower = 0, upper = 0.5),
  learning_rate = p_dbl(lower = 0, upper = 1),
  
  ## p_int for integer valued parameters
  nodes = p_int(lower = 1, upper = 32),
  k = p_int(lower = 1, upper = 4)
)

##DID NOT WORK!!!!! HAVE TO SEE!
# search_space$trafo <- function(x, param_set) {
#   x$num_nodes = rep(x$nodes, x$k)
#   x$nodes = x$k = NULL
#   return(x)
# }

library(mlr3tuning)

# create_autotuner <- function(learner) {
#   AutoTuner$new(
#     learner = learner,
#     search_space = search_space,
#     # resampling = rsmp("holdout"),
#     resampling = rsmp("cv", folds = 3),
#     measure = msr("surv.cindex"),
#     # terminator = trm("evals", n_evals = 2),
#     terminator = trm("evals", n_evals = 60),
#     tuner = tnr("random_search")
#   )
# }

#just test this run above whe ready
create_autotuner <- function(learner) {
  AutoTuner$new(
    learner = learner,
    search_space = search_space,
    resampling = rsmp("holdout"),
    measure = msr("surv.cindex"),
    terminator = trm("evals", n_evals = 2),
    tuner = tnr("random_search")
  )
}
## learners are stored in mlr3extralearners
library(mlr3extralearners)

## load learners
learners <- lrns(
  paste0("surv.",
         c(
           # "coxtime"
           # , "deephit"
           # , 
           "deepsurv"
           # , "loghaz"
           # ,"pchazard"
                    )),
  frac = 0.2, early_stopping = TRUE, #20% of nested training data will be held-back as validation data for early_stopping,
  epochs = 10,
  # epochs = 100, #maximum 10 epochs (number of times a training dataset passes through a learning algorithm)
  #as we are using early-stopping the number of epochs would usually be massively increased (say to 100 minimum)
  optimizer = "adam"
)

# apply our function
learners <- lapply(learners, create_autotuner)

library(mlr3pipelines)

create_pipeops <- function(learner) {
  po("encode") %>>% po("scale") %>>% po("learner", learner)
}

## apply our function
learners <- lapply(learners, create_pipeops)

## select holdout as the resampling strategy
resampling <- rsmp("cv"
                   , folds = 3
                   # , folds = 5
)

library(randomForestSRC)
library(pracma)


## add KM and CPH
learners <- c(learners, lrns(c("surv.kaplan", "surv.coxph")))
learner_rsf <- lrn("surv.rfsrc")
learners <- c(learners, learner_rsf)
design <- benchmark_grid(tasks, learners, resampling)
my_design <- benchmark_grid(my_tasks, learners, resampling)
bm <- benchmark(design)
my_bm <- benchmark(my_design)

## Aggreggate with Harrell's C and Integrated Graf Score
msrs <- msrs(c("surv.cindex", "surv.graf", "surv.brier"))
bm$aggregate(msrs)[, c(3, 4, 7, 8)]
my_bm$aggregate(msrs)[, c(3, 4, 7, 8)]























