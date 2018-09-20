"
Implement XGBoost with R and Caret package as following steps:
   1) Pre-implementing: Install Caret, load data
   2) Pre-processing
   3) Spliting data
   4) Feature selection
   5) Training model
   6) Parameter tunning 
"

"Pre-implementing"
#install.packages("caret", dependencies = c("Depends", "Suggests"))  #remove '#' symbol and run this line to install Caret package
#Loading caret package
library(ggplot2)
library(caret)

#Loading training data
data_frame <- read.csv('E:/environment/git/AI/ML/Data/Loan Data of Dreaming Housing Finance/train.csv', header = TRUE)


"Pre-processing data"
#Checking data information
head(data_frame, 3) #Show first 3 lines of data

#library(psych)
#describe(data_frame, na.rm = FALSE)
summary(data_frame)
#*note: Some resulted details:
#   - Gender 13 null, Married 3 null, Self_Employed 32 null, LoanAmount 22NA, Credit_History 50NA
#   - Gender Male(489) >> Female(112), Married Yes(493) > No(213), Self_Employed No(500) >> Yes(82)

#Rescale some vairables
data_frame$LoanAmount_log = log2(data_frame$LoanAmount)
data_frame$TotalAmount = data_frame$ApplicantIncome + data_frame$CoapplicantIncome
data_frame$TotalAmount_log = log2(data_frame$TotalAmount)
data_frame$LoanAmount <- NULL
data_frame$ApplicantIncome <- NULL
data_frame$CoapplicantIncome <- NULL
data_frame$TotalAmount <- NULL

#Replace NA values using KNN. Also centering and scaling numerical columns
preProcValues <- preProcess(data_frame, method = c("knnImpute", "center", "scale"))
library(RANN)
processed_data_frame <- predict(preProcValues, data_frame)
#Replace empty values
processed_data_frame$Gender <- replace(processed_data_frame$Gender, processed_data_frame$Gender == '', 'Male')
processed_data_frame$Married <- replace(processed_data_frame$Married, processed_data_frame$Married == '', 'Yes')
processed_data_frame$Dependents <- replace(processed_data_frame$Dependents, processed_data_frame$Dependents == '', '0')
processed_data_frame$Self_Employed <- replace(processed_data_frame$Self_Employed, processed_data_frame$Self_Employed == '', 'No') 

#Converting outcome variable to numeric
id <- processed_data_frame$Loan_ID
processed_data_frame$Loan_ID <- NULL
y <- factor(processed_data_frame$Gender)
processed_data_frame$Gender = as.numeric(y)
y <- factor(processed_data_frame$Married)
processed_data_frame$Married = as.numeric(y)
y <- factor(processed_data_frame$Dependents)
processed_data_frame$Dependents = as.numeric(y)
y <- factor(processed_data_frame$Education)
processed_data_frame$Education = as.numeric(y)
y <- factor(processed_data_frame$Self_Employed)
processed_data_frame$Self_Employed = as.numeric(y)
y <- factor(processed_data_frame$Property_Area)
processed_data_frame$Property_Area = as.numeric(y)
#*note: Possibly use dummy:
#dmy <- dummyVars(" ~ .", data = processed_data_frame, fullRank = T) #“fullrank=T” will create only (n-1) columns for a categorical column with n different levels.
#processed_data_frame <- data.frame(predict(dmy, newdata = processed_data_frame))

str(processed_data_frame) #take a look at data after processing


"Spliting data"
#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(processed_data_frame$Loan_Status, p = 0.75, list = FALSE)
trainSet <- processed_data_frame[index,]
testSet <- processed_data_frame[-index,]


"Feature selection"
#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs, method = "repeatedcv",
                   repeats = 3,
                   verbose = FALSE)
outcomeName <- 'Loan_Status'
predictors <- names(trainSet)[!names(trainSet) %in% outcomeName]
Loan_Pred_Profile <- rfe(trainSet[, predictors], trainSet[, outcomeName],rfeControl = control) #rfe: Recursive feature elimination
plot(Loan_Pred_Profile, type = c("g", "o"))
Loan_Pred_Profile    #remove '#' and run this line to view resulted detail
#*note: top 5 variables in the results:
#         Credit_History, TotalAmount_log, LoanAmount_log, Property_Area, Married
predictors <- c("Credit_History", "TotalAmount_log", "LoanAmount_log", "Property_Area", "Married")


"Traning Model first time"
model_xgboost <- train(trainSet[, predictors], trainSet[, outcomeName], method='xgbTree' )
predictions <- predict.train(object = model_xgboost, testSet[, predictors], type = "raw")
confusionMatrix(predictions, testSet[, outcomeName])

"Parameter tunning"
"As described in blog: https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
Parameters used in Xgboost
*There are three types of parameters: General Parameters, Booster Parameters and Task Parameters.
    -General parameters refers to which booster we are using to do boosting. The commonly used are tree or linear model
    -Booster parameters depends on which booster you have chosen
    -Learning Task parameters that decides on the learning scenario, for example, regression tasks may use different parameters with ranking tasks.
-General Parameters
    +silent : The default value is 0. You need to specify 0 for printing running messages, 1 for silent mode.
    +booster : The default value is gbtree. You need to specify the booster to use: gbtree (tree based) or gblinear (linear function).
    +num_pbuffer : This is set automatically by xgboost, no need to be set by user.
    +num_feature : This is set automatically by xgboost, no need to be set by user.
-Booster Parameters -The tree specific parameters –
    +eta : The default value is set to 0.3. You need to specify step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative. The range is 0 to 1. Low eta value means model is more robust to overfitting.
    +gamma : The default value is set to 0. You need to specify minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be. The range is 0 to ∞. Larger the gamma more conservative the algorithm is.
    +max_depth : The default value is set to 6. You need to specify the maximum depth of a tree. The range is 1 to ∞.
    +min_child_weight : The default value is set to 1. You need to specify the minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. The range is 0 to ∞.
    +max_delta_step : The default value is set to 0. Maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.The range is 0 to ∞.
    +subsample : The default value is set to 1. You need to specify the subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting. The range is 0 to 1.
    +colsample_bytree : The default value is set to 1. You need to specify the subsample ratio of columns when constructing each tree. The range is 0 to 1.
-Linear Booster Specific Parameters
    +lambda and alpha : These are regularization term on weights. Lambda default value assumed is 1 and alpha is 0.
    +lambda_bias : L2 regularization term on bias and has a default value of 0.
-Learning  Task Parameters
    +base_score : The default value is set to 0.5 . You need to specify the initial prediction score of all instances, global bias.
    +objective : The default value is set to reg:linear . You need to specify the type of learner you want which includes linear regression, logistic regression, poisson regression etc.
    +eval_metric : You need to specify the evaluation metrics for validation data, a default metric will be assigned according to objective( rmse for regression, and error for classification, mean average precision for ranking
    +seed : As always here you specify the seed to reproduce the same set of outputs.    
"
fitControl <- trainControl(method = "repeatedcv", number = 8, repeats =4)
modelLookup(model = 'xgbTree') #view tunable parameters
grid <- expand.grid(eta = c(0.1, 0.2, 0.3, 0.4), max_depth = c(5:9), colsample_bytree = c(0.2, 0.5, 0.7, 1), min_child_weight= c(1,2,4,6), nrounds=c(5,10), gamma=c(0:4))
#Retrain the model
model_xgboost <- train(trainSet[, predictors], trainSet[, outcomeName], method = 'xgbTree', trControl = fitControl, tuneGrid = grid)
#Checking result
predictions <- predict.train(object = model_xgboost, testSet[, predictors], type = "raw")
confusionMatrix(predictions, testSet[, outcomeName])