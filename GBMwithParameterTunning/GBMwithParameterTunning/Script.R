"
Implement GBM with R and Caret package as following steps:
   1) Pre-implementing: Install Caret, load data
   2) Pre-processing
   3) Spliting data
   4) Feature selection
   5) Training model
   6) Parameter tunning with tuneGrid & tuneLength
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
model_gbm <- train(trainSet[, predictors], trainSet[, outcomeName], method = 'gbm')
predictions <- predict.train(object = model_gbm, testSet[, predictors], type = "raw")
confusionMatrix(predictions, testSet[, outcomeName])
print(model_gbm)
#*note: by default Caret uses 3 random values of each tunable parameter and use the cross-validation results to find the best set of parameters for that algorithm. 
#Checking variable importance 
plot(varImp(object = model_gbm))
#*note: importance distribution is biased to some variables

"Parameter tunning"
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

#Using tuneGrid
modelLookup(model = 'gbm') #view tunable parameters
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
#Retrain the model
model_gbm <- train(trainSet[, predictors], trainSet[, outcomeName], method = 'gbm', trControl = fitControl, tuneGrid = grid)
#Checking result
plot(model_gbm)
predictions <- predict.train(object = model_gbm, testSet[, predictors], type = "raw")
confusionMatrix(predictions, testSet[, outcomeName])

#Using tuneLength
#Retrain the model
model_gbm <- train(trainSet[, predictors], trainSet[, outcomeName], method = 'gbm', trControl = fitControl, tuneLength = 10) #set tuneLength = 10 instead of 3
#Checking result
plot(model_gbm)
predictions <- predict.train(object = model_gbm, testSet[, predictors], type = "raw")
confusionMatrix(predictions, testSet[, outcomeName])
#Checking variable importance 
plot(varImp(object = model_gbm))
#*note: importance distribution is more equally. But the result are not improved much
