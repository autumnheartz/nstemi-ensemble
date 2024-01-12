################################################################################
rm(list = ls()) # clear workspace
current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path) 
setwd(current_working_dir) #set current directory to source file location

# PACKAGES
library(mlbench) #for ML models
library(caret) #for ML models
library(pROC) #for ROC plot
library(PRROC) #for PR Curve
library(e1071) #for confusion matrix
library(tibble) #for data manipulation
library(forcats) #for data manipulstion
library(doParallel) #parallel computing
library(readxl)

###############################STEP 1 - Load the data###########################
# Load features metadata
allMeta_list <- read_excel("features_metadata.xlsx", sheet = "inHospital")
allMeta_sel_list<- allMeta_list[which(allMeta_list$svm_fet %in% c("other", "yes")),] 
feats_sel_list <- allMeta_sel_list$Varnames[which(allMeta_sel_list$Types == "features")] #selected features

# Load the raw data
train_data_ALL <- read.csv("train_data.csv",
                           header = T)
test_data_ALL <- read.csv("test_data.csv",
                              header = T)

# Extract relevant features
train_data <- train_data_ALL[,c(feats_sel_list,"ptoutcome")]
test_data <- test_data_ALL[,c(feats_sel_list,"ptoutcome")]
timi_test_data <- test_data_ALL[,c("timiscorenstemi", "ptoutcome")]

###########################STEP 2 - Data Preprocessing##########################
# Change categorical non-binary features to factor
cat_nb_fet <- allMeta_sel_list$Varnames[which(allMeta_sel_list$Cat_Cont_Ord == "categorical_nonBinary")]
train_data[,cat_nb_fet] <- lapply(train_data[,cat_nb_fet], as.factor)
test_data[,cat_nb_fet] <- lapply(test_data[,cat_nb_fet], as.factor)

# Change categorical binary features to 0 & 1
cat_fet <- allMeta_sel_list$Varnames[which(allMeta_sel_list$Cat_Cont_Ord == "categorical_binary")]
train_data[,cat_fet] <- ifelse(train_data[,cat_fet] == 1, 1, 0)
train_data[,cat_fet] <- lapply(train_data[,cat_fet], as.factor)
test_data[,cat_fet] <- ifelse(test_data[,cat_fet] == 1, 1, 0)
test_data[,cat_fet] <- lapply(test_data[,cat_fet], as.factor)

# Normalize continuous & ordinal features : Standardization (mean 0, sd 1)
cont_fet <- allMeta_sel_list$Varnames[which(allMeta_sel_list$Cat_Cont_Ord %in% c("continuous", 
                                                                                 "ordinal"))]
zscore_scale <- preProcess(train_data[,cont_fet],
                           method=c("center", "scale"))
train_data[,cont_fet] <- predict(zscore_scale,
                                 train_data[,cont_fet])
test_data[,cont_fet] <- predict(zscore_scale,
                                    test_data[,cont_fet])

# Outcome
train_data$ptoutcome <- as.factor(train_data$ptoutcome)
test_data$ptoutcome <- as.factor(test_data$ptoutcome)

summary(train_data)
summary(test_data)

###############################STEP 3 - Data Conversion##########################
# Change the data to matrix
X_train <- train_data[,colnames(train_data) != "ptoutcome"]
X_test <-test_data[,colnames(test_data) != "ptoutcome"]

Y_train <- train_data[, c("ptoutcome")]
Y_test <- test_data[,c("ptoutcome")]

####################STEP4 - Individual Model Building###########################
cl <- makeCluster(3)
registerDoParallel(cl)
getDoParWorkers()

# Cross validation
set.seed(333)
fitControl <- trainControl(method = "cv",
                           number = 10,
                           search="random", #random search
                           savePredictions = 'final', # To save out of fold predictions for best parameter combinantions
                           classProbs = T, # To save the class probabilities of the out of fold predictions
                           summaryFunction = twoClassSummary,
                           index = createFolds(train_data$ptoutcome,10),
                           allowParallel = T)

# Model Building
# SVMLinear, SVMRadial, RF, XGBoost, NaiveBayes
modelNames <- c("SVMLinear", "SVMRadial", "RF", "XGBoost", "NaiveBayes")
modelIndi <- vector(mode = 'list', length = length(modelNames))
names(modelIndi) <- modelNames

# SVM Linear Kernel
modelIndi[[1]] <- train(form = ptoutcome~.,
                        data = train_data,
                        method = "svmLinear",
                        metric = "ROC",
                        trControl = fitControl,
                        tuneLength = 10)

# SVM Radial Kernel
modelIndi[[2]] <- train(form = ptoutcome~.,
                        data = train_data,
                        method = "svmRadial",
                        metric = "ROC",
                        trControl = fitControl,
                        tuneLength = 10)

# RF
modelIndi[[3]] <- train(form = ptoutcome~.,
                        data = train_data,
                        method = "rf",
                        ntree = 1000,
                        metric = "ROC",
                        trControl = fitControl,
                        tuneLength = 10)

# XGBoost 
modelIndi[[4]] <- train(form = ptoutcome~.,
                        data = train_data,
                        method = "xgbTree",
                        metric = "ROC",
                        trControl = fitControl,
                        tuneLength = 10)

modelIndi[[5]] <- train(form = ptoutcome~.,
                        data = train_data,
                        method = "nb",
                        metric = "ROC",
                        trControl = fitControl,
                        tuneLength = 10)
beepr::beep()
saveRDS(modelIndi[modelNames],"1_modelIndi.rds")

################################################################################
# Check Correlation Matrix of Accuracy
modelAll <- readRDS("1_modelIndi.rds")
modelNames <- c("SVMLinear", "SVMRadial", "RF", "XGBoost", "NaiveBayes")
results <- resamples(modelAll[modelNames])
modelCor(results)
dotplot(results)
summary(results)

# Predicting the out of fold prediction probabilities for training data
pred_prob_train <- vector(mode = 'list', length = length(modelNames))
names(pred_prob_train) <- modelNames

for (i in 1:length(pred_prob_train)) {
  pred_prob_train[[i]] <- modelAll[[i]]$pred$Death[order(modelAll[[i]]$pred$rowIndex)]
}
pred_prob_train <- as.data.frame(pred_prob_train)
pred_prob_train$ptoutcome <- modelAll[[1]]$pred$obs[order(modelAll[[1]]$pred$rowIndex)]

####################STEP5 - Ensemble Stacking Model Building###########################
# Cross validation
set.seed(333)
stackControl <- trainControl(method = "cv",
                             number = 10,
                             search="random", #random search
                             savePredictions = 'final', 
                             classProbs = T,
                             summaryFunction = twoClassSummary)

# Metalearner stacking model
modelAll$Ensemble_GLM <- train(pred_prob_train[,modelNames],
                               pred_prob_train[,"ptoutcome"],
                               method='glm',
                               trControl=stackControl,
                               metric = 'ROC')


stopCluster(cl)
beepr::beep()
saveRDS(modelAll,"1_modelAll.rds")

modelAll <- readRDS("1_modelAll.rds")

# Ensemble model summary
summary(modelAll$Ensemble_GLM)

# Define the index number for base and meta learner
idx_modelBase <- 1:5
idx_modelMeta <- 6

#########################STEP 6 - Model Evaluation (NSTEMI)########################
modelNames2_withTIMI <- c(modelNames, "Ensemble_GLM", "TIMI")

#Predicting probabilities for the test data
pred_prob_test <- vector(mode = 'list', length = length(modelNames2_withTIMI))
pred_class_test <- vector(mode = 'list', length = length(modelNames2_withTIMI))
names(pred_prob_test) <- modelNames2_withTIMI
names(pred_class_test) <- modelNames2_withTIMI

for (i in 1:length(pred_prob_test)) {
  # Base model
  if(i %in% idx_modelBase){
    pred_prob_test[[i]] <- predict(modelAll[[i]], 
                                         newdata = X_test, 
                                         type = 'prob')$Death
    pred_class_test[[i]] <- predict(modelAll[[i]], 
                                          newdata = X_test)
  }
  # Meta model
  else if (i %in% idx_modelMeta){
    temp_base_prob <- as.data.frame(pred_prob_test[idx_modelBase])
    pred_prob_test[[i]] <- predict(modelAll[[i]], 
                                         temp_base_prob,
                                         type = 'prob')$Death
    pred_class_test[[i]] <- predict(modelAll[[i]],
                                          temp_base_prob)
  }
  #TIMI
  else {
    pred_prob_test[[i]] <- timi_test_data$timiscorenstemi
    pred_class_test[[i]] <- ifelse(timi_test_data$timiscorenstemi >5, 
                                         "Death", "Alive")
  }
}
pred_prob_test <- as.data.frame(pred_prob_test)
pred_class_test <- as.data.frame(pred_class_test)
pred_prob_test$ptoutcome <- Y_test
pred_class_test$ptoutcome <- Y_test

# ROC Curve & AUC
roc_test <- vector(mode = 'list', length = length(modelNames2_withTIMI))
auc_test <- vector(mode = 'list', length = length(modelNames2_withTIMI))
names(roc_test) <- modelNames2_withTIMI
names(auc_test) <- modelNames2_withTIMI

for (i in 1:length(roc_test)) {
  #For TIMI
  if(i == length(modelNames2_withTIMI)){
    roc_test[[i]] <- roc(as.factor(pred_prob_test$ptoutcome),
                               timi_test_data$timiscorenstemi,
                               levels = c("Alive", "Death"))
    auc_test[[i]] <- round(roc_test[[i]]$auc,3)
  } else {
    roc_test[[i]] <- roc(as.factor(pred_prob_test$ptoutcome),
                               pred_prob_test[[modelNames2_withTIMI[i]]],
                               levels = c("Alive", "Death"))
    auc_test[[i]] <- round(roc_test[[i]]$auc,3)
  }
}
auc_test <- as.data.frame(auc_test)
auc_test

# PR Curve (Precision-Recall)
pr_curve_test <- list(pos=vector(mode = 'list', length = length(modelNames2_withTIMI)),
                            neg=vector(mode = 'list', length = length(modelNames2_withTIMI)),
                            auc=vector(mode = 'list', length = length(modelNames2_withTIMI)))
names(pr_curve_test$pos) <- modelNames2_withTIMI
names(pr_curve_test$neg) <- modelNames2_withTIMI
names(pr_curve_test$auc) <- modelNames2_withTIMI

for (i in 1:length(modelNames2_withTIMI)) {
  if(i == length(modelNames2_withTIMI)){
    pr_curve_test$pos[[i]] <- timi_test_data$timiscorenstemi[timi_test_data$ptoutcome 
                                                                        == "Death"]
    pr_curve_test$neg[[i]] <- timi_test_data$timiscorenstemi[timi_test_data$ptoutcome 
                                                                        == "Alive"]
    pr_curve_test$auc[[i]] <- pr.curve(pr_curve_test$pos[[i]],
                                             pr_curve_test$neg[[i]])
  } else {
    pr_curve_test$pos[[i]] <- pred_prob_test[[modelNames2_withTIMI[i]]][pred_prob_test$ptoutcome == "Death"]
    pr_curve_test$neg[[i]] <- pred_prob_test[[modelNames2_withTIMI[i]]][pred_prob_test$ptoutcome == "Alive"]
    pr_curve_test$auc[[i]] <- pr.curve(pr_curve_test$pos[[i]],
                                             pr_curve_test$neg[[i]])
  }
}
pr_curve_test$pos <- as.data.frame(pr_curve_test$pos)
pr_curve_test$neg <- as.data.frame(pr_curve_test$neg)
pr_curve_test

# Confusion matrix
cm_test <- vector(mode = 'list', length = length(modelNames2_withTIMI))
names(cm_test) <- modelNames2_withTIMI
for (i in 1:length(modelNames2_withTIMI)) {
  cm_test[[i]] <- confusionMatrix(data = as.factor(pred_class_test[, names(pred_class_test) == 
                                                                                 modelNames2_withTIMI[[i]]]),
                                        reference = pred_class_test$ptoutcome,
                                        positive="Death")
}

# Compile result
perf_result <- vector(mode = 'list', length = length(modelNames2_withTIMI))
names(perf_result) <- modelNames2_withTIMI

for (i in 1:length(modelNames2_withTIMI)) {
  perf_result[[i]] <- data.frame(auc = paste0(auc_test[modelNames2_withTIMI[i]], 
                                                    " (", round(as.numeric(ci.auc(roc_test[[modelNames2_withTIMI[i]]]))[1],3),
                                                    " - ", round(as.numeric(ci.auc(roc_test[[modelNames2_withTIMI[i]]]))[3],3),
                                                    ")"),
                                       accuracy=paste0(round(cm_test[[modelNames2_withTIMI[i]]]$overall["Accuracy"],3),
                                                       " (", round(cm_test[[modelNames2_withTIMI[i]]]$overall["AccuracyLower"],3),
                                                       " - ", round(cm_test[[modelNames2_withTIMI[i]]]$overall["AccuracyUpper"],3),
                                                       ")"),
                                       sensitivity=round(cm_test[[modelNames2_withTIMI[i]]]$byClass["Sensitivity"],3),
                                       specificity=round(cm_test[[modelNames2_withTIMI[i]]]$byClass["Specificity"],3),
                                       ppv=round(cm_test[[modelNames2_withTIMI[i]]]$byClass["Pos Pred Value"],3),
                                       npv=round(cm_test[[modelNames2_withTIMI[i]]]$byClass["Neg Pred Value"],3),
                                       mcnemar=round(cm_test[[modelNames2_withTIMI[i]]]$overall["McnemarPValue"],3),
                                       bal_acc=round(cm_test[[modelNames2_withTIMI[i]]]$byClass["Balanced Accuracy"],3),
                                       pr_auc=round(pr_curve_test$auc[[modelNames2_withTIMI[i]]]$auc.integral,3))
}
perf_result <- as.data.frame(data.table::rbindlist(perf_result))
perf_result <- cbind(model=modelNames2_withTIMI, perf_result)
View(perf_result)

# Save the result
write.csv(pred_prob_test,"1_out_NSTEMI_Ensemble_SVMFet_resultProb.csv",row.names = F)
write.csv(perf_result,"1_out_NSTEMI_Ensemble_SVMFet_resultPerf.csv",row.names = F)

save.image("1_data.RDATA")
# load("1_data.RDATA")
