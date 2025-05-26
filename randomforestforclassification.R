#Machine learning models: Random FOREST
ews_data <- read_xlsx("ews_data.xlsx")
ews_data<-as.data.frame(ews_data)
ews_data$ID<-as.factor(ews_data$ID)
ews_data$SALE <- as.numeric(ews_data$SALE)
str(ews_data)

# original time based Data split
# Train: TIME_ORIG 2–9
train_data <- ews_data %>% filter(TIME_ORIG >= 2 & TIME_ORIG <= 9)
train_data<-as.data.frame(train_data)
# Validation: TIME_ORIG 10–13
valid_data <- ews_data %>% filter(TIME_ORIG >= 10 & TIME_ORIG <= 13)
valid_data<-as.data.frame(valid_data)
# Test: TIME_ORIG 14–17
test_data  <- ews_data %>% filter(TIME_ORIG >= 14 & TIME_ORIG <= 17)
test_data<-as.data.frame(test_data)


##################RANDOM FOREST#######################
#set SALE as factor. Random effect is not valid for RF. 
train_data$SALE <- factor(ifelse(train_data$SALE == 1, "yes", "no"), levels = c("no", "yes"))
valid_data$SALE <- factor(ifelse(valid_data$SALE == 1, "yes", "no"), levels = c("no", "yes"))
test_data$SALE  <- factor(ifelse(test_data$SALE  == 1, "yes", "no"), levels = c("no", "yes"))
table(train_data$SALE)
library(randomForest)
modelLookup(randomForest())

#the full model to see the variable importance:
set.seed(123)  #for reproducibility 
rf_model_full <- randomForest(SALE ~lag1_AP + lag1_BUDGET + lag1_SBUDGET +
                           lag1_TSV + lag1_TPC + lag1_URR + lag1_totrev +
                           TIME,
                         data = train_data,ntree = 500,importance = TRUE)
importance_values <- importance(rf_model_full)
print(importance_values)
varImpPlot(rf_model_full,
           main = "Variable Importance (Random Forest)",
           type = 2,              
           n.var = min(10, nrow(importance(rf_model_full))), 
           scale = TRUE)

#RF model:
#training data:
set.seed(153)
rf_model <- randomForest(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = train_data,
  ntree = 500,mtry = 2,importance = TRUE)
rf_model
plot(rf_model)


# PLOT RF - OOB Error by Trees

# 1. Layout: 2 panel (1 grafik + 1 legend)
layout(matrix(c(1, 2), nrow = 1), widths = c(4, 1))

# 2. Sol panel: Grafik
par(mar = c(5, 4, 4, 1))  # alt, sol, üst, sağ
plot(rf_model, main = "Random Forest OOB Error by Trees")

# 3. Sağ panel: Legend için boş panel
par(mar = c(0, 0, 0, 0))
plot.new()

# 4. Legend çizimi
legend("center",
       legend = c("OOB Error (Overall)",
                  "Class 1 Error (SALE = 1)",
                  "Class 0 Error (SALE = 0)"),
       col = c("black", "red", "green"),
       lty = 1,
       lwd = 1.2,
       cex = 0.9,
       bty = "n")



# OOB hata oranlarının tamamı
oob_errors <- rf_model$err.rate[, "OOB"]

# Minimum OOB hata ve karşılık gelen ağaç sayısı
min_oob_error <- min(oob_errors)
best_ntree <- which.min(oob_errors)

# Yazdır
cat(sprintf("Minimum OOB Error: %.3f at ntree = %d\n", min_oob_error, best_ntree))
#Minimum OOB Error: 0.239 at ntree = 106


###############------------------------------

#train data performances:
train_probs_rf <- predict(rf_model, newdata = train_data, type = "prob")[, "yes"]
train_preds_rf <- factor(ifelse(train_probs_rf >= 0.5, "yes", "no"), levels = c("no", "yes"))

# AUC (from pROC package)
roc_train_rf <- pROC::roc(response = train_data$SALE, predictor = train_probs_rf)
auc_train_rf <- pROC::auc(roc_train_rf)

# F1 score (from yardstick package)
f1_train_rf <- yardstick::f_meas_vec(truth = train_data$SALE, estimate = train_preds_rf)

# Accuracy (based on confusion matrix)
conf_matrix_rf <- caret::confusionMatrix(train_preds_rf, train_data$SALE)
accuracy_train_rf <- conf_matrix_rf$overall["Accuracy"]

# Print results
cat(sprintf("Train AUC (RF): %.3f\n", auc_train_rf))
cat(sprintf("Train F1 Score (RF): %.3f\n", f1_train_rf))
cat(sprintf("Train Accuracy (RF): %.3f\n", accuracy_train_rf))


#validation set:
# 1. probability prediction (Random Forest)
valid_probs_rf <- predict(rf_model, newdata = valid_data, type = "prob")[, "yes"]
valid_preds_rf <- factor(ifelse(valid_probs_rf >= 0.5, "yes", "no"), levels = c("no", "yes"))

# 2. AUC (pROC)
roc_valid_rf <- pROC::roc(response = valid_data$SALE, predictor = valid_probs_rf)
auc_valid_rf <- pROC::auc(roc_valid_rf)

# 3. F1 Score (yardstick)
f1_valid_rf <- yardstick::f_meas_vec(truth = valid_data$SALE, estimate = valid_preds_rf)

# 4. Accuracy (caret)
conf_matrix_valid_rf <- caret::confusionMatrix(valid_preds_rf, valid_data$SALE)
accuracy_valid_rf <- conf_matrix_valid_rf$overall["Accuracy"]

# 5. Print results
cat(sprintf("VALIDATION SET (RF):\nAUC: %.3f | F1 Score: %.3f | Accuracy: %.3f\n",
            auc_valid_rf, f1_valid_rf, accuracy_valid_rf))

#test set:

# 1. probability calculation (Random Forest)
test_probs_rf <- predict(rf_model, newdata = test_data, type = "prob")[, "yes"]

# 2. class prediction (0.5 threshold ile)
test_preds_rf <- factor(ifelse(test_probs_rf >= 0.5, "yes", "no"), levels = c("no", "yes"))

# 3. AUC (pROC)
roc_test_rf <- pROC::roc(response = test_data$SALE, predictor = test_probs_rf)
auc_test_rf <- pROC::auc(roc_test_rf)

# 4. F1 Score (yardstick)
f1_test_rf <- yardstick::f_meas_vec(truth = test_data$SALE, estimate = test_preds_rf)

# 5. Accuracy (caret)
conf_matrix_test_rf <- caret::confusionMatrix(test_preds_rf, test_data$SALE)
accuracy_test_rf <- conf_matrix_test_rf$overall["Accuracy"]

# 6. Print results
cat(sprintf("TEST SET (RF):\nAUC: %.3f | F1 Score: %.3f | Accuracy: %.3f\n",
            auc_test_rf, f1_test_rf, accuracy_test_rf))


#hypermeter tuning: USING VALIDATION DATA
#random search:
set.seed(123)
ctrl_random <- trainControl(method = "repeatedcv",number = 10,repeats = 3,
  classProbs = TRUE,summaryFunction = twoClassSummary,search = "random")

model_rf_random <- train(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = valid_data,method = "rf",
  trControl = ctrl_random,metric = "ROC",tuneLength = 20)
model_rf_random # best mtry=1

# reach to final model
model_rf_random$finalModel

#performance calculation:
# 1. probability calculation:
val_probs_random <- predict(model_rf_random, newdata = valid_data, type = "prob")[, "yes"]

# 2. class prediction
val_preds_random <- factor(ifelse(val_probs_random >= 0.5, "yes", "no"), levels = c("no", "yes"))

# 3. AUC (pROC)
auc_val_random <- pROC::auc(pROC::roc(response = valid_data$SALE, predictor = val_probs_random))

# 4. F1 score (yardstick)
f1_val_random <- yardstick::f_meas_vec(truth = valid_data$SALE, estimate = val_preds_random)

# 5. Accuracy (caret)
acc_val_random <- caret::confusionMatrix(val_preds_random, valid_data$SALE)$overall["Accuracy"]

# 6. print results
cat(sprintf("Validation Set (Random Search tuned model):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_val_random, f1_val_random, acc_val_random))



#train set için: gereksiz yapma
model_rf_random_train <- train(
  SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = train_data,
  method = "rf",
  trControl = ctrl_random,
  metric = "ROC",
  tuneLength = 20
)
model_rf_random_train# 1 buldu mtry


# model_rf_random'ın içindeki gerçek randomForest modeline erişim
model_rf_random_train$finalModel

#performance calculation:# 1. Olasılık tahmini
val_probs_random_train <- predict(model_rf_random_train, newdata = train_data, type = "prob")[, "yes"]

# 2. Sınıf tahmini
val_preds_random_train <- factor(ifelse(val_probs_random_train >= 0.5, "yes", "no"), levels = c("no", "yes"))

# 3. AUC (pROC)
auc_val_random_train <- pROC::auc(pROC::roc(response = train_data$SALE, predictor = val_probs_random_train))

# 4. F1 skoru (yardstick)
f1_val_random_train <- yardstick::f_meas_vec(truth = train_data$SALE, estimate = val_preds_random_train)

# 5. Accuracy (caret)
acc_val_random_train <- caret::confusionMatrix(val_preds_random_train, train_data$SALE)$overall["Accuracy"]

# 6. Sonuçları yazdır
cat(sprintf("Train Set (Random Search tuned model):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_val_random_train, f1_val_random_train, acc_val_random_train))


#test kaldı:gereksiz yapmadım sonra

# 1. Olasılık tahmini
test_probs_random_train <- predict(model_rf_random_train, newdata = test_data, type = "prob")[, "yes"]

# 2. Sınıf tahmini
test_preds_random_train <- factor(ifelse(test_probs_random_train >= 0.5, "yes", "no"), levels = c("no", "yes"))

# 3. AUC (pROC)
auc_test_random_train <- pROC::auc(pROC::roc(response = test_data$SALE, predictor = test_probs_random_train))

# 4. F1 skoru (yardstick)
f1_test_random_train <- yardstick::f_meas_vec(truth = test_data$SALE, estimate = test_preds_random_train)

# 5. Accuracy (caret)
acc_test_random_train <- caret::confusionMatrix(test_preds_random_train, test_data$SALE)$overall["Accuracy"]

# 6. Sonuçları yazdır
cat(sprintf("Test Set (Random Search tuned model):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_test_random_train, f1_test_random_train, acc_test_random_train))



#grid_search:
ctrl_grid <- trainControl(method = "cv",number = 5,
  classProbs = TRUE,summaryFunction = twoClassSummary)

model_rf_grid <- train(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = valid_data,method = "rf",
  trControl = ctrl_grid,metric = "ROC",tuneGrid = expand.grid(mtry = 1:8))

model_rf_grid$finalModel

#performance calculation
# 1. probability calculation (ROC için)
val_probs_grid <- predict(model_rf_grid, newdata = valid_data, type = "prob")[, "yes"]

# 2. class prediction (0.5 threshold)
val_preds_grid <- factor(ifelse(val_probs_grid >= 0.5, "yes", "no"), levels = c("no", "yes"))

# 3. AUC (pROC)
auc_val_grid <- pROC::auc(pROC::roc(response = valid_data$SALE, predictor = val_probs_grid))

# 4. F1 Score (yardstick)
f1_val_grid <- yardstick::f_meas_vec(truth = valid_data$SALE, estimate = val_preds_grid)

# 5. Accuracy (caret, confusionMatrix)
acc_val_grid <- caret::confusionMatrix(val_preds_grid, valid_data$SALE)$overall["Accuracy"]


# 7. Print results
cat(sprintf("Validation Set (Grid Search tuned model):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_val_grid, f1_val_grid, acc_val_grid))


#manual seacrh for ntree
ntree_list <- c(50, 100, 200, 300, 400, 500)
results_rf <- data.frame(ntree = integer(), AUC = numeric(), F1 = numeric())

for (n in ntree_list) {
  set.seed(123)
  rf_model <- randomForest(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
    data = valid_data,ntree = n,mtry = 1)
  
# probablity prediction:
probs <- predict(rf_model, newdata = valid_data, type = "prob")[, "yes"]
preds <- factor(ifelse(probs >= 0.5, "yes", "no"), levels = c("no", "yes"))
  
# performance metrics:
auc_val <- pROC::auc(pROC::roc(valid_data$SALE, probs))
f1_val <- yardstick::f_meas_vec(truth = valid_data$SALE, estimate = preds)
  
# print results
results_rf <- rbind(results_rf,
                      data.frame(ntree = n, AUC = as.numeric(auc_val), F1 = as.numeric(f1_val)))}

results_rf # 300 ntree en iyi AUC-F1 combinationa sahip

#FINAL MODEL
#ntree 300, mtry= 1
set.seed(123)
final_rf_model <- randomForest(
  SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = train_data,
  ntree = 300,
  mtry = 1
)
final_rf_model
# 1. Olasılık tahmini
train_probs_final <- predict(final_rf_model, newdata = train_data, type = "prob")[, "yes"]

# 2. Sınıf tahmini
train_preds_final <- factor(ifelse(train_probs_final >= 0.5, "yes", "no"), levels = c("no", "yes"))

# 3. AUC
auc_train_final <- pROC::auc(pROC::roc(response = train_data$SALE, predictor = train_probs_final))

# 4. F1 Score
f1_train_final <- yardstick::f_meas_vec(truth = train_data$SALE, estimate = train_preds_final)

# 5. Accuracy
acc_train_final <- caret::confusionMatrix(train_preds_final, train_data$SALE)$overall["Accuracy"]

# 6. Sonuçları yazdır
cat(sprintf("Train Set → AUC: %.3f | F1: %.3f | Accuracy: %.3f\n",
            auc_train_final, f1_train_final, acc_train_final))


#validation set:

set.seed(123)
final_rf_model_valid <- randomForest(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = valid_data,ntree = 300,mtry = 1)
final_rf_model_valid


# prediction:
val_probs_final <- predict(final_rf_model, newdata = valid_data, type = "prob")[, "yes"]
val_preds_final <- factor(ifelse(val_probs_final >= 0.5, "yes", "no"), levels = c("no", "yes"))

# metrics:
auc_val_final <- pROC::auc(pROC::roc(valid_data$SALE, val_probs_final))
f1_val_final <- yardstick::f_meas_vec(truth = valid_data$SALE, estimate = val_preds_final)
acc_val_final <- caret::confusionMatrix(val_preds_final, valid_data$SALE)$overall["Accuracy"]
cat(sprintf("Validation Set (Final RF model):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_val_final, f1_val_final, acc_val_final))

#test set:
set.seed(123)
final_rf_model_test <- randomForest(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = test_data,ntree = 300,mtry = 1)
final_rf_model_test
# Tahmin
test_probs_final <- predict(final_rf_model, newdata = test_data, type = "prob")[, "yes"]
test_preds_final <- factor(ifelse(test_probs_final >= 0.5, "yes", "no"), levels = c("no", "yes"))

# Metrikler
auc_test_final <- pROC::auc(pROC::roc(test_data$SALE, test_probs_final))
f1_test_final <- yardstick::f_meas_vec(truth = test_data$SALE, estimate = test_preds_final)
acc_test_final <- caret::confusionMatrix(test_preds_final, test_data$SALE)$overall["Accuracy"]
cat(sprintf("Test Set (Final RF model):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_test_final, f1_test_final, acc_test_final))


# Final Model 2 =ntree=106, mtry=2
# Train Set
set.seed(153)
final_rf_model2 <- randomForest(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = train_data,ntree = 106,mtry = 2)
final_rf_model2
# Train Predictions & Metrics
train_probs_final2 <- predict(final_rf_model2, newdata = train_data, type = "prob")[, "yes"]
train_preds_final2 <- factor(ifelse(train_probs_final2 >= 0.5, "yes", "no"), levels = c("no", "yes"))
#metrics:
auc_train_final2 <- pROC::auc(pROC::roc(train_data$SALE, train_probs_final2))
f1_train_final2 <- yardstick::f_meas_vec(truth = train_data$SALE, estimate = train_preds_final2)
acc_train_final2 <- caret::confusionMatrix(train_preds_final2, train_data$SALE)$overall["Accuracy"]

cat(sprintf("Train Set (Final Model 2):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_train_final2, f1_train_final2, acc_train_final2))


# Final Model 2:validation
val_probs_final2 <- predict(final_rf_model2, newdata = valid_data, type = "prob")[, "yes"]
val_preds_final2 <- factor(ifelse(val_probs_final2 >= 0.5, "yes", "no"), levels = c("no", "yes"))
#metrics:
auc_val_final2 <- pROC::auc(pROC::roc(valid_data$SALE, val_probs_final2))
f1_val_final2 <- yardstick::f_meas_vec(truth = valid_data$SALE, estimate = val_preds_final2)
acc_val_final2 <- caret::confusionMatrix(val_preds_final2, valid_data$SALE)$overall["Accuracy"]
cat(sprintf("Validation Set (Final Model 2):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_val_final2, f1_val_final2, acc_val_final2))


# Final Model 2 - Test Set
test_probs_final2 <- predict(final_rf_model2, newdata = test_data, type = "prob")[, "yes"]
test_preds_final2 <- factor(ifelse(test_probs_final2 >= 0.5, "yes", "no"), levels = c("no", "yes"))
#metrics:
auc_test_final2 <- pROC::auc(pROC::roc(test_data$SALE, test_probs_final2))
f1_test_final2 <- yardstick::f_meas_vec(truth = test_data$SALE, estimate = test_preds_final2)
acc_test_final2 <- caret::confusionMatrix(test_preds_final2, test_data$SALE)$overall["Accuracy"]
cat(sprintf("Test Set (Final Model 2):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_test_final2, f1_test_final2, acc_test_final2))


#final model 3= mtry=1, ntree=106
# Final Model 3 - Train Set
set.seed(123)
final_rf_model3 <- randomForest(SALE ~ lag1_AP + lag1_BUDGET + lag1_TSV + lag1_TPC + TIME,
  data = train_data,ntree = 106,mtry = 1)
final_rf_model3
# Train Predictions & Metrics
train_probs_final3 <- predict(final_rf_model3, newdata = train_data, type = "prob")[, "yes"]
train_preds_final3 <- factor(ifelse(train_probs_final3 >= 0.5, "yes", "no"), levels = c("no", "yes"))

auc_train_final3 <- pROC::auc(pROC::roc(train_data$SALE, train_probs_final3))
f1_train_final3 <- yardstick::f_meas_vec(truth = train_data$SALE, estimate = train_preds_final3)
acc_train_final3 <- caret::confusionMatrix(train_preds_final3, train_data$SALE)$overall["Accuracy"]

cat(sprintf("Train Set (Final Model 3):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_train_final3, f1_train_final3, acc_train_final3))


# Validation Set
val_probs_final3 <- predict(final_rf_model3, newdata = valid_data, type = "prob")[, "yes"]
val_preds_final3 <- factor(ifelse(val_probs_final3 >= 0.5, "yes", "no"), levels = c("no", "yes"))

auc_val_final3 <- pROC::auc(pROC::roc(valid_data$SALE, val_probs_final3))
f1_val_final3 <- yardstick::f_meas_vec(truth = valid_data$SALE, estimate = val_preds_final3)
acc_val_final3 <- caret::confusionMatrix(val_preds_final3, valid_data$SALE)$overall["Accuracy"]

cat(sprintf("Validation Set (Final Model 3):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_val_final3, f1_val_final3, acc_val_final3))


# Test Set
test_probs_final3 <- predict(final_rf_model3, newdata = test_data, type = "prob")[, "yes"]
test_preds_final3 <- factor(ifelse(test_probs_final3 >= 0.5, "yes", "no"), levels = c("no", "yes"))

auc_test_final3 <- pROC::auc(pROC::roc(test_data$SALE, test_probs_final3))
f1_test_final3 <- yardstick::f_meas_vec(truth = test_data$SALE, estimate = test_preds_final3)
acc_test_final3 <- caret::confusionMatrix(test_preds_final3, test_data$SALE)$overall["Accuracy"]

cat(sprintf("Test Set (Final Model 3):\nAUC = %.3f | F1 Score = %.3f | Accuracy = %.3f\n",
            auc_test_final3, f1_test_final3, acc_test_final3))




#SHAP????? white box tekrar dön incele!!!!
install.packages("iml")

library(iml)
X <- train_data %>% select(lag1_AP, lag1_BUDGET, lag1_TSV, lag1_TPC, TIME)
y <- train_data$SALE

# Tahmin fonksiyonu
predictor <- Predictor$new(
  model = final_rf_model3,
  data = X,
  y = y,
  type = "prob",
  class = "yes"
)

# SHAP değerlerini hesapla
shap <- Shapley$new(predictor, x.interest = X[1, ])
plot(shap)
