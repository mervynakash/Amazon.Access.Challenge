set.seed(123)
setwd("E:/Kaggle/Amazon Employee Access Challenge/")

library(dplyr)
library(ggplot2)
library(ROCR)
library(pROC)
library(rpart)
library(caret)

train <- read.csv("train.csv", stringsAsFactors = F)
test <- read.csv("test.csv", stringsAsFactors = F)

train$ACTION <- as.factor(train$ACTION)

inTrain <- createDataPartition(train$ACTION, p = .75, list = F)

new.train <- train[inTrain,]
new.test <- train[-inTrain,]

#str(train)

colSums(is.na(train))
colSums(is.na(test))


#######################################################################################
#==================================== Using Logistic Regression ======================#

model1 <- glm(ACTION~., data = train, family = binomial(link = "logit"), control = list(maxit = 50))
anova(model1, test = "Chisq")
summary(model1)

model2 <- glm(ACTION~ROLE_ROLLUP_1+ROLE_CODE+ROLE_TITLE, data = train, family = binomial(link = "logit"), control = list(maxit = 50))
anova(model2, test = "Chisq")
anova(model1,model2, test = "Chisq")

model3 <- glm(ACTION~ROLE_ROLLUP_1+ROLE_CODE, data = train, family = binomial(link = "logit"), control = list(maxit = 50))
anova(model1,model3, test = "Chisq")

modelrandom <- glm(ACTION~ROLE_ROLLUP_1+ROLE_CODE+ROLE_TITLE, data = new.train, family = binomial(link = "logit"), control = list(maxit = 50))

predict.random <- predict(modelrandom,newdata = new.test, type = "response")
predict.random <- ifelse(predict.random > 0.9, 1,0)

pr <- prediction(predict.random, new.test$ACTION)
perf <- performance(pr, measure = "tpr", x.measure = "fpr")

plot(perf)

roc_obj <- roc(new.test$ACTION,predict.random)
auc(roc_obj)

df <- data.frame(A = new.test$ACTION, B = predict.random)
View(df)
count = 0
for(i in seq(nrow(df))){
  if(df[i,1] == df[i,2]){
    next()
  } else {
    count = count + 1
  }
}

accuracy_log <- mean(predict.random == new.test$ACTION)*100

pred.test <- predict(modelrandom, test, type = "response")
pred.test <- ifelse(pred.test > 0.9, 1, 0)

#######################################################################################
#==================================== Using rpart ====================================#

#new.train.rp <- new.train
#new.train.rp$ACTION <- as.factor(new.train.rp$ACTION)
#new.test.rp <- new.test
#new.test.rp$ACTION <- as.factor(new.test.rp$ACTION)

mod_rp <- rpart(ACTION~., data = new.train, method = "class", control = rpart.control(cp = 0))
plot(mod_rp);text(mod_rp)
printcp(mod_rp)

pred.base <- as.data.frame(predict(mod_rp, new.test, method = "class"))
pred.base <- ifelse(pred.base$`0` > pred.base$`1`, 0, 1)

accuracy_amazon <- mean(pred.base == new.test$ACTION)*100

cp_table <- data.frame(mod_rp$cptable)
cp_val <- cp_table$CP[which(cp_table$xerror == min(cp_table$xerror))][1]

mod.prun <- prune(mod_rp, cp = cp_val)
plot(mod.prun);text(mod.prun, xpd = TRUE)

pred.base.prune <- as.data.frame(predict(mod.prun, new.test, method = "class"))
pred.base.prune <- ifelse(pred.base.prune$`0` > pred.base.prune$`1`, 0, 1)

accuracy_amazon_prune <- mean(pred.base.prune == new.test$ACTION)*100


bestmodel <- data.frame(log = accuracy_log, ctree = accuracy_amazon, ctree_pruned = accuracy_amazon_prune)


true_pred <- as.data.frame(predict(mod.prun, test, method = "class"))
true_pred <- ifelse(true_pred$`0` > true_pred$`1`, 0, 1)

#=============== Rpart file ============#
df <- data.frame(Id = 1:length(true_pred),Action = true_pred)
View(df)
write.csv(x = df, "submission_file.csv", row.names = F)

#=============== Logistic file ============#
df2 <- data.frame(Id = 1:length(pred.test), Action = pred.test)
write.csv(x = df2, file = "submission_logit.csv", row.names = F)






#=================================== RANDOM FOREST ===================================# 

control = trainControl(method = "repeatedcv", repeats = 3, number = 10)

mtry1 = floor(sqrt(ncol(new.train)-1))

model_rf <- train(ACTION~., data=new.train, trControl = control, tuneGrid = expand.grid(mtry = c(2,3,4,5)), method = "rf")
print(model_rf)
pred_rf <- predict(model_rf, test)

df = data.frame(id = test$id, Action = pred_rf)
View(df)

write.csv(df, file = "Submission_rf.csv", row.names = F)
