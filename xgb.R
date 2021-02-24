require(caret)
require(corrplot)
require(Rtsne)
require(xgboost)
require(stats)
require(knitr)
require(ggplot2)
require(MASS)

### set up input data ##
load("input.RData")
data=input[["score.org"]]
#data=data[,grep("HCV",colnames(data))][,c(10,18,30)] ## HCV only
info=input[["info"]]
data.hr = data[info$Group == "HR",]
data.hcc = data[info$Group == "HCC",]
data.pc = data[info$Group == "PC",]

# split PC to half for training  150 HCC 337 HR 412 PC#
#set.seed(123456)
#train.index=createDataPartition(1:nrow(data.pc), p=0.5, list=FALSE,times=1)
#train.pc=data.pc[train.index,]
#test.pc=data.pc[-train.index,]

# set up train and test data #
train=rbind(data.hcc,data.pc)
train=train[,colSums(train)> 0] # 10 
test=rbind(data.hr)
test=test[,colnames(train)]


# lable train and test #
train.label =info[match(rownames(train),info$Species),]$Group
train.label =factor(train.label,levels=c("PC","HCC"))
levels(train.label) = 0:1
train.label=as.numeric(train.label)-1 

test.label =info[match(rownames(test),info$Species),]$Group
test.label =factor(test.label,levels=c("PC","HR"))
levels(test.label) =  0:1
test.label=as.numeric(test.label)-1 


 
## create 100 random variables with zero inflated and possion distrubtions ##
#n=10
#variable=list()
#set.seed(123456)
#for (i in seq_len(n)){
#prob=seq(0.1,0.9,length.out=n)
#lambda=5
#isZero = rbinom(n = nrow(train), size = 1, prob = prob[i])
#variable[[paste0("var_",i)]] = ifelse(isZero==1, 0, rnegbin(sum(isZero==0), mu=lambda,theta=5))
#}
#variable=do.call(cbind.data.frame, variable)
#train = as.matrix(cbind(train,variable))
#dim(train)



# convert data to matrix


dtrain <- xgb.DMatrix(data = as(train, "dgCMatrix"), label = train.label)
dtest <- xgb.DMatrix(data = as(test, "dgCMatrix"), label = test.label)




## grid search ###
#searchGridSubCol <- expand.grid(subsample = 1, 
#                                colsample_bytree = c(0.5, 1),
#                                max_depth = c(3, 4, 5),
#                                min_child = c(1,2), 
#                                eta = c(0.1,0.2,0.3)
#)

#ntrees <- 200
#set.seed(123456)
#system.time(
#ErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
#  currentSubsampleRate <- parameterList[["subsample"]]
#  currentColsampleRate <- parameterList[["colsample_bytree"]]
#  currentDepth <- parameterList[["max_depth"]]
#  currentEta <- parameterList[["eta"]]
#  currentMinChild <- parameterList[["min_child"]]
#  xgboostModelCV <- xgb.cv(data =  dtrain, nrounds = ntrees,  nfold=5,
#                       metrics = "error", verbose = TRUE, "eval_metric" = "error","eval_metric" = "auc",
#                     "objective" = "binary:logistic", "max.depth" = currentDepth, "eta" = currentEta,                               
#                     "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate
#                      , print_every_n = 10, "min_child_weight" = currentMinChild, booster = "gbtree",
#                     early_stopping_rounds=20,maximize=T)
  
#  xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
#  error <- tail(xvalidationScores$test_error_mean, 1)
#  auc  <- tail(xvalidationScores$test_auc_mean, 1)	
#  terror <- tail(xvalidationScores$train_error_mean,1)
#  tauc <- tail(xvalidationScores$train_auc_mean,1)
#  output <- return(c(error,auc, terror,tauc, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta, currentMinChild))
#  
#}))

#output <- as.data.frame(t(ErrorsHyperparameters))
#head(output)
#varnames <- c("TestError","TestAuc","TrainError","TrainAuc", "SubSampRate", "ColSampRate", "Depth", "eta", "currentMinChild")
#names(output) <- varnames
#head(output)
#write.csv(output, "xgb_gridsearch.csv")


# xgboost parameters
param <- list("booster" = "gbtree", ## gblinear or gbtree
		  "objective" = "binary:logistic",    # classification
		  "base_score" =0.5,	
		  "eval_metric" = "error", 
              "eval_metric" = "auc",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used
		  #"tree_method" = "exact", # for small dataset 
              "max_depth" = 3,    # maximum depth of tree 
              "eta" = 0.1,    # step size shrinkage 0.1
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 0.5,  # subsample ratio of columns when constructing each tree 1
              "min_child_weight" = 1  # minimum sum of instance weight needed in a child 1
              )

set.seed(123456)
cv <- createFolds(train.label, k = 10)
# k-fold cross validation, with timing
bst.cv <- xgb.cv(param=param, data=dtrain,folds=cv, nrounds=200, early_stopping_rounds=20,maximize=T)
best=bst.cv$best_iteration

plot(bst.cv$evaluation_log$train_auc_mean,ylim=c(0.5,1),ylab="AUC",xlab="N rounds")
points(bst.cv$evaluation_log$test_auc_mean,col="red")
abline(v=best,col="gray60")
legend("topleft",c("train.cv","test.cv"),col=c(1,2),pch=1)


# real model fit training, with full data
bst <- xgboost(param=param, data=dtrain,nrounds=best, verbose=0)
xgb.save(bst, "xgboost.model")

## predict on test ##
pred=predict(bst,dtest)
names(pred)=rownames(test)
pred.acc=as.data.frame(pred)
pred.acc$acc=info$Accession[match(rownames(pred.acc),info$Species)]
write.csv(pred.acc,"pred.acc.hr.csv")
library(beeswarm)
# acc 21239 die of HCC #
pat.hcc=as.character(info$Species[which(info$Accession %in% c("24761","21523","21239"))])
beeswarm(pred~test.label,xlab="",ylab="",pwcol=1+as.numeric(names(pred) %in% pat.hcc),labels="")
boxplot(pred~test.label,add=T,xlab="",ylab="Predict Score",names=c("PC","HR"))

## predict on the train ##
pred.train=predict(bst,dtrain)
names(pred.train)=rownames(train)
pred.train.acc=as.data.frame(pred.train)
pred.train.acc$acc=info$Accession[match(rownames(pred.train.acc),info$Species)]
pred.train.acc$group=info$Group[match(rownames(pred.train.acc),info$Species)]
write.csv(pred.train.acc,"pred.acc.hcc.pc.csv")


# get the trained model
model = xgb.dump(bst, with_stats=TRUE)
# get the feature real names
names = colnames(train)
# compute feature importance matrix
importance_matrix = xgb.importance(names, model=bst)
print(importance_matrix)
write.csv(importance_matrix,"importance_matrix1.csv")
# plot
gp = xgb.ggplot.importance(importance_matrix)
#gp = xgb.plot.importance(importance_matrix)

print(gp)  

anova=read.csv("anova.result.csv",header=T,row.names=1)
gp$data$risk=anova[gp$data$Feature,]$risk
ggplot2::ggplot(importance_matrix, ggplot2::aes(x = factor(Feature, 
        levels = rev(Feature)), y = Importance, width = 0.05), 
        environment = environment()) + ggplot2::geom_bar(ggplot2::aes(fill = gp$data$risk), 
        stat = "identity", position = "identity") + ggplot2::coord_flip() + 
        ggplot2::xlab("Features") + ggplot2::ggtitle("Feature importance") + 
        ggplot2::theme(plot.title = ggplot2::element_text(lineheight = 0.9, 
            face = "bold"), panel.grid.major.y = ggplot2::element_blank())




#### THE XGBoost Explainer
library(xgboostExplainer)
explainer = buildExplainer(bst,dtrain, type="binary", base_score = 0.5, trees_idx = NULL)
pred.breakdown = explainPredictions(bst, explainer, dtest)
cat('Breakdown Complete','\n')
weights = rowSums(pred.breakdown)
pred.xgb = 1/(1+exp(-weights))
cat(max(pred-pred.xgb),'\n')
idx_to_get = as.integer(104)
showWaterfall(bst, explainer, dtest, test ,idx_to_get, type = "binary")
idx_to_get = as.integer(123)
showWaterfall(bst, explainer, dtest, test ,idx_to_get, type = "binary")

####### IMPACT AGAINST VARIABLE VALUE
plot(test[,"Hepatitis C virus genotype 1a (isolate 1) (HCV)"], as.matrix(pred.breakdown[, "Hepatitis C virus genotype 1a (isolate 1) (HCV)"])[,1], cex=0.4, pch=16, xlab = "Satisfaction Level", ylab = "Satisfaction Level impact on log-odds")






##### permutation test ##########
perm.n=2000
pct=table(train.label)
label=list()
for (i in seq_len(perm.n)){
set.seed(123456+i)
label[[i]]=sample(c("0","1"),size = pct["0"]+pct["1"],replace=T,prob=c(pct["0"],pct["1"])/(pct["0"]+pct["1"]))
}

auc.perm=NA
feature=list()
names = colnames(train)
for (i in seq_len(perm.n)){
set.seed(123456)
dtrain.perm <- xgb.DMatrix(data = as(train, "dgCMatrix"), label = as.numeric(label[[i]]))

bst.cv <- xgb.cv(param=param, data=dtrain.perm,nfold=5, nrounds=200, early_stopping_rounds=20,maximize=T,verbose = 0)
auc.perm[i]=bst.cv$evaluation_log[bst.cv$best_iteration]$test_auc_mean
bst <- xgboost(param=param, data=dtrain,nrounds=bst.cv$best_iteration, verbose=0)
importance_matrix = xgb.importance(names, model=bst)
feature[[i]] = match(names,importance_matrix$Feature)
}
table(auc.perm > 0.73)
feature.1=do.call(cbind.data.frame, feature)
rownames(feature.1)=names
colnames(feature.1)=paste0("perm",1:2000)
feature.count=sort(rowSums(feature.1 > 1,na.rm=T),decreasing=T)
write.csv(feature.count,"feature.count.csv")



### add regression on features ####
library(pROC)
mod.base=glm(train.label~train[,importance_matrix$Feature[1]],family="binomial")
predpr.base <- predict(mod.base,type=c("response"))
roccurve.base <- roc(train.label ~ predpr.base)
plot(roccurve.base)
auc(roccurve.base)

roccurve=list()
roccurve[[1]]=roccurve.base
for (i in 2:nrow(importance_matrix)){
mod=glm(train.label~train[,importance_matrix$Feature[1:i]],family="binomial")
predpr <- predict(mod,type=c("response"))
roccurve[[i]] <- roc(train.label ~ predpr)
}

library(RcolorRamp)
plot(roccurve.base,col="blue")
for (i in 2:nrow(importance_matrix)){
plot(roccurve[[i]],add=T,col= colorRampPalette(c("grey", "black"))(45)[i])
}
legend("bottomright",paste("AUC:", round(auc(roccurve.base),digits=2),round(auc(roccurve[[i]]),digits=2),sep="-"),col=1,lty=1,lwd=2)




## try glmnet method ##
library(glmnet)
cv.glmnet.fit=cv.glmnet(data.matrix(train), train.label,family = "binomial",type.measure = "auc",nfolds = 5,alpha = 1)
log(cv.glmnet.fit$lambda.min)
max(cv.glmnet.fit$cvm)
plot(cv.glmnet.fit)
tmp_coeffs <- coef(cv.glmnet.fit, s = "lambda.min")
glmnet.features=data.frame(name = tmp_coeffs@Dimnames[[1]][tmp_coeffs@i + 1], coefficient = tmp_coeffs@x)
glmnet.features=glmnet.features[-1,]
write.csv(glmnet.features[order(glmnet.features$coefficient,decreasing=T),],"glmnet.features.csv")



lambdas = NULL
select.features.all =NULL
n=100
for (i in 1:n)
{
    fit <- cv.glmnet(data.matrix(train), train.label,family = "binomial",type.measure = "auc",nfolds = 5,alpha = 1)
    errors = data.frame(fit$lambda,fit$cvm)
    lambdas <- rbind(lambdas,errors)
    tmp <- coef(fit, s = "lambda.min")	
    select.features=as.data.frame(as.matrix(tmp))[,1]	
    select.features.all=cbind(select.features,select.features.all)
}
# take mean cvm for each lambda
lambdas <- aggregate(lambdas[, 2], list(lambdas$fit.lambda), mean)
rownames(select.features.all)=tmp@Dimnames[[1]]
count=as.numeric(apply(select.features.all,1,function(x) table(x==0)["FALSE"]))
mean(as.numeric(apply(select.features.all,2,function(x) table(x==0)["FALSE"])))
select.features.sum=cbind(count,rowMeans(select.features.all))
write.csv(select.features.sum,"select.features.sum.csv")

# select the best one
bestindex = which(lambdas[2]==min(lambdas[2]))
bestlambda = lambdas[bestindex,1]

# and now run glmnet once more with it
fit <- glmnet(data.matrix(train), train.label,family = "binomial",alpha = 1,lambda=bestlambda)



















