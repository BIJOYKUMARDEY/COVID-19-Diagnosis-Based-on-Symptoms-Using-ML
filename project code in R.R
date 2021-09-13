data<-read.csv(file.choose(),header=T)
View(data)
str(data)
summary(data)

# 0 means covid negative, 1 means covid positive disease which have symptoms as covid
data$corona_result[data$corona_result=="negative"]<-0
data$corona_result[data$corona_result=="positive"]<-1

# 0 means age below 60 , 1 means age 60 and above 60 
data$age_60_and_above[data$age_60_and_above=="No"]<-0
data$age_60_and_above[data$age_60_and_above=="Yes"]<-1

# 0 means male , 1 means female
data$gender[data$gender=="male"]<-0
data$gender[data$gender=="female"]<-1

# 0 means Abroad, 1 means contact with other person, 2 means other case 
data$test_indication[data$test_indication=="Abroad"]<-0
data$test_indication[data$test_indication=="Contact with confirmed"]<-1
data$test_indication[data$test_indication=="Other"]<-2
str(data)

#convert the variables to factor
factor<-c("cough","fever","sore_throat","shortness_of_breath","head_ache","corona_result","age_60_and_above","gender","test_indication")
data[factor]<-lapply(data[factor],factor)
str(data)
summary(data)
table(data$corona_result)# Here negative case 89.945% then if our model accuracy is greater than 89.945% then our model will be good

# Handling missing values
library(mice)
library(VIM)
md.pattern(data)
impute<-mice(data[,2:10],m=3,seed=0)
print(impute)
barplot(table(data$gender))
head(impute$imp$gender)
tail(impute$imp$gender)
barplot(table(data$age_60_and_above))
head(impute$imp$age_60_and_above)
tail(impute$imp$age_60_and_above)
newdata<-complete(impute,2)
summary(newdata)
View(newdata)

# split the data into testing and training set
set.seed(0)
split<-sample(2,nrow(newdata),replace=T,prob=c(0.7,0.3))
train<-newdata[split==1,]
test<-newdata[split==2,]
summary(train)
summary(test)

####Decision tree
library(party)
tree<-ctree(corona_result~.,data=train,controls = ctree_control(mincriterion = 0.99,minsplit=10000))
tree
plot(tree)

prediction<-predict(tree,train)
prediction
tab11<-table(predict=prediction,actual=train$corona_result)
tab11
sum(diag(tab11))/sum(tab11)# accuracy rate 91.5%

prediction<-predict(tree,test)
prediction
tab11<-table(prediction,test$corona_result)
tab11
sum(diag(tab11))/sum(tab11)#accuracy rate 91.473%


# create dummy variables
library(xgboost)
library(magrittr)
library(Matrix)

trainy<-train$corona_result=="1"
trainx<-model.matrix(corona_result~.-1,data=train)
trainx<-trainx[,-1]

testy<-test$corona_result=="1"
testx<-model.matrix(corona_result~.-1,data=test)
testx<-testx[,-1]

#create Dmatrix
xmatrix<-xgb.DMatrix(data=trainx,label=trainy)
xmatrix_t<-xgb.DMatrix(data=testx,label=testy)

# eXtreme Gradient Boosting Model
model<-xgb.train(data=xmatrix,nrounds = 100,objective="multi:softmax",eta=0.2,num_class=2,max_depth=6,watchlist=list(train=xmatrix,test=xmatrix_t))
model
e<-data.frame(model$evaluation_log)
plot(e$iter,e$train_mlogloss,col="blue")
lines(e$iter,e$test_mlogloss,col="red")
imp<-xgb.importance(colnames(xmatrix),model)
imp
xgb.plot.importance(imp)

#prediction and accuracy rate for train data
trainpred<-predict(model,xmatrix)
tab<-table(trainy,trainpred)
sum(diag(tab))/sum(tab) # accuracy rate 92.12421 %
tab

#prediction and accuracy rate for test data
testpred<-predict(model,xmatrix_t)
tab1<-table(testy,testpred)
tab1
sum(diag(tab1))/sum(tab1) # accuracy rate 92.12249 %

#save model to RDS file
saveRDS(model,file="ntcc_model.RDS")

