mtcars
?mtcars #gives deatils about the command
class(mtcars) #data structure
x=1:5
class(x)
str(mtcars)
?str  # display the structure of the data
y=c(1,3,5)
class(y)
class(as.integer(y)) #convert one structure to another structure
summary(mtcars)
z=c(1L,3L,5L)
class(z)





#vector----
x=1:100
x
?c
(x3=c(2,4,36,3))
x3
print(x3)

(x4=rnorm(n=100, mean=60, sd=10))
plot(x4)
plot(density(x4))
hist(x4)
hist(x4, breaks = 10,  col=1:10)
range(x4)
boxplot(x4)
plot(sort(x4), type = "b")  # both lines and dots
sort(x4)
sort(x4, decreasing = TRUE)
plot(x4, type = "l")
x4[x4>65]
mean(x4[x4>65]) # mean of the data morethan 65
x4[-c(1:10)]
x4[x4>65]
length(x4[x4>65])
sum(x4>65)
rev(x4)
x4[x4>65 & x4<80]





#matrices----
(data = c(10,30,40,44,22,55))
(m1 = matrix(data = data, nrow = 2))
(m1 = matrix(data = data, nrow = 2, byrow = T)) # fill the entries by row
rownames(m1) = c('R1', 'R2')
m1
colnames(m1) = c('C1', 'C2','C3')
m1
rownames(m1) = c('R1','')
m1[,1]
m1[1,]
colSums(m1)
rowSums(m1)
rowMeans(m1)
apply(m1, 2, FUN=min)
m1
apply(m1[, 'C2', drop=F], 2,FUN = max)





#dataframe----
(rollno = paste('IIMLN', 1:13, SEP='-'))
(name = paste('STUDENT', 1:13, sep = '&'))
(AGE = round(runif(13, min =24, max=32),2))
(marks = trunc(rnorm(13, mean=60, sd=10)))
set.seed(12)
(gender=sample(c('M','F'), size = 13, replace=T, prob = c(0.7,0.3)))
table(gender)
(x=c(-14.35, 14.35, -14.55, 15.35))  
floor(x); ceiling(x);trunc(x)

set.seed(66)
(grade=sample(c('EX', 'GOOD', 'SAT'),size = 13, replace = T, prob = c(0.6,0.3,0.1)))
table(grade)
prop.table(table(grade))
sapply(list(rollno, name, AGE, marks, gender, grade), length)
(students = data.frame(rollno, name, AGE, marks, gender, grade))           



write.csv(students, 'data/students.csv', row.names=F)
df1= read.csv('data/students.csv')
df1
class(students)
summary(students)
library(dplyr)
students %>% group_by(gender) %>% tally()
students %>% group_by(gender) %>% summarise(mean(AGE), n(), min(marks), max(marks))
students %>% group_by(gender, grade) %>% summarise(mean(AGE))  





# day 2
library(dplyr)
(mtcars)
table(mtcars$cyl)
summary(mtcars$cyl)
mtcars %>% group_by(cyl) %>%tally()
mtcars %>% group_by(cyl) %>% summarise(count = n())
xtabs(~cyl,data=mtcars)
ftable(mtcars$cyl)


#gear & cyl
table(mtcars$cyl, mtcars$gear)
table(mtcars$cyl, mtcars$gear, mtcars$am, dnn =c('cyl', 'gear', 'Automanual'))

df=mtcars
head(df)
tail(df)
dfsam= ifelse(df$am==0, 'Auto', 'Manual')
df
mtcars %>% mutate(TxType = ifelse(am==0, 'Auto', 'Manual'))
mtcars %>% mutate(TxType = ifelse(am==0, 'Auto', 'Manual')) %>%group_by(TxType) %>% summarise(count = n())
mtcars
df = mtcars
df$mpg
df[, 'mpg']
df
head(df)
df = df %>% mutate(TxType = ifelse(am==0, 'Auto', 'Manual'))
head(df)
# increase mileage by 10%
?mutate
?ifelse
df$mgg *1.1
head(df)
#add mpg + wt into new column MPGWT
df$mpg + df$wt
df$MPGWT = df$mpg * 1.1 + df$wt
head(df)
df
?df
?head(df)
#top 2 cars from mpg from each gear type : use group_by & top_n
?top_n
df %>% group_by(gear) %>%top_n(n=2, wt=mpg) %>%select(gear, mpg)
df %>% group_by(gear) %>%arrange(-mpg) %>% select(gear, mpg)
df %>% group_by(gear) %>% top_n(n=2, wt=-mpg) %>%select(gear,mpg)


# list out details of any 2 cars picked randomly; then do 25% of the cars
df %>% sample_n(2)
df %>% sample_frac(0.25) %>%arrange(mpg)
#ascending gear, descending mpg
df %>% sample_frac(0.25) %>% arrange(gear,desc(mpg))


sort(df$mpg)
df[order(df$mpg),]
# find mean, weight, mileage and horse power for each gear type
df %>% group_by(gear) %>%summarise(avg=mean(mpg))
df %>% group_by(gear) %>%summarise_at(c('mpg', 'wt','hp','disp'), c(mean))


#graphs----
hist(df$mpg)
barplot(table(df$gear), col=1:3)
pie(table(df$gear))
plot(df$wt, df$mpg)
pie(table(df$gear), labels=)


library(ggplot2)
library(reshape2)
(rollno= paste('IIM', 1:10, sep='_'))
(name= paste('sName', 1:10, sep='_'))
(gender= sample(c('M', 'F'), size=10, replace=T))
(program = sample(c('BBA', 'MBA'), size = 10, replace=T))
(marketing=trunc(rnorm(10, mean=60, sd=10)))
(finance=trunc(rnorm(10, mean=55, sd=12)))plot()
students<-data.frame(rollno, name, gender, program, marketing, finance, stringsAsFactors = F)
stringsAsFactors = F
head(students)




(meltsum1 <- melt(students, id.vars=c('rollno', 'name', 'gender', 'program'), variable.name='subject', value.name = 'marks'))


students
head(students, n=2)
students %>% group_by(gender) %>% summarise(count = n())
ggplot(students %>% group_by(gender) %>% summarise(count = n()), aes(x=gender, y=count)           ) + geom_bar(stat='identity')

ggplot(students %>% group_by(program, gender) %>% summarise(COUNT = n()), aes(x=gender, y=COUNT, fill=program)) + geom_bar(stat='identity', position = position_dodge2(.7)) + geom_text(aes(label=COUNT), position = position_dodge2(.7)) + labs(title='Gender wise - program count')


#subject
ggplot(meltsum1 %>% group_by(program, gender, subject) %>% summarise(meanmarks = round(mean(marks))), aes(x=gender, y=meanmarks, fill=program)) + geom_bar(stat='identity', position = position_dodge2(.7)) + geom_text(aes(label=meanmarks), position = position_dodge2(.7)) + labs(title= 'subject-program-gender-meanmarks') + facet_grid(~subject)


ggplot(mtcars, aes(x=wt, y=mpg, size=hp, color=factor(gear), shape=factor(am)))+geom_point()




#day3 : 23 Feb
(x=c(1,2,4,5))
(x2=c(1,2,NA,4,NA,,5))    #ERROR
(x2=c(1,2,NA,4,NA,5))
sum(x2)
sum(x2, na.rm = T)
length(x2)
is.na(x2)
sum(c(T,F,T,F,T))
sum(is.na(x2))
sum(is.na(x2))/length(x2)   # perc of missing values
x2
mean(x2, na.rm = T)
x2[is.na(x2)]



library(VIM)
data(sleep)
sleep
?sleep
head(sleep)
tail(sleep)
str(sleep)
?str
dim(sleep)
length(sleep)
summary(sleep)
(x=200:300)
quantile(x)
quantile(x, seq(0,1,0.25))
quantile(x, seq(0,1,0.01))
library(dplyr)

head(sleep)
is.na(sleep)
sum(is.na(sleep))
colSums(is.na(sleep))
rowSums(is.na(sleep))
complete.cases(sleep)
?complete.cases
sum(complete.cases(sleep))
sleep[complete.cases(sleep),]
sleep[!complete.cases(sleep),]
xy=colSums(is.na(sleep))
xy
xy[xy>0]
c1<-names(xy[xy>0])
sleep[,c1]
sleep%>%select(c1) %>% length()
sleep%>%select(-c1) %>% length()

'%notin%'<-Negate('%in%')
c2<-names(sleep)%notin%c1
sleep[,c2]


# data partitioning
(x=1:100)
set.seed(134)
s1<-sample(x, size=70)
length(s1)
sum(s1)

s2<-sample(x,size=0.7*length(x))
length(s2)
x

mtcars
mtcars%>%sample_n(24)
mtcars%>%sample_frac(0.7)
dim(mtcars);nrow(mtcars)
(index=sample(1:nrow(mtcars), size=0.7*nrow(mtcars)))
mtcars[index,]
mtcars[-index,]; dim(mtcars[index,])
mtcars[-index,]


library(caTools)
sample=sample.split(Y=mtcars$am, SplitRatio = 0.7)
sample
table(sample)
prop.table(table(sample))
y1=mtcars[sample==T,] #true set
y2=mtcars[sample==F,]
prop.table(table(y1$am))
table(y1$am); prop.table(table(y1$am))
table(y2$am); prop.table(table(y2$am))


library(caret)
(intrain<-createDataPartition(y=mtcars$am, p=0.7, list=F))
train<-mtcars[intrain,]
test<-mtcars[-intrain,]
prop.table(table(train$am)); prop.table(table(test$am))
?prop.table



#linear regression
women
head(women)
model=lm(weight~height, data=women)
summary(model)
#y=mx+c
#weight=3.45*height+-87.51
plot(x=women$height, y=women$weight)
abline(model)
residuals(model)
women$weight
predwt<-predict(model, newdata=women, type='response')
head(women)
3.45*58-87
cbind(women, predwt,res = women$weight- predwt, res2=residuals(model))
sqrt(sum(residuals(model)^2))   #SSE


(range(women$height))
ndata=data.frame(height=c(66.5,75.8))
predict(model, newdata=ndata, type='response') 
confint(model)     



#case2
link="https://docs.google.com/spreadsheets/d/1h7HU0X_Q4T5h5D1Q36qoK40Tplz94x_HZYHOJJC_edU/edit#gid=2023826519"
library(gsheet)
df=as.data.frame(gsheet2tbl(link))
head(df)
model2=lm(Y~X, data=df)
plot(df$X, df$Y)
abline(model2)
resid(model2)
range(df$X)
ndata=data.frame(X=c(3,4))
predict(model2, newdata=ndata, type='response')

#assumptions
plot(model)
link1="https://docs.google.com/spreadsheets/d/1h7HU0X_Q4T5h5D1Q36qoK40Tplz94x_HZYHOJJC_edU/edit#gid=1595306231"
df2=as.data.frame(gsheet2tbl(link1))
head(df2)

model3<- lm(sqty~price+promotion, data=df2)
plot(df2$price, df2$sqty)
plot(df2$promotion, df2$sqty)
abline(model3)
resid(model3)
summary(model3)
range(df$promotion)
plot(model3)
plot(df2$price, df2$sqty)
abline(model3, col=2)
plot(df2$promotion, df2$sqty)
abline(model3, col2)
range(df$price)
range(df$promotion)
(ndata3=data.frame(price=c(60,75), promotion=c(300,500)))
predict(model3, newdata = ndata3, type='response')
plot(model3)




# Decision Tree - Classification

#we want predict for combination of input variables, is a person likely to survive or not



#import data from online site

path = 'https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/titanic_train.csv'

titanic <- read.csv(path)

head(titanic)

names(titanic)

data = titanic[,c(2,3,5,6,7)]  #select few columns only

head(data)

dim(data)

#load libraries

library(rpart)

library(rpart.plot)

str(data)

#Decision Tree

names(data)

prop.table(table(data$Survived))

str(data)

data$Pclass = factor(data$Pclass)

fit <- rpart(Survived ~ ., data = data, method = 'class')

fit

rpart.plot(fit, extra = 104, cex=.8,nn=T)  #plot head(data)

printcp(fit) #select complexity parameter

prunetree2 = prune(fit, cp=.018)

rpart.plot(prunetree2, cex=.8,nn=T, extra=104)

prunetree2

nrow(data)

table(data$Survived)

# predict for Female, pclass=3, siblings=2, what is the chance of survival



#Predict class category or probabilities

(testdata = sample_n(data,2))

predict(prunetree2, newdata=testdata, type='class')

predict(prunetree2, newdata=testdata, type='prob')

str(data)

testdata2 = data.frame(Pclass=factor(2), Sex=factor('male'), Age=15, SibSp=2)

testdata2

predict(prunetree2, newdata = testdata2, type='class')

predict(prunetree2, newdata = testdata2, type='prob')



#Use decision trees for predicting

#customer is likely to buy a product or not with probabilities

#customer is likely to default on payment or not with probabilities

#Student is likely to get selected, cricket team likely to win etc



#Imp steps

#select columns for prediction

#load libraries, create model y ~ x1 + x2 

#prune the tree with cp value

#plot the graph

#predict for new cases



#rpart, CART, classification model

#regression decision = predict numerical value eg sales

#clustering----
set.seed(1234)
(marks1=trunc(rnorm(n=30, mean=70, sd=8)))
sum(marks1)
df5<-data.frame(marks=marks)
head(df5)
#
km3<-kmeans(df5, centers=3)
attributes(km3)
km3$cluster
km3$centers
sort(df5$marks1)
cbind(df5, km3$cluster) #which row which cluster
df5$cluster=km3$cluster
head(df5)
df5%>%arrange(cluster)
dist(df5[1:5,])
#---------------------------------------------------

set.seed(1234)
(marks1=trunc(rnorm(n=30, mean=70, sd=8)))
(marks2=trunc(rnorm(n=30, mean=120, sd=10)))
df6<-data.frame(marks1, marks2)
head(df6)
#
km3B<-kmeans(df6, centers=3)
attributes(km3B)
km3B$cluster
km3B$centers
km3B$size
cbind(df6, km3B$cluster)
df6$cluster=km3B$cluster
head(df6)
df6%>%arrange(cluster)
df6[1:5,]
dist(df6[1:5,])
plot(df6$marks1, df6$marks2, col=df6$cluster)


#word cloud----
library(wordcloud2)
df=data.frame(word=c('IIML','MDI','IMT','IIT'), freq=c(50,20,23,15))  
df
par(mar=c(0,0,0,0))
wordcloud2(df)

head(demoFreq)
par(mar=c(0,0,0,0))
wordcloud2(demoFreq, size = 2, minRotation = -pi/2, maxRotation = -pi/2)
