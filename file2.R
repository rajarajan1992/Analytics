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
                                            