
install.packages("lawstat")
library(lawstat)
install.packages("Rcmdr")
library(Rcmdr)
install.packages("ggplot2")
library(ggplot2)
install.packages("dplyr")
library(dplyr)
install.packages("nortest")
library(nortest)
install.packages("stats")
library(stats)
####Question 1 Applebees Dataset

#Load in data
Applebees=read.table('Q1.txt', header =T)


####1.a 
#ANOVA table based on revenue and restaurant
A =aov(Revenue~as.factor(Restaurant), Applebees)
summary(A)

####1.b
#Removing Restaurant 3 so we can do appropriate comparison betwen Restuarant 1 and 2
Applebees_b = subset(Applebees, Restaurant == 1 | Restaurant == 2)
Applebees_b <- setNames(Applebees_b, c("Revenue_b", "Restaurant_b", "Week_b"))
attach(Applebees_b)

#t-test
t.test(Revenue_b~Restaurant_b, var.equal = TRUE, Applebees_b)

####1.d
#To test normality
#Visual diagnostic QQ plot
qqnorm(Revenue_b)
qqline(Revenue_b)

### Anderson-Darling

ad.test(Revenue_b)

#To test Homogenous Variance
#Visaul Diagnostic side by side boxplot
ggplot(data = Applebees_b, mapping = aes(group=Restaurant_b , x=Restaurant_b, y = Revenue_b)) +
  geom_boxplot() +
  labs(y = "Revenue", x= "Restaurant", title= "Revenue by Restaurant Box Plot")

#Databased diagnostic
leveneTest(Revenue_b,Restaurant_b, data=Applebees_b)

####1.e
#Creating new intermediate model
Applebees_c <-Applebees
Applebees_c["New_Restaurant"]<- NA
Applebees_c$New_Restaurant <- c(1,1,1,1,1,1,1,1,1,1,2,2,2,2,2)
Applebees_c <- setNames(Applebees_c, c("Revenue_c", "Restaurant_c", "Week_c", "New_Restaurant_c"))
attach(Applebees_c)

#ANOVA table based on new model
C =aov(Revenue_c~as.factor(New_Restaurant_c), Applebees_c)
summary(C)

####1.g
#To test normality
#Visual diagnostic QQ plot
qqnorm(Revenue_c)
qqline(Revenue_c)

#### Anderson-Darling

ad.test(Revenue_c)

#To test Homogenous Variance
#Visaul Diagnostic side by side boxplot
ggplot(data = Applebees_c, mapping = aes(group=New_Restaurant_c , x=New_Restaurant_c, y = Revenue_c)) +
  geom_boxplot() +
  labs(y = "Revenue", x= "Restaurant", title= "Revenue by Restaurant Box Plot")

#Databased diagnostic
attach(Applebees_c)
leveneTest(Revenue_c,as.factor(New_Restaurant_c), data=Applebees_c)


####1.i

D =aov(Revenue~as.factor(Restaurant)+as.factor(Week), Applebees)
summary(D)

####1.k
#Creating intermediate model
Applebees_d <-Applebees
Applebees_d["New_Restaurant"]<- NA
Applebees_d$New_Restaurant <- c(1,1,1,1,1,1,1,1,1,1,2,2,2,2,2)
Applebees_d <- setNames(Applebees_d, c("Revenue_d", "Restaurant_d", "Week_d", "New_Restaurant_d"))
attach(Applebees_d)

#ANOVA table based on new model
E =aov(Revenue_d~as.factor(New_Restaurant_d)+as.factor(Week_d), Applebees_d)
summary(E)

####1.m
#To test normality
#Visual diagnostic QQ plot
qqnorm(Revenue_d)
qqline(Revenue_d)

### Anderson-Darling

ad.test(Revenue_d)

#To test Homogenous Variance
#Databased diagnostic
leveneTest(Revenue_d~as.factor(Restaurant_d), data=Applebees_d)

####1.n
J =aov(log(Revenue)~as.factor(Restaurant)+as.factor(Week), Applebees)
summary(J)

####1.o
#Rank-sum procedure for 1.b
wilcox.test(Revenue_b~as.factor(Restaurant_b))

#Log transofrmation w/ ANOVA

#Tukeys test for additivity, check for interaction effect
tukeys.add.test <- function(y,A,B){
  ## Y is the response vector
  ## A and B are factors used to predict the mean of y
  ## Note the ORDER of arguments: Y first, then A and B
  dname <- paste(deparse(substitute(A)), "and", deparse(substitute(B)),
                 "on",deparse(substitute(y)) )
  A <- factor(A); B <- factor(B)
  ybar.. <- mean(y)
  ybari. <- tapply(y,A,mean)
  ybar.j <- tapply(y,B,mean)
  len.means <- c(length(levels(A)), length(levels(B)))
  SSAB <- sum( rep(ybari. - ybar.., len.means[2]) * 
                 rep(ybar.j - ybar.., rep(len.means[1], len.means[2])) *
                 tapply(y, interaction(A,B), mean))^2 / 
    ( sum((ybari. - ybar..)^2) * sum((ybar.j - ybar..)^2))
  aovm <- anova(lm(y ~ A+B))
  SSrem <- aovm[3,2] - SSAB
  dfdenom <- aovm[3,1] - 1
  STATISTIC <- SSAB/SSrem*dfdenom
  names(STATISTIC) <- "F"
  PARAMETER <- c(1, dfdenom)
  names(PARAMETER) <- c("num df", "denom df")
  D <- sqrt(SSAB/  ( sum((ybari. - ybar..)^2) * sum((ybar.j - ybar..)^2)))
  names(D) <- "D estimate"
  RVAL <- list(statistic = STATISTIC, parameter = PARAMETER, 
               p.value = 1 - pf(STATISTIC, 1,dfdenom), estimate = D,
               method = "Tukey's one df F test for Additivity", 
               data.name = dname)
  attr(RVAL, "class") <- "htest"
  return(RVAL)
}
  
  tukeys.add.test(y = Revenue, A= Restaurant, B = Week)


####Question 2 Dog and Cat Food Data
#Load in data
Iams=read.table('Q2.txt',header=T)
head(Iams)

####2.b
M =aov(Fat~as.factor(Formula)*as.factor(Plant), Iams)
summary(M)


####2.e
#Formula 4 has the lowest avg fat, see if its equal to 9
#Creating new dataset with only Formula 4
Iams_reducedfat <- subset(Iams, Formula == 4)
#Dropping Formula and Plant variables
Iams_reducedfat <- Iams_reducedfat %>% select (-Formula, -Plant)

#One sample t-test
ab <-t.test(Iams_reducedfat, mu=9)


####2.f
#Main effect of Formula at level 1
#Creating new dataset for Formula at level 1
Formula_1_dataset <- subset(Iams, Formula == 1)
Formula_1_dataset <- setNames(Formula_1_dataset, c("Fat_1", "Formula_1", "Plant_1"))
attach(Formula_1_dataset)
#Mean response at level 1 of Formula
mean(Fat_1)

#Grand mean
mean(Fat)

#Main effect of Formula at level 1
mean(Fat_1) - mean(Fat)

#Main effect of Formula at level 2
#Creating new dataset for Formula at level 2
Formula_2_dataset <- subset(Iams, Formula == 2)
Formula_2_dataset <- setNames(Formula_2_dataset, c("Fat_2", "Formula_2", "Plant_2"))
attach(Formula_2_dataset)
#Mean response at level 2 of Formula
mean(Fat_2)

#Grand mean
mean(Fat)

#Main effect of Formula at level 1
mean(Fat_2) - mean(Fat)

####2.g

#Testing assumptions
ad.test(Fat)

#Testing homogenous variance
leveneTest(Fat~as.factor(Formula), data=Iams)


#Family wise confidence Intervals
lm(formula = Fat~as.factor(Formula)*as.factor(Plant), data=Iams )

M =aov(Fat~as.factor(Formula)*as.factor(Plant), Iams)
summary(M)
