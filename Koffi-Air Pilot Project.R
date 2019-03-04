library(ggplot2)
library(dplyr)
library(caTools)
library(caret)
library(e1071)
library(AppliedPredictiveModeling)

#Import the dataset
Train <- read.csv("train.csv", stringsAsFactors=FALSE)
numeric <- sapply(Train, is.numeric)

#Correlation between predictors
library(corrplot)
cor(Train[numeric]) 
corrplot(cor(Train[numeric]), order="hclust")

#Scaling Variables
Trainpp <- preProcess(Train[numeric], method = "scale")
Train_set <- predict(Trainpp, Train[numeric]) 

# I- Apply PCA on all predictors
TrainPCA <- prcomp(Train_set, scale. = TRUE)

# Importance of components
summary(TrainPCA)

#PCA components
head(TrainPCA$rotation)

#show the transformed values  
head(TrainPCA$x)

# Select feature variables 
screeplot(TrainPCA, type="lines")

#Compute the percentage of variance for each component
percentVariancePCA = TrainPCA$sd^2/sum(TrainPCA$sd^2)*100

#Percentage Variance plot
plot(percentVariancePCA, xlab="Number of Components", ylab="Percentage of Total Variance", type="l", main="PCA")

#add a training set with principal components
FinalTrain <- data.frame(Train[2], TrainPCA$x, Train[28])

#we are interested in first 16 PCAs, the experiment and the event
FinalTrain <- subset(FinalTrain, select= c(1:17, 28))

#Encode categorical data
FinalTrain$experiment <- as.numeric(factor(FinalTrain$experiment,
                                           levels = c('CA', 'DA', 'SS'),
                                           labels = c(1, 2, 3)))
FinalTrain$event <- as.numeric(factor(FinalTrain$event,
                                      levels = c('A', 'B', 'C', 'D'),
                                      labels = c(1, 2, 3, 4)))
