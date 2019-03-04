data= import
head(traintry)
traintry <- train
traintry$experiment<- NULL
traintry$event<- NULL
traintry.pca <- prcomp(traintry[,c(1:26)], center = TRUE,scale. = TRUE)
summary(traintry.pca)
str(traintry.pca)
library(FactoMineR)
traintry2 = PCA(traintry, graph = FALSE)
traintry2$eig
traintry2$var
