# data_mining
## Description 
This project is composed of two data mining tasks, a regression problem and a classification problem. The regression problem is to predict the boston housing price given a bunch of characteristics. The classification task is to classify tumor into malignant or benign given some of the attributes of a tumor. 
## Boston Housing Project
This project uses negative mean squared error as the scoring metric. I investigate the performance of OLS, Lasso, ElasticNet, Gradient Boosting Trees and GLM-Gamma. Out of these methods, GLM-Gamma seems to perform the best as Gamma can approximate an exponential distribution, which is assumed to be the distribution followed the housing price.
In the future, I'm going to try Gamma Lasso for regression tasks with many features and a response following Gamma distribution.
