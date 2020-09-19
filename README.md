# deployMachineLearning


# Feature Engineering
> Missing data

> **Labels** are present as categorical variables and cannot be used in training machine learning models
**Cardinality** problem is there as many unique variables exist for a feature
**One Techniques** can be hot encoding or creating dummy variables.
However it may become difficult and computational for models such as decision trees

> **Distribution** of variables is important for some alogorithms , such as regression models, clustering algorithms

> **Scaling** of variables can effect model performance especially if there is a big difference in the weights of the numerical value

> **Some algorithms** are sensitive to outliers


# Feature Selection
> Best features that will help us predict the future value
> Less features and simple models are easy to interpret and implement
>  Shorter training times
> Enhanced generalisation by reduce over fitting
> Less json messages to be sent to the model
> less lines of code for error handling, error handlers need to be written for each variable/input
> Less information to log
> Less feature engineering code to treat dataß