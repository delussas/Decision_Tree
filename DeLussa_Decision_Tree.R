### Decision Trees Project
# Scott DeLussa 7/7/2020

setwd("~/Documents/Stockton Graduate/Machine_Learning/Decision Trees")

library(gmodels)
library(rpart)
library(rpart.plot)
library(caret)

# Load the data and get a general idea of what it consists of
loan_data <- read.csv("credit_risk_dataset.csv", header = T, sep = ',')
str(loan_data)

# I want to create a decision tree model to predict credit defaults, so I will look at the loan status to see the distribution of non defaults to defaults. Considering more people payoff their loan than default, it is safe to assume 0 is non default and 1 is default
barplot(table(loan_data$loan_status))

# Familiarization 
# Defaults represent less than one quarter of the data set
CrossTable(loan_data$loan_status)

# This table shows the relationship between the grade, or the risk of the loan, and the loan status, or the outcome of non default/ default. As the grade decreases, defaults increase. 
CrossTable(loan_data$loan_grade, loan_data$loan_status, prop.r = T, prop.c = F, prop.t = F, prop.chisq = F)

# This histogram shows the number of loans per each dollar ammount borrowed
hist_1 <- hist(loan_data$loan_amnt)

# Using the breaks argument, we can see how the loans are split up / grouped together. $2,000 increments seems large considering the majority of the loans are below $15,000  
hist_1$breaks

# It looks like most loans are taken in $5,000 increments
hist_2 <- hist(loan_data$loan_amnt, breaks = 100, xlab = "Loan Amount", main = "Histogram of the Loan Amounts")

# The intent of loans are pretty evenly distributed, with defauults occuring most often in Debt Consolidation, and least often in Venture loans
CrossTable(loan_data$loan_intent, loan_data$loan_status, prop.r = T, prop.c = F, prop.t = F, prop.chisq = F)

# Now to create the trees

# Set a seed to reproduce the data
set.seed(123)

# Using the sample funciton, we created a training data set that consists of 2/3 of the observations in the entire data set. First, we store the row indexed numbers of a random sample in the variable index_train, then we use those index numbers for the training set. Likewise, we excluse the training set indexed observations for to create an unseen test set
index_train <- sample(1:nrow(loan_data), 2/3 * nrow(loan_data))
training_set <- loan_data[index_train, ]
test_set  <- loan_data[-index_train, ]



### Constructing a decsion tree which includes r.part.control to relax the complexity parameter from .01 to .001. The complexity parameter is the treshold for the model to stop creating splits. If the cost of adding another variable to the decision tree from the current node is above the value of cp, then tree building does not continue. Relaxing the cp is advised on complex data sets. The method argument is used to create a classification tree, which is necessary for predicting a default, which is a qualitative event.
decision_tree <- rpart(loan_status ~ ., method = "class", data = training_set, control = rpart.control(cp = 0.001))

# Plot the decision tree, the argument uniform = TRUE is used to get equal-sized branches. The text function applies the categories to their respective node, essentially labeleing the decision tree
plot(decision_tree, uniform = T)
text(decision_tree)

# Construct a decision tree including the argument parms and changin the proportion of non defaults to 0.7 and the defaults to .3 (they should always sum up to 1). Additionally, include control = rpart.control(cp = 0.002) as well.
decision_balanced <- rpart(loan_status ~ ., method = "class", data = training_set, control = rpart.control(cp = 0.001), parms = list(prior = c(.7, .3)))

# Plot the decision tree
plot(decision_balanced, uniform = T)
text(decision_balanced)



### Considering the size of the above tree, overfitting is possible. To combat this, we can prune the trees. Plotcp allows us to analyse cross validation error results. It's similar to an elbow curve in that you want to select the number of splits with the minimum number of cross validation errors in the tree, keeping the size of the tree as small as possible
plotcp(decision_balanced)

# The below function prints a table of information about CP, splits, and errors.
printcp(decision_balanced)

# To simplify the above table, we can leverage the which.min function to find the minimum cross validation error. 
index <- which.min(decision_balanced$cptable [ , "xerror"])
index

# Use the above index to select the index of the minimum cross validation error within the CP table
tree_min <- decision_balanced$cptable[index, "CP"]

# Used the prune() function to obtain the pruned tree. 
pdecision_balanced <- prune(decision_balanced, cp = tree_min)

# Plotting the pruned tree using the prp funciton
prp(pdecision_balanced)



### Now we will include a loss matrix, changing the relative importance of misclassifying a default as non-default versus a non-default as a default. We will create a matrix that penalizes a misclassification of a default as a nondefault 10 times more than a mislassification of a nondefault as a default. Again, 0 represents nondefault and 1 represents default.
tree_loss_matrix <- rpart(loan_status ~ ., method = "class", data = training_set, control = rpart.control(cp = .001), parms = list(loss = matrix(c(0, 10, 1, 0), ncol = 2)))
plot(tree_loss_matrix, uniform = T)
text(tree_loss_matrix)

# Pruning the tree with the loss matrix

# Creates an index for of the row with the minimum cross validation error
loss_tree_index <- which.min(tree_loss_matrix$cptable [ , "xerror"])

# Creates an index to select the index of the minimum cross validation error within the CP table
loss_tree_min <- tree_loss_matrix$cptable[loss_tree_index, "CP"]

# Prune the tree using cp = loss_tree_min
ptree_loss_matrix <- prune(tree_loss_matrix, cp = loss_tree_min)

# Plot the pruned tree using the prp function. Using the argument extra = 1, displays the number of observations that fall in the node
prp(ptree_loss_matrix, extra = 1)



### Lastly, we can also assign weights to outcomes. Here we will use a weight of 3 for defaults and 1 for nondefaults. Since defaults are more heavily weighted, the model will assign a higher importance to classifying defaults correctly. The below code reads, if the result of a loan status is 0, or non default, the weight is 1, otherwise, the weight is 3.
event_weight <- ifelse(training_set$loan_status == 0, 1, 3)

# Set the minimum number of splits that are allowed in a node to 5, and the minimum number of observations allowed in leaf nodes to 2 by using the arguments minsplit and minbucket in rpart.control respectively.
tree_weights <- rpart(loan_status ~ ., method = "class", data = training_set, weights = event_weight, control = rpart.control(minsplit = 5, minbucket = 2, cp = .001))

# Again, see where the cross validation error rate can be minimized
plotcp(tree_weights)

# Creates an index for the row with the minimum cross validation error
weight_index <- which.min(tree_weights$cptable[ , "xerror"])

# Create the minimum value for the CP
weight_min <- tree_weights$cptable[weight_index, "CP"]

# Now, prune the tree using the comlpexity parameter where the cross validation error is minimized
ptree_weights <- prune(tree_weights, cp = weight_min)

# Plot the above pruned tree
prp(ptree_weights, extra = 1)



### Now use the test set to make predicitons based on the models created above. Use the pred function, along with each pruned tree, and the test data set. Again, this is a classificaition model as specified by the type
pred_balanced <- predict(pdecision_balanced, newdata = test_set, type = "class")
pred_loss_matrix <- predict(ptree_loss_matrix, newdata = test_set, type = "class")
pred_weights <- predict(ptree_weights, newdata = test_set, type = "class")

# Below is a confusion matrix for each prediction. The confusion matrix is a 2x2 table that lists:
# True Positives: The number of 
confmat_balanced <- table(test_set$loan_status, pred_balanced)
confmat_balanced

confmat_loss_matrix <- table(test_set$loan_status, pred_loss_matrix)
confmat_loss_matrix

confmat_weights <- table(test_set$loan_status, pred_weights)
confmat_weights

# Now we can compute the accuracy for each model. We can see the balanced model barely beats out the weighted model.
acc_balanced <- sum(diag(confmat_balanced)) / nrow(test_set)
acc_balanced

acc_loss_matrix <- sum(diag(confmat_loss_matrix)) / nrow(test_set)
acc_loss_matrix

acc_weights <- sum(diag(confmat_weights)) / nrow(test_set)
acc_weights

# Next, we can compute the sensitivity and specificity
# The sensitivity represents the proportion of true positives correctly identified
sens_balanced <- (confmat_balanced[1]) / (confmat_balanced[1]+confmat_balanced[2])
sens_balanced
# The specificity represents the proportion of true negatives correctly identified
spec_balanced <- (confmat_balanced[4]) / (confmat_balanced[3]+confmat_balanced[4])
spec_balanced

# Using the caret package, we can also verify the sensitivity and specificity more efficiently
sensitivity(confmat_balanced)
specificity(confmat_balanced)

sensitivity(confmat_loss_matrix)
specificity(confmat_loss_matrix)

sensitivity(confmat_weights)
specificity(confmat_weights)

# Finally we can conclude the balanced model is the superior model for this data set, given it has the highest accuracy, specificity, and is a close second in sensitivity. 

