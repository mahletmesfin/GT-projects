---
title: "Homework1"
output: html_document
date: "2024-08-27"
---


#### Question 2.1
# 1.) Describe a situation or problem from your job, everyday life, current events, etc., for which a classification model would be appropriate. List some (up to 5) predictors that you might use.
# 
# Problem: You are receiving an excessive amount of emails each day, it is unbearable to look through each one trying to determine which is spam and which is relevant.
# 
# Predictors:

# Keywords- spam emails use a lot of similar phrases to get readers attention such as "Congrats!", "Free", "You Won!"
# 
# Links or attachments- spam emails contain links/ certain attachments e.g. "Click this link to view your free gift!"
# 
# Sender's email: the sender's email address often contains random letters or an easy to spot fake email address.
# 
# Overuse of Punctuation marks: In order to grab reader's attention spammer's often excessively use exclamation marks, special characters, question marks, etc.
# 
# Subject line: Spam emails often contain punctuation marks instead of text to unlock reader's curiosity on what the message might contain.

#*********************************************************************************************************************************************
# Question 2.2
# 1.) Using the support vector machine function ksvm contained in the R package kernlab, find a good classifier for this data. Show the equation of your classifier, and how well it classifies the dara points in the full data set.

# I used the Support Vector Machine function 'ksvm' from the 'kernlab' package to train a linear SVM model on the data set.
# The model uses a linear kernel, and can be represented by the following classifier equation:
# f(x) = sign(w^Tx +b)

# Where:
# X is the feature vector of an observation.
# w is the weight vector (coefficients)
# b is the bias (intercept)

# From the trained model, the coefficients and the intercept are:
#   coefficients:
# A1: -0.0010065348, A2: -0.0011729048, A3: -0.0016261967, A8: 0.0030064203, A9:  1.0049405641, A10: -0.0028259432, A11:  0.0002600295, A12:-0.0005349551, A14:-0.0012283758, A15:  0.1063633995       
 
# The Intercept of the model: 0.08158492

#The model achieved an accuracy of approximately 86.39% on the full data set.


# Install and load package
install.packages("kernlab")
library(kernlab)
exists("ksvm", where = "package:kernlab")

# Set R's working directory to where the script file is then print setwd code (location of code)
setwd("C:/Users/elsab/OneDrive/Documents/GT")

# Run getwd to confirm working directory and presence of file
getwd()

# Run list.files()
list.files()

# Load the file
df <- read.table(file = "credit_card_data-headers.csv",sep = "\t", header = TRUE)

# Run head(), summary(), and dim() on the data ensuring it is what is expected
head(df)
summary(df)
dim(df)

# Check the structure of the data frame
str(df)

# Drop the first column if it's not relevant
df2 <- df[,-1]

# Scaling the data 

df2_scale <-scale(df2)

# Removing missing values
df2_scale_noNA <- na.omit(df2_scale)

# Ensure the response variable is a factor
df2_scale_noNA[, ncol(df2_scale_noNA)] <- as.factor(df2_scale_noNA[,ncol(df2_scale_noNA)])

# Prepare the data for the SVM model
#model_linear <- ksvm (as.matrix(data[,1:10]), as.factor(data[,11]), type = "C-svc", kernel = "vanilladot", C = 100, scaled = TRUE)
model <- ksvm (as.matrix(df[,1:10]),as.factor(df[,11]), type = "C-svc", kernel = "vanilladot", C = 100, scaled = TRUE)
print(model)

# Calculate a1..am (coefficients)
coefficients <- colSums(model@xmatrix[[1]] * model@coef[[1]]) 
print(coefficients)

# Calculate a0 (intercept)
intercept <- -model@b
print(intercept)

# See what the model predicts
pred <- predict(model, df[,1:10])
pred


# See what fraction of the model's prediction match the actual classification
accuracy <- sum(pred == df[,11]) /nrow(df)
accuracy
#****************************************************************************************************************************************

# 2.) You're welcome but not required, to try other(nonlinear) kernels as well; we're not covering them in this course, but they can sometimes be useful and might provide better predictions than vanilladot.

# Using the 'ksvm' function from the 'kernlab' package,I tested the radical basis function ('RBF') kernel in order to see if it would improve classification performance compared to the linear kernel ('vanilladot').
# The accuracy of the SVM model with the RBF kernel was approximately 94.80%.
# This suggests the RBF kernel provided better performance compared to the linear kernel.
# Here is the model used: model_rbf <- ksvm(as.matrix(df2_scale_noNA[, -ncol(df2_scale_noNA)]),
#                                           as.factor(df2_scale_noNA[, ncol(df2_scale_noNA)]),
#                                           type = "C-svc",
#                                           kernel = "rbfdot",
#                                           C = 100,
#                                           scaled = TRUE)


install.packages("kernlab")
library(kernlab)
exists("ksvm", where = "package:kernlab")

# Set R's working directory to where the script file is then print setwd code (location of code)
setwd("C:/Users/elsab/OneDrive/Documents/GT")

# Run getwd to confirm working directory and presence of file
getwd()

# Run list.files()
list.files()

# Load the file
df <- read.table(file = "credit_card_data-headers.csv",sep = "\t", header = TRUE)

# Run head(), summary(), and dim() on the data ensuring it is what is expected
head(df)
summary(df)
dim(df)

# Check the structure of the data frame
str(df)

# Drop the first column if it's not relevant
df2 <- df[,-1]

# Scaling the data 

df2_scale <-scale(df2)

# Removing missing values
df2_scale_noNA <- na.omit(df2_scale)

# Ensure the response variable is a factor
df2_scale_noNA[, ncol(df2_scale_noNA)] <- as.factor(df2_scale_noNA[,ncol(df2_scale_noNA)])

# Train model using RBF kernel
model_rbf <- ksvm(as.matrix(df2_scale_noNA[, -ncol(df2_scale_noNA)]),
                   as.factor(df2_scale_noNA[, ncol(df2_scale_noNA)]),
                   type = "C-svc",
                   kernel = "rbfdot",
                   C = 100,
                   scaled = TRUE)

# Print model
print(model_rbf)

# Predict using RBF model
pred_rbf <- predict(model_rbf, df2_scale_noNA[, -ncol(df2_scale_noNA)])

# Calculate accuracy of RBF model
accuracy_rbf <-sum(pred_rbf == df2_scale_noNA[, ncol(df2_scale_noNA)]) /nrow(df2_scale_noNA)
print(paste("RBF Kernel Accuracy =", accuracy_rbf)) 

#*************************************************************************************************************************************************


# 3.) Using the k-nearest- neighbors classification function 'kknn' contained in the R 'kknn' package, suggest a good value of k, and show how well it classifies that data points in the full data set.
# Using the K-nearest-neighbors classification function 'kknn' from the 'kknn' package, I tested several values of k to determine the most effective classifier for the data set.
# The values tested were 1,3,5,7,9, and 11. The best value was k= 5, which resulted in the highest accuracy of approximately 85.32% on the full data set.
# This value of k provides a good balance between model complexity and performance, as it achieves high accuracy while avoding overfitting.


# Install package
install.packages ("kknn")
library(kknn)

# Set R's working directory to where the script file is then print setwd code (location of code)
setwd("C:/Users/elsab/OneDrive/Documents/GT")

# Run getwd to confirm working directory and presence of file
getwd()

# Run list.files()
list.files()

# Load the file
df <- read.table(file = "credit_card_data-headers.csv",sep = "\t", header = TRUE)

# Run head(), summary(), and dim() on the data ensuring it is what is expected
head(df)
summary(df)
dim(df)

# Check the structure of the data frame
str(df)

# Drop the first column if it's not relevant
df2 <- df[,-1]

# Scaling the data 

df2_scale <-scale(df2)

# Removing missing values
df2_scale_noNA <- na.omit(df2_scale)

# Ensure the response variable is a factor
df2_scale_noNA[, ncol(df2_scale_noNA)] <- as.factor(df2_scale_noNA[,ncol(df2_scale_noNA)])

# Convert the matrix to a data frame
df2_scale_noNA <- as.data.frame(df2_scale_noNA)

# Prepare variables
response_var <- names(df2_scale_noNA)[ncol(df2_scale_noNA)]
formula <- as.formula(paste(response_var,"~ ."))


# Initialize variables, create empty vector for predictions
best_k <- NULL
best_accuracy <- 0

# Try different values for k
k_values <- c(1,3,5,7,9,11)

for (k in k_values)
  
  # Storing predictions
  predictions <- c()

# Loop over each row for leave-one-out cross-validation
for(i in 1:nrow(df2_scale_noNA)) {
  
  # Exclude the i-th data point from the training set
  train_data <- df2_scale_noNA[-i, ]
  
  #Get the i-th data point
  test_data <- df2_scale_noNA[i, , drop = FALSE]
  
  # Fit the KNN model  
  model <- kknn(formula, train_data, test_data, k = k, scale = TRUE)
  
  # Prediction
  predictions[i] <- round (fitted(model))
  print(predictions)
}
  
  # Calculate accuracy
  
  accuracy <- sum(predictions == df2_scale_noNA[,response_var]) /nrow(df2_scale_noNA)
  
  if(accuracy > best_accuracy){
    best_accuracy <- accurracy
    best_k <-k
  }
  print(paste("k =", k, "Accuracy =", accuracy))
  
  print(paste("Best k =", best_k, "with Accuracy =", best_accuracy))
#*************************************************************************************************************************  
  
# CITATIONS
# Title: Setting a Working Directory
# Author: RPubs
# Date: November 19, 2022
# Availability: https://rpubs.com/em_/wdInR

# Title: OpenAI
# Author:ChatGBT
# Date:August 24, 2024
# Code version: 4
# Availability: https://www.openai.com/chatgpt


  







