# Ahaan Tagare
# Student - 33865799

# STATISTICS AND STATISTICAL DATA MINING

### **Predictive Modelling for Loan Approval: A Comparative Analysis of Machine Learning Techniques**

#Dataset - <https://www.kaggle.com/datasets/zaurbegiev/my-dataset?select=credit_train.csv>

#The code's libraries perform a number of vital operations for data analysis and machine learning projects While dplyr and tidyr are used for data manipulation and tidying, R. gbm is used for gradient boosting models. Caret provides a uniform interface for training and assessing machine learning models, while ggplot2 makes sophisticated visualizations possible. ROC curves and AUC are used by pROC to assess classifier performance, while randomforest creates random forest models. rpart and rpart.plot with an emphasis on decision tree visualizations and models. PDP uses partial dependence plots to help visualize model behavior, and Rumble offers a graphical user interface for data science. NeuralNetTools helps with neural network analysis, nnet is used for neural network training, and coefplot visualizes regression model coefficients. Data preparation and imbalanced datasets are handled by DMwR, while DiagrammeR generates graphical diagrams for network or workflow visualization. Finally, data can be reshaped between wide and long formats using reshape2. Together, these packages offer a complete toolkit for modeling, data wrangling, visualization, and model validation, have suppressed the warnings because of the setting control in the PC in this code which would make no impact on the model.

suppressPackageStartupMessages({
  suppressWarnings({
    library(gbm)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(caret)
    library(randomForest)
    library(pROC)
    library(rpart)
    library(rpart.plot)
    library(rattle)
    library(pdp)
    library(coefplot)
    library(NeuralNetTools)
    library(nnet)
    library(DMwR)
    library(DiagrammeR)
    library(reshape2)
    library(scales)
  })
})


# Data preprocessing
data <- read.csv("Credit_data.csv")
str(data)


# Create the flowchart
flowchart <- grViz("
digraph workflow {
  graph [layout = dot, rankdir = TB]

  # Nodes
  node [shape = box, style = filled, color = lightblue]
  A [label = 'Load and Inspect Data']
  B [label = 'Data Cleaning and Preprocessing']
  C1 [label = 'Handle Missing Values']
  C2 [label = 'Variable Conversion']
  D [label = 'Feature Engineering']
  D1 [label = 'Calculate DTI']
  D2 [label = 'Credit Utilization']
  E [label = 'Train-Test Split']
  F [label = 'Model Training and Evaluation']
  F1 [label = 'Decision Tree']
  F2 [label = 'Random Forest']
  F3 [label = 'Logistic Regression']
  F4 [label = 'Gradient Boosting']
  F5 [label = 'Artificial Neural Network']
  G [label = 'Model Evaluation']
  G1 [label = 'Accuracy, Precision, Recall, F1 Score']
  H [label = 'Model Comparison']
  I [label = 'ROC Analysis']

  # Edges
  A -> B
  B -> C1
  B -> C2
  C1 -> D
  C2 -> D
  D -> D1
  D -> D2
  D1 -> E
  D2 -> E
  E -> F
  F -> F1
  F -> F2
  F -> F3
  F -> F4
  F -> F5
  F1 -> G
  F2 -> G
  F3 -> G
  F4 -> G
  F5 -> G
  G -> G1
  G1 -> H
  H -> I
}
")

# Render the flowchart


#This code prepares data for a pie chart that displays column counts by loading a dataset and counting non-missing values for each column. After determining the cumulative midpoints and proportions for label placement, it gives each column a distinct color. The text labels are positioned inside the chart slices after a pie chart with personalized labels, colors, and a legend is created using ggplot2. As a result, column names and counts are shown in an eye-catching pie chart.

# Data preprocessing
data <- read.csv("Credit_data.csv")  
str(data)  


column_names <- colnames(data)  
counts <- colSums(!is.na(data))  

# Prepare the data forpie chart
pie_data <- data.frame(Column = column_names, Count = counts)


pie_data$Proportion <- pie_data$Count / sum(pie_data$Count)  
pie_data$Cumulative <- cumsum(pie_data$Proportion) - pie_data$Proportion / 2  


pie_data$Legend_Label <- paste0(pie_data$Column, " - ", scales::comma(pie_data$Count))

 
color_palette <- colorRampPalette(c("lightblue", "darkblue"))(nrow(pie_data))  



ggplot(pie_data, aes(x = "", y = Count, fill = Legend_Label)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  theme_void() +
  labs(
    title = "Pie Chart of Dataset Columns with Counts",
    fill = "Columns"  
  ) +
  theme(legend.position = "right") +  
  scale_fill_manual(
    values = setNames(color_palette, pie_data$Legend_Label),  # Match colors explicitly to labels
    labels = pie_data$Legend_Label
  ) +
  geom_text(
    aes(
      y = Cumulative * sum(Count), 
      label = paste(Column, scales::comma(Count), sep = "\n")  # Add names and counts
    ),
    size = 3,  
    angle = 90 - (pie_data$Cumulative * 360),  
    hjust = 0.5, color = "black"  
  )




#This code preprocesses the data by substituting 0 for any missing values in the dataset. The "Short Term" value is changed to 0 and other values to 1, converting the Term column to binary values. Additionally, it uses the mean of the current non-missing values to fill in the missing values in the Months.since.last.delinquent column. These changes get the data ready for additional modeling or analysis.


data[is.na(data)] <- 0
data$Term <- ifelse(data$Term == "Short Term", 0, 1)
data$Months.since.last.delinquent[is.na
(data$Months.since.last.delinquent)] <- mean(data$Months.since.last.delinquent, na.rm = TRUE)


#This code cleans the `Years.in.current.job` column by removing the word years and replacing 10+ with 10. It converts the column to numeric, suppressing warnings for non-numeric values. Missing or invalid values are replaced with 0. Finally, it prints the dimensions of the cleaned data set using `dim(data)`.


# Clean Years 

data$Years.in.current.job <- gsub(" years", "", data$Years.in.current.job)
data$Years.in.current.job[data$Years.in.current.job == "10+"] <- 10
data$Years.in.current.job <- suppressWarnings(as.numeric(data$Years.in.current.job))
data$Years.in.current.job[is.na(data$Years.in.current.job)] <- 0

print(dim(data))


#The Home.Ownership and Purpose columns are transformed into factors by this code. For both columns, it retrieves the levels (categories) and the numeric codes that correspond to them. It generates a table for each that associates the levels with their respective numeric codes (purpose_table for Purpose and home_ownership_table for Home.Ownership). The mapping of factor levels to corresponding codes is then shown by printing these tables.


# Converting the columns to factors
data$Home.Ownership <- as.factor(data$Home.Ownership)
data$Purpose <- as.factor(data$Purpose)


home_ownership_levels <- levels(data$Home.Ownership)
home_ownership_codes <- as.integer(data$Home.Ownership)


purpose_levels <- levels(data$Purpose)
purpose_codes <- as.integer(data$Purpose)


home_ownership_table <- data.frame(Level = home_ownership_levels, 
Code = unique(home_ownership_codes))


purpose_table <- data.frame(Level = purpose_levels,
Code = unique(purpose_codes))


print(home_ownership_table)
print(purpose_table)






#This code performs feature engineering by creating two new features: `DTI` (Debt-to-Income Ratio) and `Credit.Utilization`. The `DTI` is calculated as the ratio of `Monthly.Debt` to `Annual.Income`, and `Credit.Utilization` is calculated as the ratio of `Current.Credit.Balance` to `Maximum.Open.Credit`. To ensure numerical stability, any missing or infinite values resulting from these calculations are replaced with 0. Finally, the first few values of both `DTI` and `Credit.Utilization` are printed using `head()` to verify the successful computation and handling of these features.


# Feature Engineering
data$DTI <- data$Monthly.Debt / data$Annual.Income
data$DTI[is.na(data$DTI) | is.infinite(data$DTI)] <- 0
data$Credit.Utilization <- data$Current.Credit.Balance / 
data$Maximum.Open.Credit
data$Credit.Utilization[is.na(data$Credit.Utilization)] <- 0

print(head(data$DTI)) 
print(head(data$Credit.Utilization))


# This code creates a new binary variable, LoanApproval, by assigning a value of `1` if `Current.Loan.Amount` exceeds 500,000 and `0` otherwise. After printing the dimensions of the dataset (`dim(data)`), it partitions the data set into training and testing subsets using a 70-30 split, controlled by setting a random seed (`set.seed(123)`) for reproducibility. The `createDataPartition()` function ensures that the distribution of `LoanApproval` is preserved in both subsets. Finally, it prints the dimensions of `train_data` and `test_data` to confirm the partitioning.

data$LoanApproval <- ifelse(data$Current.Loan.Amount > 500000, 1, 0)
print(dim(data))


trainIndex <- createDataPartition(data$LoanApproval, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

dim(train_data)  
dim(test_data)


#Finally, it prints the dimensions of `train_data` and `test_data` to confirm the partitioning.undefined`undefined`undefined`undefined`undefined`undefined`undefined


train_data[sapply(train_data, is.infinite)] <- 0
test_data[sapply(test_data, is.infinite)] <- 0




#The code first uses scale to standardize the numeric columns that are extracted from train_data. The standardized data (prcomp) is then subjected to Principal Component Analysis (PCA), and a summary of the PCA is then given. To see each principal component's explained variance, a scree plot is created. Head(pca_train_data) is used to display the pca_train_data, which is created by combining the target variable LoanApproval with the first few principal components that are kept in pca_data.\


#Box plot
train_data$Dataset <- "Train"
test_data$Dataset <- "Test"

# Combine 
combined_data <- rbind(
  train_data[, c("Annual.Income", "Dataset")],
  test_data[, c("Annual.Income", "Dataset")]
)

# Filter 
filtered_data <- combined_data[combined_data$Annual.Income < 550000, ]  # Keep values < 500,000

# box plot
ggplot(filtered_data, aes(x = Dataset, y = Annual.Income, fill = Dataset)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 8, alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "Box Plot of Annual Income by Dataset (Filtered)",
    x = "Dataset",
    y = "Annual Income"
  ) +
  scale_fill_manual(values = c("Train" = "skyblue", "Test" = "orange")) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  scale_y_continuous(labels = scales::comma)  



#The code first uses scale to standardize the numeric columns that are extracted from train_data. The standardized data (prcomp) is then subjected to Principal Component Analysis (PCA), and a summary of the PCA is then given. To see each principal component's explained variance, a scree plot is created. Head(pca_train_data) is used to display the pca_train_data, which is created by combining the target variable LoanApproval with the first few principal components that are kept in pca_data.




train_data_numeric <- train_data[, sapply(train_data, is.numeric)]

# Standardize 
train_data_scaled <- scale(train_data_numeric)

pca <- prcomp(train_data_scaled, center = TRUE, scale. = TRUE)

# Summary 
summary(pca)

screeplot(pca, main = "Scree Plot")
pca_data <- data.frame(pca$x)


pca_train_data <- cbind(pca_data, LoanApproval = train_data$LoanApproval)

head(pca_train_data)


#The code trains a Decision Tree model (`dt_model`) on the `train_data` dataset, using several predictors and the `LoanApproval` target variable with the method `class` for classification. It then visualizes the trained decision tree using `rpart.plot`. The model is used to make predictions (`dt_predictions`) on the `test_data`, ensuring that both the predictions and actual values have the same factor levels. The confusion matrix (`dt_conf_matrix`) is calculated to evaluate the model's performance, and it is printed for review.



# Decision Tree Model
dt_model <- rpart(
  LoanApproval ~ Annual.Income + DTI + Credit.Score + 
    Years.in.current.job + Credit.Utilization + Home.Ownership + Purpose,
  data = train_data,
  method = "class",    
  control = rpart.control(cp = 0.01)  
)

# Visualize 
rpart.plot(dt_model, main = "Decision Tree for Loan Approval")

# Test Data
dt_predictions <- predict(dt_model, test_data, type = "class")  


test_data$LoanApproval <- factor(test_data$LoanApproval)
dt_predictions <- factor(dt_predictions, levels = levels(test_data$LoanApproval))  

dt_conf_matrix <- confusionMatrix(dt_predictions, test_data$LoanApproval)

print(dt_conf_matrix)



#The code uses the train_data data set, a number of predictors, and the Loan Approval goal variable to train a Random Forest model (rf_model). The trained model (rf_predictions) is then used to forecast loan approval results on the test_data. The model's performance is assessed by converting the predictions to factors and computing a confusion matrix (rf_conf_matrix). Finally, a partial Credit dependency diagram.To illustrate how credit varies, a score feature is created.Score has an impact on the model's forecasts.\


# Random Forest Model
train_data$LoanApproval <- factor(train_data$LoanApproval)
test_data$LoanApproval <- factor(test_data$LoanApproval)

rf_model <- randomForest(
  LoanApproval ~ Annual.Income + DTI + Credit.Score + Years.in.current.job + Credit.Utilization + Home.Ownership + Purpose,
  data = train_data,
  ntree = 100
)

rf_predictions <- predict(rf_model, test_data)
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$LoanApproval)

# Test Data
rf_predictions <- predict(rf_model, test_data)

# Convert 
rf_predictions <- factor(rf_predictions, levels = levels(test_data$LoanApproval))

# Confusion Matrix
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$LoanApproval)

print(rf_conf_matrix)

partialPlot(rf_model, train_data, Credit.Score, main = "Partial Dependence: Credit.Score")

# Based on many predictors, the algorithm fits a Logistic Regression model (log_reg_model) to forecast LoanApproval. It models the likelihood of loan approval using the glm function with a binomial family. The test data is used to make the predictions, and anticipated probabilities greater than 0.5 are categorized as 1 (approved). The confusion matrix, or log_reg_conf_matrix, is used to evaluate the model's performance.\

# Logistic Regression Model
log_reg_model <- glm(LoanApproval ~ Annual.Income + DTI + 
Credit.Score + Years.in.current.job + Credit.Utilization + 
Home.Ownership + Purpose,data = train_data, family = binomial)

log_reg_predictions <- predict(log_reg_model, test_data, type = "response")
log_reg_predictions <- ifelse(log_reg_predictions > 0.5, 1, 0)
log_reg_predictions <- factor(log_reg_predictions, levels = levels(test_data$LoanApproval))
log_reg_conf_matrix <- confusionMatrix(log_reg_predictions, 
test_data$LoanApproval)

print(log_reg_conf_matrix)

coefplot(log_reg_model, intercept = FALSE, main = "Feature Importance")

# Extract Coefficients
coefficients <- summary(log_reg_model)$coefficients

# Data Frame
coeff_df <- data.frame(
  Predictor = rownames(coefficients),
  Estimate = coefficients[, "Estimate"]
)

# Remove the intercept 
coeff_df <- coeff_df[coeff_df$Predictor != "(Intercept)", ]


# Bar Chart
ggplot(coeff_df, aes(x = reorder(Predictor, Estimate), y = Estimate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Logistic Regression Coefficients",
       x = "Predictor Variables",
       y = "Coefficient Estimate")

# The goal variable, LoanApproval, is used to train the Gradient Boosting model (gb_model), which makes predictions about it using a number of features. The predict() method is used to make the model's predictions, and the results are classified using a threshold of 0.5. A confusion matrix is generated to evaluate the model's performance, comparing predicted values with actual values Lastly, the summary() function is used to illustrate the feature importance, illustrating how each predictor affects the model's predictions.

# Gradient Boosting Model
train_data$LoanApproval <- as.numeric(as.character(train_data$LoanApproval))
test_data$LoanApproval <- as.numeric(as.character(test_data$LoanApproval))

gb_model <- gbm(LoanApproval ~ Annual.Income + DTI + 
Credit.Score + Years.in.current.job + Credit.Utilization + 
  Home.Ownership + Purpose,
                data = train_data,
                distribution = "bernoulli",
                n.trees = 100,
                interaction.depth = 3,
                shrinkage = 0.01)

gb_predictions <- predict(gb_model, test_data, 
n.trees = 100, type = "response")
gb_predictions <- ifelse(gb_predictions > 0.5, 1, 0)
gb_predictions <- factor(gb_predictions, levels = c(0, 1))
test_data$LoanApproval <- factor(test_data$LoanApproval, 
levels = c(0, 1))
gb_conf_matrix <- confusionMatrix(gb_predictions, 
test_data$LoanApproval)

print(gb_conf_matrix)

# Visualize 
summary(gb_model, main = "Feature Importance - Gradient Boosting Model")


# The Neural Network model (`ann_model`) is trained using the `nnet` package, with 5 hidden neurons and regularization (decay) to avoid overfitting. After training, predictions are made on the test data, and the results are converted to a factor for comparison with actual values. A confusion matrix (`ann_conf_matrix`) is generated to evaluate the model's performance. The `plotnet()` function is used to visualize the neural network structure, showcasing the relationships between input features and output predictions.


train_data$LoanApproval <- factor(train_data$LoanApproval, levels = c(0, 1))
test_data$LoanApproval <- factor(test_data$LoanApproval, levels = c(0, 1))

# Neural Network Model
set.seed(123)  # For reproducibility
ann_model <- nnet(
  LoanApproval ~ Annual.Income + DTI + Credit.Score + Years.in.current.job 
  + Credit.Utilization + Home.Ownership + Purpose,
  data = train_data,
  size = 5,              
  decay = 0.01,          
  maxit = 200,           
  linout = FALSE         
)
ann_predictions <- predict(ann_model, newdata = test_data, type = "class")

ann_predictions <- factor(ann_predictions, 
levels = levels(test_data$LoanApproval))

# Confusion Matrix
ann_conf_matrix <- confusionMatrix(ann_predictions, 
test_data$LoanApproval)
print(ann_conf_matrix)

# Visualize 
plotnet(ann_model, circle_col = c("lightblue", "orange"))


# Based on confusion matrices, the evaluate_model function computes evaluation metrics (Accuracy, Precision, Recall, and F1 Score) for various classification models. Confusion matrices of Random Forest, Logistic Regression, Gradient Boosting, Decision Tree, and Artificial Neural Network models are all subjected to the function. The final table including these measurements is printed to assess and compare the model performance after the metrics are saved in a data frame for comparison.\

# Function evaluation metrics
evaluate_model <- function(conf_matrix) {
  accuracy <- conf_matrix$overall['Accuracy']
  precision <- conf_matrix$byClass['Pos Pred Value']
  recall <- conf_matrix$byClass['Sensitivity']
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  return(list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  ))
}


rf_conf_matrix <- confusionMatrix(factor(rf_predictions), factor(test_data$LoanApproval))
log_reg_conf_matrix <- confusionMatrix(log_reg_predictions, 
test_data$LoanApproval)
gb_conf_matrix <- confusionMatrix(gb_predictions, test_data$LoanApproval)
dt_conf_matrix <- confusionMatrix(factor(dt_predictions), factor(test_data$LoanApproval)) 
ann_conf_matrix <- confusionMatrix(ann_predictions, test_data$LoanApproval)

# Calculate 
rf_metrics <- evaluate_model(rf_conf_matrix)
log_reg_metrics <- evaluate_model(log_reg_conf_matrix)
gb_metrics <- evaluate_model(gb_conf_matrix)
dt_metrics <- evaluate_model(dt_conf_matrix) # Added Decision Tree
ann_metrics <- evaluate_model(ann_conf_matrix)

# Combine 
model_comparison <- data.frame(
  Model = c("Random Forest", "Logistic Regression", 
"Gradient Boosting", "Decision Tree", "Artificial Neural Network"),
  Accuracy = c(rf_metrics$Accuracy, log_reg_metrics$Accuracy, 
  gb_metrics$Accuracy, dt_metrics$Accuracy, ann_metrics$Accuracy),
  Precision = c(rf_metrics$Precision, log_reg_metrics$Precision, gb_metrics$Precision, dt_metrics$Precision, ann_metrics$Precision),
  Recall = c(rf_metrics$Recall, log_reg_metrics$Recall, gb_metrics$Recall, dt_metrics$Recall, ann_metrics$Recall),
  F1_Score = c(rf_metrics$F1_Score, log_reg_metrics$F1_Score, 
gb_metrics$F1_Score, dt_metrics$F1_Score, ann_metrics$F1_Score))


print(model_comparison)


# Using AUC (Area Under the Curve) from ROC (Receiver Operating Characteristic) analysis, the given code assesses many models (Random Forest, Logistic Regression, Gradient Boosting, Decision Tree, and Artificial Neural Network). The projected probability for each model's AUC values are computed and shown. The ROC curves for each model are plotted in various colors, with the baseline indicated by a diagonal line. The model names and associated AUC values are displayed in a legend that is appended to the plot. This makes it easier to compare the models' performances visually.\

suppressMessages({
  suppressWarnings({
    rf_probabilities <- predict(rf_model, test_data, type = "prob")[, 2]
    rf_roc <- roc(test_data$LoanApproval, rf_probabilities)
    rf_auc <- auc(rf_roc)

    log_reg_probabilities <- predict(log_reg_model, test_data, type = "response")
    log_reg_roc <- roc(test_data$LoanApproval, log_reg_probabilities)
    log_reg_auc <- auc(log_reg_roc)

    gb_probabilities <- predict(gb_model, test_data, n.trees = 100, type = "response")
    gb_roc <- roc(test_data$LoanApproval, gb_probabilities)
    gb_auc <- auc(gb_roc)

    dt_probabilities <- predict(dt_model, test_data, type = "prob")[, 2] 
    dt_roc <- roc(test_data$LoanApproval, dt_probabilities)
    dt_auc <- auc(dt_roc)

    ann_probabilities <- predict(ann_model, test_data, type = "raw")
    ann_roc <- roc(test_data$LoanApproval, ann_probabilities)
    ann_auc <- auc(ann_roc)

    # Print the AUC for each model
    cat("Random Forest AUC: ", rf_auc, "\n")
    cat("Logistic Regression AUC: ", log_reg_auc, "\n")
    cat("Gradient Boosting AUC: ", gb_auc, "\n")
    cat("Decision Tree AUC: ", dt_auc, "\n")
    cat("Artificial Neural Network AUC: ", ann_auc, "\n")
  })
})





plot(rf_roc, col = "blue", main = "ROC Curves for All Models", lwd = 2)
lines(log_reg_roc, col = "green", lwd = 2)
lines(gb_roc, col = "red", lwd = 2)
lines(dt_roc, col = "purple", lwd = 2)  
lines(ann_roc, col = "orange", lwd = 2)  


abline(a = 0, b = 1, col = "black", lty = 2)

legend(x = 0.5, y = 0.5,  
       legend = c(paste("Random Forest (AUC:", round(rf_auc, 3), ")"),
                  paste("Logistic Regression (AUC:", round(log_reg_auc, 3), ")"),
                  paste("Gradient Boosting (AUC:", round(gb_auc, 3), ")"),
                  paste("Decision Tree (AUC:", round(dt_auc, 3), ")"),
                  paste("ANN (AUC:", round(ann_auc, 3), ")")),  
       col = c("blue", "green", "red", "purple", "orange"),  
       lwd = 2,
       cex = 0.8)




