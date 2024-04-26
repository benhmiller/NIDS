# Experiment 1 - Decision Tree on Binary Classification

Complete with command to run on test set: `./642test-$(uname -p) <MODEL_FILE_NAME> <THRESHOLD>`

### Q1: What hyperparameters did you vary, and what values did you try? What were some trends you observed?

Within the first experiment concerning the binary classification performed by a decision tree, I opted to set the random_state to 42 for the sake of consistency across multiple runs, then performed cross validation testing while varying the max_depth hyperparameter. In order to ascertain a better unerstanding of how the depth affected the model, I plotted cross-validation accuracy vs. the varying values of maximum tree depth and observed the relationship. Ultimately, after noticing a steep increase in accuracy around depths 5-7 and a plateau of accuracy around depths 13-14, I opted to test a range of depths from 10-15. Though accuracy marginally increased in the range 20-30 (often peaking around 24-26), I felt that limiting the depth to this shorter range would better avoid overfitting and lead to a more accurate model when testing on unseen data.

### Q2: Provide an image of the trained decision tree. Discuss what you observe.

![Experiment 1: Decision Tree](../figures/exp1/decision_tree.png)

### Q3: Provide a plot of the feature importance. What were some of the most important features? Why do you think these features were important?

![Experiment 1: Feature Importances](../figures/exp1/feature_importances.png)

### Q4: Plot a confusion matrix on the validation data. Discuss what you observe.

![Experiment 1: Confusion Matrix](../figures/exp1/confusion_matrix.png)

### Q5: Plot a ROC curve on the validation data. What do you find to be the optimal threshold for predicting something as positive?

![Experiment 1: ROC Curve](../figures/exp1/ROC_curve.png)

---
---

# Experiment 2 - Decision Tree on Multiclass Classification

Complete with command to run on test set: `./642test-$(uname -p) <MODEL_FILE_NAME> <THRESHOLD>`

### Q1: What hyperparameters did you vary, and what values did you try? What were some trends you observed?

### Q2: Provide an image of the trained decision tree. Discuss what you observe.

![Experiment 2: Decision Tree](../figures/exp2/decision_tree.png)

### Q3: Provide a plot of the feature importance. What were some of the most important features? Why do you think these features were important?

![Experiment 2: Feature Importances](../figures/exp2/feature_importances.png)

### Q4: Plot a confusion matrix on the validation data. Discuss what you observe.

![Experiment 2: Confusion Matrix](../figures/exp2/confusion_matrix.png)

---
---

# Experiment 3 - Neural Network on Binary Classification

Complete with command to run on test set: `./642test-$(uname -p) <MODEL_FILE_NAME> <THRESHOLD>` 

### Q1: What hyperparameters did you vary, and what values did you try? What were some trends you observed?

### Q2: Provide a plot of the loss curve over the epochs trained. What do you observe as the number of epochs increases?

![placeholder image](placeholder.png)

### Q3: Plot a confusion matrix on the validation data. Discuss what you observe.

![placeholder image](placeholder.png)

### Q4: Plot a ROC curve on the validation data. What do you find to be the optimal threshold for predicting something as positive?

![placeholder image](placeholder.png)

---
---


# Experiment 4 - Neural Network on Multiclass Classification

Complete with command to run on test set: `./642test-$(uname -p) <MODEL_FILE_NAME> <THRESHOLD>` 

### Q1: What hyperparameters did you vary, and what values did you try? What were some trends you observed?

### Q2: Provide a plot of the loss curve over the epochs trained. What do you observe as the number of epochs increases?

![placeholder image](placeholder.png)

### Q3: Plot a confusion matrix on the validation data. Discuss what you observe.

![placeholder image](placeholder.png)

