# General Questions

### Q1: What are some advantages and disadvantages of using a supervised machine learning classifier for intrusion detection compared to a signature-based (non-learning) classifier?


When utilizing a supervised machine learning classifier for intrusion detection compared to a signature-based (non-learning) classifier, one may note numerous advantages, such as:
- By learning from labeled data, a ML classifier does not need manually defined rules to identify threats as it instead relies on data extraction and pattern recognition to classify activity as normal or malicious. Further, such adaptive classification increases the ability of the model to detect novel threats and flag unseen attacks, a quality lacked by a signature-based classifier.

Disadvantages, however, do exist, such as:
- For a supervised ML classifier to function optimally, the quality and quantity of data must prove sufficient, which may become a challenge
- Models may prove subject to overfitting which could negatively affect their performance when classifying unseen data.

### Q2: What is the importance of using a validation set?

When attempting to a train a model for classification tasks, utilizing a validation set enables one to compare the performance among models of the same type, but built with different hyperparameters. Consequently, one may determine how hyperparameter settings affect accuracy and thereby select those which achieve optimal performance. Finally, the use of a validation set enables one to gauge the performance of the final selected model by enabling the user to reserve data in an unseen (test) data set for the model to classify following the training/tuning phase.

### Q3: Assuming a binary classification setting, describe what false negative rate and false positive rate represents in the context of this dataset, and how they impact a system in which the NIDS is deployed.

In the context of this dataset:

- False Negative Rate: The proportion of malicious samples classified as benign (conceptually, a measurement of the system's failure to identify malicious attacks)
- False Positive Rate: The proportion of benign samples classified as malicious (conceptually, a measurement of the system's "false alarms")

Assessing the impact on the NIDS deployment:
- When evaluating the effect of these metrics on the system in which the NIDS is deployed, a high false negative rate would indicate high system vulnerability to attack. It implies the NIDS fails to identify genuine threats, potentially resulting in increased security breaches and compromises.
- In contrast, a high false positive rate would indicate high resource waste within the system due to significant examinations of flagged activitiy that ultimately proves benign.

### Q4: What differences did you observe between the decision tree models and the neural networks? What are some pros and cons of using each model from a security perspective?

Replace this text with your answer (should be between 2-4 sentences).

### Q5: Between decision trees and neural networks, which model would you recommend using in security sensitive domains and why? There are no wrong answers here, but answer should be supported with sufficient justification.

Replace this text with your answer (should be between 2-4 sentences).

### Q6: What did you notice when switching from the binary classification to multiclass classification setting? How does that impact security?

Replace this text with your answer (should be between 2-4 sentences).

### Q7: What information does an NIDS (and therefore the system admins) need to access to be effective, and how does that relate to the privacy of users accessing the network?

Replace this text with your answer (should be between 2-4 sentences).