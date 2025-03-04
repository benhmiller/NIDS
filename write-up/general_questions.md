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

When evaluating the decision tree and neural network models, one of the most noticeable differences was the inability to understand how exactly the NN found patterns to identify threats. While the decision tree offers the option to examine each node to understand the filtering of traffic, the neural network fails to provide such functionality. Alas, in real-world settings, complex data relationships may struggle to be mapped by an efficient decision tree, meaning a neural network, which excels in this task, could prove far more favorable for the job.

### Q5: Between decision trees and neural networks, which model would you recommend using in security sensitive domains and why? There are no wrong answers here, but answer should be supported with sufficient justification.

Personally, I feel that a decision tree model would prove more effective in security sensitive domains as, if the system generates an alert, an engineer may  evaluate the decision tree and manually understand how the system arrived at its conclusion. Of course, if the security personel preferred to implement a more adaptable model to handle novel and evolving threats, then a neural network may be recommended—though, notably, at the cost of interpretability and human understandability.

### Q6: What did you notice when switching from the binary classification to multiclass classification setting? How does that impact security?

When switching from the binary classification to multiclass classification setting, I noticed that the performance of the NIDS became much harder to gauge. In particular, when attempting to classify traffic among a variety of potential categories, the system could perform well at separating certain types of traffic, but struggle with others. For example, in my experience, both the neural network and decision tree classifiers struggled to identify class 3 traffic (R2L traffic), despite performing well in other areas. For real-world security systems, this could suggest that even a well-tuned model for particular types of threats may fail for certain classes, leaving the network vulnerable and open to attack.

### Q7: What information does an NIDS (and therefore the system admins) need to access to be effective, and how does that relate to the privacy of users accessing the network?

In order to prove effective, a network intrusion detection system needs access to: network traffic data such as source and destination IP addresses, ports, and other network insights; databases of known benign and malicious activity to train and interpret patterns for future classification; and logging and reporting tools to record events and generate alerts


 With regard to the privacy of users accessing the network, users should be required to consent to data traffic collection (i.e. browsing activity, application usage, user communcations, etc.) in order to ensure transparency and allow the NIDS to verify their actions are benign. Consequently, design of the NIDS should ensure only authorized personnel are capable of accessing the data utilized by the system. Finally, to protect user confidentiality, designers may need to implement data anonymization or encryption techniques that prevent anything but the system from viewing information about users. 