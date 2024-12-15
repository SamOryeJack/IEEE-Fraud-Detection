# dsci_598_capstone
This repository will be used for Maryville DSCI 598 Capstone Project - Kaggle IEEE Fraud Detection 

# Project Team Members
Chavon Jackson, Paul Desmond, Greg Ballard

# Competition Page
https://www.kaggle.com/c/ieee-fraud-detection

## Project Description
In this competition, you will build a classification model to predict whether or not an online financial transaction is fraudulent. This is a binary classification problem. Some important characteristics of this competition are mentioned below.

The target variable, isFraud, is integer-encoded so that a label value of 0 represents a non-fraudulent transaction and a label value of 1 represents a fraudulent transaction.

The dataset contains 432 features, 49 of which are categorical and 383 of which are numerical. A list of the categorical features is provided on the data pageLinks to an external site. of the competition.
Some of the categorical features contain very many levels. For example, one of the categorical variables contains over 13,000 different values.
Some of the feature columns contain missing values.
The columns in the dataset are split across two files which must be merged.
The dataset contains over 590,000 training observations and 510,000 test observations.
Submissions in this competition is scored using the Area Under the Curve (AUC) metric.
 
## Challenges
The DataFrames encountered in this project will be quite large. For example, the merged but unprocessed training set will take up about 2.7 GB of memory. Kaggle virtual machines provide 16GB of memory. We will need to be careful about memory management, only loading datasets as they are needed and deleting DataFrames from memory as they are no longer needed.
The large number of levels found in some of the categorical features will present a challenge. If you apply one-hot-encoding to these features, one column will be created for each level in the encoded array. This will cause the size of the dataset to explode, resulting in an array that is unlikely to fit into memory. Even it is does fit, our training algorithms will struggle when presented with a dataset containing tens of thousands of features. We will need to perform some exploratory data analysis (EDA) to determine which levels for each categorical variable are the most valuable for predicting the target variable. We will keep only these levels and will discard the rest.
The size of this dataset will likely cause your cross-validation to require a significant amount of time to run. You can employ techniques mentioned in the page Grid Search Execution Time to address this concern.
 
# Main Outcome
By combining optimization, rigorous evaluation, and interpretability, the team delivered several high-performing models. 


# Team Approach
The team adopted a hybrid collaboration model to navigate challenges such as differing time zones, personal responsibilities, and varying work commitments. Recognizing the need for flexibility, the team emphasized independent analysis, code creation, and regular peer reviews over synchronous collaboration. This approach balanced individual accountability with team alignment, ensuring steady progress despite logistical constraints.

## Key strategies included:
Focused Independent Work: Each member tackled specific tasks, allowing for efficient use of their time.
Regular Peer Reviews: Feedback loops helped refine the codebase and improve overall quality.
Adaptable Communication: Team members reached out proactively to provide support and stay aligned on goals.

## Team Contributions
Research and Exploration: Team members conducted in-depth research and experimentation with various models and techniques.
Code Development: Multiple iterations of the codebase were developed, incorporating enhancements and optimizations at each step.
Peer Reviews and Refinements: Active peer reviews helped identify bugs, improve performance, and ensure code quality.
Final Deliverables: The team worked collaboratively to integrate standalone scripts into the final submission, ensuring compliance with project requirements.
Highlights of Individual Contributions:

# Modelling Process
The team undertook a highly iterative process, creating and refining 20 to 30 versions for each model we worked on. These iterations were essential to test various hypotheses, evaluate performance, and address challenges that emerged during the modeling process. Each iteration involved:

Experimentation with different preprocessing techniques.
Exploration of various machine learning algorithms.
Tuning hyperparameters for optimal performance.
Incorporating feedback from peer reviews to improve code and analysis.
This iterative workflow allowed the team to converge on robust solutions through continual experimentation and learning.


## MODEL 1: LIGHTGPM and Hyperopt Model (Submital)
The finalized model focused on training, evaluating, and analyzing a LightGBM-based solution for binary classification. The workflow emphasized robust performance optimization, thorough evaluation, and interpretability. The process began with data analysis and preprocessing, including checks for missing values, duplicates, and column consistency between datasets. Standardization and normalization were verified to maintain data consistency and compatibility for modeling. Hyperparameter optimization was performed using Hyperopt, with a focus on tuning key parameters such as learning_rate, num_leaves, and max_depth, ensuring a balance between overfitting and underfitting. The evaluation process utilized Stratified K-Fold Cross-Validation to provide reliable performance metrics while accounting for class imbalances across multiple data splits. AUC (Area Under the Curve) served as the primary metric for model evaluation, with visualizations like AUC distributions and scatter plots helping to analyze the relationship between hyperparameters and performance.
To address model interpretability, the team employed SHAP (SHapley Additive exPlanations) values to identify influential features driving the model’s predictions. Correlation heatmaps and feature importance plots provided additional insights into how features and hyperparameters impacted the results. Finally, the model was used to generate predictions on the test dataset, which were formatted into a submission-ready CSV file. This approach ensured that the results were both high-performing and competition-ready.
The IEEE-CIS Fraud Detection Kaggle competition posed numerous challenges, including handling data type inconsistencies, large datasets, and efficient hyperparameter tuning. Balancing interpretability with performance optimization and selecting appropriate evaluation metrics required significant problem-solving and analytical skills. Additionally, navigating collaborative dynamics and coordinating across time zones added another layer of complexity. Ultimately, by combining optimization, rigorous evaluation, and interpretability, the team delivered a well-documented and high-performing LightGBM model. This process reinforced the importance of adaptability, teamwork, and problem-solving in achieving success in machine learning projects.
## File Path to Code
https://github.com/gregofkickapoo/dsci_598_capstone/blob/main/LightGBM_w_HyperOpt_Pipeline/V5%20Edited%20Version%20v5.ipynb
## Kaggle Submission Results for this model
SUBMITTAL WAS PRIVATE SCORE OF 0.85 and Public Score was .88
## LIGHTGPM and Hyperopt Model Kaggle Submission Results
Private Socre was .85 and Public Score was 0.885



