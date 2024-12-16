# Project Team Members
## Paul Desmond, Greg Ballard, Chavon Jackson

# Competition Page
https://www.kaggle.com/c/ieee-fraud-detection

# Main Outcome
The team delivered several high-performing models by combining optimization, rigorous evaluation, and interpretability. Best model scored 
## HyperOpt/LightGPM Model Results: Private Socre was .85, and Public Score was 0.885
## Optimized LightGPM Model Results: Private Socre was .93, and Public Score was 0.91
##
##
## Project Description
In this competition, we will build several classification models to predict whether or not an online financial transaction is fraudulent. This is a binary classification problem. Some important characteristics of this competition are mentioned below.

The target variable, isFraud, is integer-encoded so that a label value of 0 represents a non-fraudulent transaction, and a label value of 1 represents a fraudulent transaction.

The dataset contains 432 features, 49 of which are categorical and 383 of which are numerical. A list of the categorical features is provided on the data pageLinks to an external site. of the competition.
Some of the categorical features have many levels. For example, one categorical variable contains over 13,000 different values.
Some of the feature columns need values added.
The columns in the dataset are split across two files, which must be merged.
The dataset contains over 590,000 training observations and 510,000 test observations.
Submissions in this competition are scored using the Area Under the Curve (AUC) metric.
 
## Challenges
The DataFrames encountered in this project are large, with the merged but unprocessed training set taking up about 2.7 GB of memory. The many levels found in some of the categorical features presented a challenge. If you apply one-hot-encoding to these features, one column will be created for each level in the encoded array. This will cause the size of the dataset to explode, resulting in an array that is unlikely to fit into memory. Even if it does fit, our training algorithms will struggle when presented with a dataset containing tens of thousands of features. We must perform some exploratory data analysis (EDA) to determine which levels for each categorical variable are the most valuable for predicting the target variable. We will keep only these levels and will discard the rest.
The size of this dataset caused your cross-validation to require a significant amount of time to run. To address this concern, you can employ techniques mentioned in the page Grid Search Execution Time.

Developing this model presented various challenges across technical, conceptual, and collaborative domains, which required innovative solutions and continuous refinement of our approach. One of the primary hurdles involved addressing data quality issues, such as handling missing values and ensuring consistency between training and test datasets. These preprocessing steps were crucial to prevent biases or information loss that could negatively affect the model’s performance. Additionally, feature selection proved a significant challenge due to the many potential features. Identifying the most impactful ones without introducing redundancy or overfitting required iterative forward selection, which was computationally intensive and time-consuming.

Hyperparameter tuning added another layer of complexity, as finding the optimal combination of parameters, such as learning rate, maximum depth, and subsampling fractions, involved exploring a vast search space. Tools like Hyperopt helped streamline this process, but it remained resource-intensive. Model evaluation was also challenging, as we needed to balance achieving high AUC performance with preventing overfitting. Incorporating Stratified K-Fold Cross-Validation added robustness but significantly increased training and evaluation time. At the same time, ensuring the interpretability of the model was critical. While achieving high performance was a priority, we needed to explain the model’s predictions in an actionable way. This required integrating tools like SHAP to analyze feature importance and understand the relationships between features and predictions. Collaboration introduced challenges as team members worked across different schedules and time zones, making synchronous communication difficult. Managing contributions, code versions, and aligning objectives demanded a structured and flexible approach to communication. 

Computational resource limitations also posed difficulties, especially given the iterative nature of feature selection and hyperparameter tuning. Early stopping mitigated some resource constraints, but the sheer volume of experiments required careful planning and execution. Despite these challenges, we overcame them through careful planning, collaboration, and persistence. Robust preprocessing, thoughtful experimentation, and leveraging modern tools like LightGBM, Hyperopt, and SHAP enabled us to optimize the model effectively. Collaborative problem-solving and regular communication ensured that all team members contributed meaningfully, even when faced with logistical constraints. These obstacles strengthened our workflow and the final model, resulting in a high-performing and interpretable solution.
##
##
##

# Team Approach
The team adopted a hybrid collaboration model to address challenges such as differing time zones, personal responsibilities, and varying work commitments. By emphasizing flexibility, the team prioritized independent analysis, code creation, and regular peer reviews over synchronous collaboration. This approach balanced individual accountability with team alignment, enabling steady progress despite logistical constraints.

Team contributions spanned several critical areas. Research and exploration played a significant role as team members experimented with various models and techniques. Code development progressed through multiple iterations, incorporating enhancements and optimizations at each stage. Active peer reviews identified bugs, improved performance, and maintained code quality. Finally, the team collaborated to integrate standalone scripts into the final submission, ensuring the deliverables met all project requirements. This combination of individual effort and collaborative refinement resulted in a high-quality final product.

# Modelling Process
The team followed a highly iterative modeling process, creating and refining 20 to 30 versions for each model. These iterations were crucial for testing hypotheses, evaluating performance, and addressing challenges encountered during modeling. Each iteration involved experimenting with different preprocessing techniques, exploring various machine learning algorithms, and tuning hyperparameters to achieve optimal performance. Feedback from peer reviews was incorporated, further enhancing the code and analysis. This iterative workflow allowed the team to converge on robust solutions through continuous experimentation and learning, ensuring the final model was high-performing and reliable.
##
##
##
## MODEL LIGHTGPM and Hyperopt Model
The finalized model focused on training, evaluating, and analyzing a LightGBM-based solution for binary classification. The workflow emphasized robust performance optimization, thorough evaluation, and interpretability. The process began with data analysis and preprocessing, including checks for missing values, duplicates, and column consistency between datasets. Standardization and normalization were verified to maintain data consistency and compatibility for modeling. Hyperparameter optimization was performed using Hyperopt, with a focus on tuning key parameters such as learning_rate, num_leaves, and max_depth, ensuring a balance between overfitting and underfitting. The evaluation process utilized Stratified K-Fold Cross-Validation to provide reliable performance metrics while accounting for class imbalances across multiple data splits. AUC (Area Under the Curve) served as the primary metric for model evaluation, with visualizations like AUC distributions and scatter plots helping to analyze the relationship between hyperparameters and performance.
###
To address model interpretability, the team employed SHAP (SHapley Additive exPlanations) values to identify influential features driving the model’s predictions. Correlation heatmaps and feature importance plots provided additional insights into how features and hyperparameters impacted the results. Finally, the model was used to generate predictions on the test dataset, which were formatted into a submission-ready CSV file. This approach ensured that the results were both high-performing and competition-ready.
The IEEE-CIS Fraud Detection Kaggle competition posed numerous challenges, including handling data type inconsistencies, large datasets, and efficient hyperparameter tuning. Balancing interpretability with performance optimization and selecting appropriate evaluation metrics required significant problem-solving and analytical skills. Additionally, navigating collaborative dynamics and coordinating across time zones added another layer of complexity. Ultimately, by combining optimization, rigorous evaluation, and interpretability, the team delivered a well-documented and high-performing LightGBM model. This process reinforced the importance of adaptability, teamwork, and problem-solving in achieving success in machine learning projects.
## File Path to Code
https://github.com/gregofkickapoo/dsci_598_capstone/blob/main/LightGBM_w_HyperOpt_Pipeline/V5%20Edited%20Version%20v5.ipynb
## LIGHTGPM and Hyperopt Model Kaggle Submission Results
Private Socre was .85 and Public Score was 0.885
##
##
##
##
##
# Optimized LightGBM Framework Model
This script represents our team’s approach to building a robust binary classification model using LightGBM, focusing on exploratory data analysis (EDA), preprocessing, feature selection, and model optimization. We addressed common issues in raw datasets, such as handling missing values, encoding categorical variables into numerical formats, and normalizing features to ensure consistency and compatibility with the model. These preprocessing steps were critical to creating a clean and reliable dataset to maximize the model’s predictive power. We used LightGBM’s LGBMClassifier to train the model, which is well-suited for large datasets and structured data. The dataset was split into 75% training and 25% validation subsets, ensuring robust evaluation. We carefully tuned the model’s hyperparameters, including the number of estimators, maximum tree depth, learning rate, and subsampling fractions, to balance performance and efficiency. We implemented early stopping to prevent overfitting and save computational resources, which halts training if the validation performance does not improve after 100 rounds. The primary evaluation metric for the model was the AUC (Area Under the Curve), chosen for its effectiveness in binary classification tasks, as it measures the model’s ability to distinguish between classes. One of the key features of this script is the forward feature selection process we implemented. This iterative method tests each feature by temporarily adding it to the current set of selected features and evaluating its impact on the model’s performance. Features that improve the AUC score are retained, while those that do not are excluded. This dynamic process continues until no additional features enhance the score, ensuring that the final model uses only the most relevant features. This feature selection process automates much of the required manual effort, making the workflow more efficient and scalable. Once the best features were identified, we retrained the final LightGBM model on the selected feature set, focusing on maximizing the AUC score. The iterative logs and verbose output throughout the process provided valuable insights into which features contributed most to the model’s success, making the model both high-performing and interpretable.

This script reflects our team’s structured and methodical approach to machine learning. By combining thoughtful data preprocessing, automated feature selection, and robust model training techniques, we created a powerful and efficient model capable of handling complex datasets while providing clear insights into its decision-making process.
## File Path to Code
](https://github.com/gregofkickapoo/dsci_598_capstone/tree/main/Optimized_LightGBM_Model)
## LIGHTGPM and Hyperopt Model Kaggle Submission Results
Private Socre was .93, and Public Score was 0.91

