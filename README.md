# dsci_598_capstone
This repository will be used for Maryville DSCI 598 Capstone Project - Kaggle IEEE Fraud Detection 

# Project Team Members
Chavon Jackson, Paul Desmond, Greg Ballard

# Competition Page
https://www.kaggle.com/c/ieee-fraud-detection

# Main Outcome
By combining optimization, rigorous evaluation, and interpretability, the team delivered several high-performing models. Best model scored 
## HyperOpt/LightGPM Model Results: Private Socre was .85 and Public Score was 0.885
## Optimized LightGPM Model Results: Private Socre was .93 and Public Score was 0.91


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
The large number of levels found in some of the categorical features presented a challenge. If you apply one-hot-encoding to these features, one column will be created for each level in the encoded array. This will cause the size of the dataset to explode, resulting in an array that is unlikely to fit into memory. Even it is does fit, our training algorithms will struggle when presented with a dataset containing tens of thousands of features. We will need to perform some exploratory data analysis (EDA) to determine which levels for each categorical variable are the most valuable for predicting the target variable. We will keep only these levels and will discard the rest.
The size of this dataset caused your cross-validation to require a significant amount of time to run. You can employ techniques mentioned in the page Grid Search Execution Time to address this concern.

Developing this model presented a variety of challenges across technical, conceptual, and collaborative domains, which required innovative solutions and continuous refinement of our approach. One of the primary hurdles involved addressing data quality issues, such as handling missing values and ensuring consistency between training and test datasets. These preprocessing steps were crucial to prevent biases or information loss that could negatively affect the model’s performance. Additionally, feature selection proved to be a significant challenge due to the large number of potential features. Identifying the most impactful ones without introducing redundancy or overfitting required iterative forward selection, which was both computationally intensive and time-consuming.

Hyperparameter tuning added another layer of complexity, as finding the optimal combination of parameters, such as learning rate, maximum depth, and subsampling fractions, involved exploring a vast search space. Tools like Hyperopt helped streamline this process, but it remained resource-intensive. Model evaluation was also challenging, as we needed to balance achieving high AUC performance with preventing overfitting. Incorporating Stratified K-Fold Cross-Validation added robustness but significantly increased training and evaluation time. At the same time, ensuring the interpretability of the model was critical. While achieving high performance was a priority, we needed to explain the model’s predictions in an actionable way. This required integrating tools like SHAP to analyze feature importance and understand the relationships between features and predictions.

Collaboration introduced its own set of challenges, as team members worked across different schedules and time zones, making synchronous communication difficult. Managing contributions, code versions, and aligning objectives demanded a structured and flexible approach to communication. Computational resource limitations also posed difficulties, especially given the iterative nature of feature selection and hyperparameter tuning. Early stopping mitigated some resource constraints, but the sheer volume of experiments required careful planning and execution.

Despite these challenges, we overcame them through careful planning, collaboration, and persistence. Robust preprocessing, thoughtful experimentation, and leveraging modern tools like LightGBM, Hyperopt, and SHAP enabled us to optimize the model effectively. Collaborative problem-solving and regular communication ensured that all team members contributed meaningfully, even when faced with logistical constraints. These obstacles ultimately strengthened both our workflow and the final model, resulting in a high-performing and interpretable solution.

 


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


## MODEL LIGHTGPM and Hyperopt Model (Submital)
The finalized model focused on training, evaluating, and analyzing a LightGBM-based solution for binary classification. The workflow emphasized robust performance optimization, thorough evaluation, and interpretability. The process began with data analysis and preprocessing, including checks for missing values, duplicates, and column consistency between datasets. Standardization and normalization were verified to maintain data consistency and compatibility for modeling. Hyperparameter optimization was performed using Hyperopt, with a focus on tuning key parameters such as learning_rate, num_leaves, and max_depth, ensuring a balance between overfitting and underfitting. The evaluation process utilized Stratified K-Fold Cross-Validation to provide reliable performance metrics while accounting for class imbalances across multiple data splits. AUC (Area Under the Curve) served as the primary metric for model evaluation, with visualizations like AUC distributions and scatter plots helping to analyze the relationship between hyperparameters and performance.
To address model interpretability, the team employed SHAP (SHapley Additive exPlanations) values to identify influential features driving the model’s predictions. Correlation heatmaps and feature importance plots provided additional insights into how features and hyperparameters impacted the results. Finally, the model was used to generate predictions on the test dataset, which were formatted into a submission-ready CSV file. This approach ensured that the results were both high-performing and competition-ready.
The IEEE-CIS Fraud Detection Kaggle competition posed numerous challenges, including handling data type inconsistencies, large datasets, and efficient hyperparameter tuning. Balancing interpretability with performance optimization and selecting appropriate evaluation metrics required significant problem-solving and analytical skills. Additionally, navigating collaborative dynamics and coordinating across time zones added another layer of complexity. Ultimately, by combining optimization, rigorous evaluation, and interpretability, the team delivered a well-documented and high-performing LightGBM model. This process reinforced the importance of adaptability, teamwork, and problem-solving in achieving success in machine learning projects.
## File Path to Code
https://github.com/gregofkickapoo/dsci_598_capstone/blob/main/LightGBM_w_HyperOpt_Pipeline/V5%20Edited%20Version%20v5.ipynb
## LIGHTGPM and Hyperopt Model Kaggle Submission Results
Private Socre was .85 and Public Score was 0.885


# Optimized LightGBM Framework Model
This script represents our team’s approach to building a robust binary classification model using LightGBM, with a focus on exploratory data analysis (EDA), preprocessing, feature selection, and model optimization. We began by addressing common issues in raw datasets, such as handling missing values, encoding categorical variables into numerical formats, and normalizing features to ensure consistency and compatibility with the model. These preprocessing steps were critical to creating a clean and reliable dataset that would maximize the model’s predictive power. To train the model, we used LightGBM’s LGBMClassifier, which is well-suited for large datasets and structured data. The dataset was split into 75% training and 25% validation subsets, ensuring robust evaluation throughout the process. We carefully tuned the model’s hyperparameters, including the number of estimators, maximum tree depth, learning rate, and subsampling fractions, to balance performance and efficiency. To prevent overfitting and save computational resources, we implemented early stopping, which halts training if the validation performance does not improve after 100 rounds. The primary evaluation metric for the model was the AUC (Area Under the Curve), chosen for its effectiveness in binary classification tasks, as it measures the model’s ability to distinguish between classes. One of the key features of this script is the forward feature selection process we implemented. This iterative method tests each feature by temporarily adding it to the current set of selected features and evaluating its impact on the model’s performance. Features that improve the AUC score are retained, while those that do not are excluded. This dynamic process continues until no additional features enhance the score, ensuring that the final model uses only the most relevant features. This feature selection process automated much of the manual effort typically required, making the workflow more efficient and scalable. Once the best features were identified, we retrained the final LightGBM model on the selected feature set, focusing on maximizing the AUC score. The iterative logs and verbose output throughout the process provided valuable insights into which features contributed most to the model’s success, making the model both high-performing and interpretable.

Overall, this script reflects our team’s structured and methodical approach to machine learning. By combining thoughtful data preprocessing, automated feature selection, and robust model training techniques, we created a model that is both powerful and efficient, capable of handling complex datasets while providing clear insights into its decision-making process.
## File Path to Code
https://github.com/gregofkickapoo/dsci_598_capstone/blob/main/LightGBM_w_HyperOpt_Pipeline/V5%20Edited%20Version%20v5.ipynb
## LIGHTGPM and Hyperopt Model Kaggle Submission Results
Private Socre was .93 and Public Score was 0.91

