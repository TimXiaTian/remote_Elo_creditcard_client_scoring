# Elo_creditcard_client_scoring

### 1. Project Overview

This competition is a collaboration between Elo, one of the largest payment brands in Brazil, and Kaggle. Both the prize and the data are provided by Elo. Unlike online payment platforms such as Alipay and PayPal, Elo focuses on credit cards as the core of its financial products. In Brazil, online payments are predominantly managed by local banks, with credit cards being the primary means for online transactions. Established in 2011 through a joint venture by three major Brazilian banks, Elo is the largest local online payment brand in Brazil, mainly responsible for online payment services.

In addition to being a payment gateway, Elo operates as an "O2O" (online-to-offline) platform. Through its app, users can browse various local services such as dining, lodging, movies, and travel, and make payments online using their credit cards. This model is similar to the "Pocket Life" app used for China Merchants Bank credit cards.

Elo's core objective in utilizing machine learning algorithms is to better recommend local dining, entertainment, and travel services to users within the app. This includes showcasing popular restaurants and notifying users of discounts and offers. The goal of this competition is to predict the loyalty score for each user. Essentially, this involves predicting a score for each user, making it a regression problem at its core. 

### 2. Dataset Overview

There are seven data files in total, which can be categorized into three groups:

1. **Basic Information Datasets**: 
   - `Data_Dictionary`: A data dictionary containing definitions for all fields in the dataset.
   - `sample_submission`: A sample submission file illustrating the format for competition submissions.

2. **Training and Testing Datasets**: 
   - `train`: The training dataset used to build the model.
   - `test`: The testing dataset used to evaluate the model. Both datasets have consistent features.

3. **Supplementary Datasets**: 
   - `historical_transactions.csv`: Records of historical transactions made with credit cards.
   - `new_merchant_transactions.csv`: Records of new transactions with merchants.
   - `merchants.csv`: Additional information about merchants present in the historical and new transaction datasets. Including these supplementary datasets can help in uncovering patterns and improving the model's accuracy during the modeling process.

### 3. Exploratory Data Analysis

#### 3.1 Data Loading and Memory Management

First, load the data. Given the large scale of the datasets, import the `gc` package for memory management. Use `del` to delete objects and `gc.collect()` to manually clear memory when needed.

#### 3.2 Initial Exploration

Perform a simple exploratory data analysis (EDA) to check data quality. Validate data correctness and check for missing values and outliers.

- **Data Correctness Validation**:

  Ensure the data adheres to basic logic. For example, verify that credit card IDs are unique and there are no duplicate IDs between the training and test datasets.

- **Checking for Missing Values**:

  Identify and handle any missing values in the datasets.

- **Outlier Detection**:

  Identify outliers using methods like probability density histograms or the 3σ rule. Columns with many outliers may indicate special user categories and should be analyzed separately during feature extraction and modeling.

#### 3.3 Consistency Analysis

Consistency analysis compares the distribution of features between the training and test datasets to ensure they come from the same overall population. This helps confirm that the same underlying patterns affect both datasets. 

- **Univariate Analysis**:

  Compare the distribution of individual variables using relative frequency distributions (probability distributions).

- **Multivariate Joint Distribution**:

  Examine the joint distribution of multiple variables. For example, if feature 1 has two levels (0/1) and feature 2 has two levels (A/B), the joint distribution will include four levels (0A, 0B, 1A, 1B).

- **Practical Importance of Consistency Analysis**:

  Consistency analysis guides subsequent modeling:

  - If distributions are consistent, features come from the same population, and model performance potential is high. Focus on feature engineering and modeling techniques.
  - If distributions are inconsistent, the model performance potential is limited, and overfitting is likely. Use cross-validation to prevent overfitting and employ additional techniques.

#### 3.4 Data Preprocessing

Preprocessing includes essential steps that do not affect subsequent feature engineering, modeling, or multi-table associations.

##### 3.4.1 Field Type Annotation and Conversion

1. **Labeling Continuous and Discrete Variables**:

   Different field types influence feature transformation methods.

2. **Object Type Field Conversion**:

3. **Text/Sequence Field Processing**:

##### 3.4.2 Correctness Validation

1. **Uniqueness of ID**:

   Remove duplicate IDs.

2. **Consistency with Data Dictionary**:

##### 3.4.3 Feature Transformation

1. **Continuous Variables**:

   - Normalization/Standardization
   - Discretization (Binning)

2. **Discrete Variables**:

   - One-Hot Encoding
   - Ordinal Encoding (Dictionary Order Encoding)

   Discrete variables can be encoded numerically based on sorted order. Variable types include continuous, nominal (categorical without order), and ordinal (categorical with order). Ordinal variables can be one-hot encoded if order information is not needed.

##### 3.4.4 Missing Value Handling

1. **Identification**:
   - Detect NaN and None values
   - Identify missing values based on specific symbols

2. **Handling**:
   - Deletion
   - Imputation
   - Marking with special symbols

##### 3.4.5 Outlier Analysis

1. **Identification**:

   - 3-Sigma Rule
   - Box Plot Method

2. **Handling**:

   - Capping Method

     Cap infinite values at a maximum explicit value.

   - Marking with Special Symbols

#### 3.5 Merging Tables and Exporting

Merge the necessary tables and export the resulting dataset for further analysis.

### 4. Feature Engineering

Feature engineering is crucial for improving model performance. For cleaned data, the focus is on feature creation (derivation) and feature selection. This involves creating new features that could positively impact the model and then selecting the best ones to ensure model stability and efficiency.

#### 4.1 General Combined Feature Creation

General combined features are created by aggregating different discrete features with continuous feature values and summing them by card_id. This method captures the dataset's information from multiple dimensions and allows smooth concatenation with the training and test datasets for modeling.

#### 4.2 Business Statistical Feature Creation

Group by id and calculate various statistical measures for different fields within each group. These statistics are used as features in the modeling process. This method, using pandas' groupby function, results in features with fewer missing values and fewer additional columns.

#### 4.3 NLP Feature Derivation

The dataset contains many ID-related columns (e.g., card_id, merchant_id, merchant_category_id, state_id, subsector_id, city_id) that are closely related to users' transaction behaviors. We can use NLP methods like CountVectorizer and TF-IDF to derive new features:

1. **Processing Specific Features**:
   - Define `nlp_features` (e.g., merchant_id, merchant_category_id, state_id, subsector_id, city_id).
   - Convert these features to string type for text processing.

2. **Generating Feature Combinations**:
   - Divide features based on month_lag into new, historical, and all transaction data.
   - Group by card_id and concatenate results into strings, creating new feature columns.

3. **Instantiate CountVectorizer and TfidfVectorizer**:
   - Use CountVectorizer to convert text data into sparse matrices, representing word frequencies.
   - Use TfidfVectorizer to convert text data into sparse matrices, representing TF-IDF values.

4. **Processing Feature Text**:
   - Define a list of all newly generated features (e.g., merchant_id_new, merchant_id_hist, merchant_id_all).
   - Use CountVectorizer and TfidfVectorizer to transform each feature into sparse matrices, adding results to train_x and test_x.

#### 4.4 User Behavior Features

Each credit card transaction record includes a transaction time. We need to derive additional features based on transaction times to describe user behavior patterns, such as:

- Time difference between the most recent and first transaction.
- Time difference between the credit card activation date and the first transaction.
- Average time interval between transactions.
- Aggregating transactions by location/product category and calculating statistics (e.g., mean, variance).

1. **Calculate Time-related Features (Historical Transactions)**:
   - Calculate time-related features, such as the time difference between the first transaction and activation date.

2. **Calculate Time-related Features (New Transactions)**:
   - Similarly, calculate time-related features for new transactions.

3. **Calculate Unique Value and Ratio Features**:
   - Calculate unique values and the ratio of transaction counts to unique values.

4. **Calculate Statistics and Quantiles**:
   - Calculate statistics and quantiles for purchase amounts and derived features.

5. **Generate Pivot Features**:
   - Create pivot table features, including transaction counts and unique purchase dates.

6. **Handle Pre-activation Transactions (New Transactions)**:
   - Calculate transaction records and features for transactions before card activation.

Focus on user behavior features from the last two months for more recent and valuable data.

1. **Filter Data**:
   - Filter transactions where month_lag >= -2 (recent two months).

2. **Calculate Month Difference Feature**:
   - Calculate month difference from purchase_date to the current date, adding month_lag.

3. **Generate Traditional Features**:
   - Calculate transaction counts, average authorization flags, and sums for each card_id.

4. **Generate Unique Value and Ratio Features**:
   - Calculate unique values and the ratio of transaction counts to unique values.

5. **Generate Statistical Features**:
   - Calculate various statistics for purchase amounts, including sum, mean, standard deviation, and median.

6. **Generate Pivot Features**:
   - Create pivot table features, including transaction counts and unique purchase dates.

#### 4.5 Second-order Cross Features

Construct second-order features, such as summing transaction amounts across different product combinations. Higher-order features can make the feature matrix sparser and create many features, potentially leading to dimensional explosion. Focus on user behavior data for second-order cross features.

1. **Calculate Unique Values for Second-order Features**:
   - Define feature combinations to calculate unique values for each entity.
   - Perform statistics (e.g., mean, max, std) on these unique values.

2. **Calculate Counts for Second-order Features**:
   - Define features for calculating counts.
   - Perform statistics (e.g., mean, max, std) on these counts.

3. **Calculate Sums, Means, and Standard Deviations for Second-order Features**:
   - Define feature combinations to calculate sums for each entity.
   - Perform statistics (e.g., mean, max, std) on these sums.

#### 4.6 Anomaly Detection Features

The training dataset labels contain extreme outliers, which may be special markers with important information. Perform two-stage modeling:

1. **First Stage**: Use a classification model to identify anomalous users and split the dataset into normal and anomalous user datasets.
2. **Second Stage**: Perform regression modeling on the two datasets separately.

Create aggregated fields based on anomalous users, such as average transaction counts and amounts.

By systematically applying these feature engineering steps, we can better capture the underlying patterns and relationships in the data, leading to improved model performance.



### 5. Model Training

#### 5.1 Feature Selection

##### 5.1.1 Filter Methods

- **Pearson Correlation Coefficient**:
  Measures the linear relationship between two variables, ranging from -1 to 1. Calculate the correlation between each feature and the target variable to select features with high correlation.

- **Maximal Information Coefficient (MIC)**:
  Measures any type of relationship (linear or non-linear) between two variables, ranging from 0 to 1. Helps identify features that have any form of association with the target variable. However, MIC is computationally intensive, especially on large datasets.

##### 5.1.2 Wrapper Methods

Use models to select effective features. For example, a random forest model can output feature importance scores. Train a random forest model quickly, then use these scores to select important features for hyperparameter optimization and cross-validation. Wrapper methods often yield more effective features than correlation coefficients because they measure the actual importance of features for the target.

##### 5.1.3 Manual Selection

1. Define multiple conditions for selecting columns to delete.
2. Traverse column names and add unwanted columns to the delete list based on conditions.
3. Select the final features to retain based on comprehensive conditions.
4. Delete unnecessary columns and save the processed data.

#### 5.2 Hyperparameter Tuning

##### 5.2.1 Grid Search

To improve search efficiency, use RandomizedSearchCV to determine the approximate range first, then use GridSearchCV for precise parameter tuning.

1. Enumerate all parameter combinations.
2. Perform cross-validation for each combination:
   - For n combinations, train n models, each with 5-fold cross-validation, resulting in 5 cross-validation scores per model.
3. Select and train the best model:
   - Calculate the mean cross-validation score for each model and select the model with the highest mean score. Use its parameters as the best parameters for final training on the entire training set.

##### 5.2.2 Bayesian Optimization

Bayesian optimization uses prior calculations to iteratively adjust the search process, making it faster and more efficient than grid search. Hyperopt, a type of Bayesian optimizer, supports searching both continuous and discrete variables. Supported algorithms include random search, simulated annealing, and Tree of Parzen Estimators (TPE).

##### 5.2.3 Parameter Callback

For some models, not all hyperparameters need to be searched. To prevent some hyperparameters from being set to default values during multiple model instantiations, create a parameter callback function to repeatedly specify these fixed parameters.

#### 5.3 Model Training

##### 5.3.1 LightGBM

LightGBM, developed by Microsoft, is an implementation of Gradient Boosting Decision Trees (GBDT). It supports efficient parallel training, faster training speed, lower memory consumption, higher accuracy, and distributed processing for large-scale data.

1. **Data Preprocessing**:
   - Remove unnecessary columns (e.g., id and target) from the training dataset, retaining only features for training.

2. **Wrap Training Data**:
   - Use LightGBM's Dataset class to wrap the training data.

3. **Define Hyperparameter Optimization Objective Function**:
   - Define an internal function to input a set of hyperparameters and output the corresponding loss value (RMSE). This function uses LightGBM's cross-validation (cv) to calculate the minimum RMSE for the given hyperparameters.

4. **Define Parameter Space**:
   - Use Hyperopt's hp module to define the parameter space, including learning_rate, bagging_fraction, feature_fraction, num_leaves, reg_alpha, reg_lambda, bagging_freq, and min_child_samples.

5. **Perform Hyperparameter Optimization**:
   - Use Hyperopt's fmin function to search for hyperparameters. The search algorithm uses tpe.suggest (TPE: Tree of Parzen Estimators) and sets the maximum evaluation count to 30.

##### 5.3.2 XGBoost

XGBoost (Extreme Gradient Boosting) is another implementation of the GBDT algorithm. Unlike LightGBM, which supports only L2 regularization, XGBoost supports both L1 and L2 regularization, helping prevent overfitting.

##### 5.3.3 CatBoost

CatBoost, developed by Russian search engine Yandex and open-sourced in July 2017, has gained popularity for its powerful performance and fast execution. One of CatBoost's standout features is its ability to handle categorical features using a mix of one-hot and mean encoding strategies, effectively integrating feature engineering into the model training process. Additionally, CatBoost introduces a novel gradient boosting mechanism that balances empirical risk and structural risk, enhancing accuracy while preventing overfitting.

By systematically applying these model training steps, we can better capture the underlying patterns and relationships in the data, leading to improved model performance.

### 6. Model Ensemble

Common strategies for model ensemble include Voting and Stacking. The purpose of model ensemble is to leverage the strengths of different models to produce more reliable results. In Voting, we aggregate the predictions of different models on the test set by averaging or weighting them. Stacking is more complex and involves training a new model on the predictions of previously trained models.

#### 6.1 Voting Ensemble

Voting ensemble can be divided into three types: mean ensemble, weighted ensemble, and trick ensemble.

##### 6.1.1 Mean Ensemble

In mean ensemble, we take the average of the predictions from multiple models.

##### 6.1.2 Weighted Ensemble

In weighted ensemble, we assign different weights to the predictions of different models based on their performance. For example, if models A and B have scores of 2 and 3 respectively (assuming lower scores are better), we assign weights of 3/5 to model A and 2/5 to model B in the final prediction.

##### 6.1.3 Trick Ensemble

Trick ensemble involves assigning weights based on specific rules or heuristics to achieve better performance.

#### 6.2 Stacking Ensemble

Stacking involves using the out-of-fold (OOF) predictions of base models to train a new meta-model, which makes the final predictions on the test set.

1. **Data Preparation**:
   - **oof_1, oof_2, oof_3**: OOF predictions from the training set for each base model.
   - **predictions_1, predictions_2, predictions_3**: Predictions on the test set from each base model.
   - **y**: Target variable for the training set.
   - **eval_type**: Evaluation type, either 'regression' or 'binary'.

2. **Cross-Validation Strategy**:
   - Use RepeatedKFold for multiple rounds of cross-validation to enhance model stability.

3. **Model Training and Prediction**:
   - For each fold:
     - Split the training set into training and validation data.
     - Train a Bayesian Ridge regression model on the training data.
     - Predict on the validation data and record the results in the `oof` array.
     - Predict on the test set and accumulate the predictions.

4. **Model Evaluation**:
   - Calculate and print the evaluation score (RMSE or Logloss) for the model on the validation data based on the evaluation type.

By combining different models through these ensemble techniques, we can enhance the robustness and accuracy of our predictions.

### 7. Two-Stage Modeling

1. In the first stage, use a classification model to identify whether a user is an anomalous user, dividing the dataset into normal and anomalous user subsets.
2. In the second stage, perform regression modeling on the two subsets separately.

During the two-stage modeling process, cross-validation and model ensemble should be applied at each stage to maximize model performance. This involves training three sets of models (and corresponding ensemble processes) to handle the classification task, the regression task for normal users, and the regression task for anomalous users.

#### 7.1 Model Training

To efficiently call three different models and perform classification and regression predictions, we define a function to handle all the model training processes.

1. **Function Signature and Initialization**:
   The function parameters can be set as:
   - **X**: Feature matrix of the training dataset.
   - **X_test**: Feature matrix of the test dataset.
   - **y**: Target variable of the training dataset.
   - **params**: Hyperparameters of the model.
   - **folds**: Cross-validation strategy (e.g., KFold).
   - **model_type**: Type of model, can be 'lgb' (LightGBM), 'xgb' (XGBoost), or 'cat' (CatBoost).
   - **eval_type**: Evaluation type, can be 'regression' or 'binary'.
   - **oof**: Array to store out-of-fold predictions.
   - **predictions**: Array to store test set predictions.
   - **scores**: List to store evaluation scores for each fold.

2. **Cross-Validation Training**:
   - Use the cross-validation strategy to split the data into training and validation sets.
   - `trn_idx` and `val_idx` represent the indices of the training and validation sets in the current fold.

3. **Training and Prediction Based on Model Type**:
   - **LightGBM**:
     - Create LightGBM training and validation datasets.
     - Use the `lgb.train` method to train the model, setting early stopping rounds and validation sets.
     - Store the current fold's predictions in the `oof` array and accumulate predictions on the test set.
   - **XGBoost**:
     - Create XGBoost training and validation datasets.
     - Use the `xgb.train` method to train the model, setting early stopping rounds and validation sets.
     - Store the current fold's predictions in the `oof` array and accumulate predictions on the test set.
   - **CatBoost Regression**:
     - Create and train a CatBoost regression model.
     - Set the evaluation metric to RMSE and use early stopping.
     - Store the current fold's predictions in the `oof` array and accumulate predictions on the test set.
   - **CatBoost Binary Classification**:
     - Create and train a CatBoost binary classification model.
     - Set the evaluation metric to Logloss and use early stopping.
     - Store the current fold's predictions in the `oof` array and accumulate predictions on the test set.

4. **Model Performance Evaluation**:
   - Calculate the evaluation score for the current fold based on the evaluation type and store it in the `scores` list.

5. **Output Cross-Validation Results**:
   - Output the mean and standard deviation of the cross-validation scores to evaluate the model's stability.

#### 7.2 Model Ensemble

1. **Stacking**:
   - Combine out-of-fold predictions and test set predictions from different models to train a new meta-model. This involves using the out-of-fold predictions as training data and the actual training targets as labels to train a new model. The test set predictions from the base models are used as input for the meta-model to generate final predictions.

2. **Trick**:
   - Adjust the predicted values for anomalous users and normal users by weighting them to generate the final prediction. This method primarily relies on the output probabilities from the classification model and predefined coefficients to adjust the predictions.

3. **TrickStacking**:
   - Combine the predictions generated by the Trick method with the predictions from the Stacking model by weighted averaging. This further integrates the model predictions to improve stability and accuracy.

By following these steps, we can efficiently handle the two-stage modeling process, incorporating classification and regression models, and leveraging ensemble techniques to achieve the best possible performance.