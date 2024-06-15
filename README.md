### Report on Support Vector Machine (SVM) Classifier for Breast Cancer Diagnosis

*** Introduction:
 This report outlines the process of utilizing Support Vector Machine (SVM) classifiers to predict breast cancer diagnosis based on clinical features. 
The dataset used is the "Breast_cancer_data.csv", containing attributes related to breast cancer tumors and their diagnostic outcomes.

*** Dataset Overview:

**Attributes:**
- The dataset includes various clinical features related to breast cancer tumors, with the target variable being the diagnosis (`M` for malignant, `B` for benign).
  
**Exploratory Data Analysis:
- Initial exploration involved loading the dataset using Pandas (`pd.read_csv()`), examining the first few rows (`cancer.head()`), basic statistics
    (`cancer.describe()`), and checking for data types and missing values (`cancer.info()`).
- The distribution of the target variable (`cancer["diagnosis"].value_counts()`) provided insight into the class distribution, which is essential for
    understanding potential class imbalance.

*** Data Visualization:

**Correlation Analysis:
- Calculated the correlation matrix (`corr`) and visualized it using a heatmap (`sns.heatmap()`) to understand the relationships between different features.
    This analysis helps in identifying highly correlated features that may influence the prediction model.

*** Data Preprocessing:

**Feature Selection:
- Selected the first two columns (`x`) as features and the last column (`y`) as the target variable for initial modeling.

*** Data Splitting:
- Split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`, with a test size of 20% and a random state of 1
  to ensure reproducibility.

*** SVM Model Construction:

**Linear SVM:**
- Constructed a Linear SVM classifier (`SVC` with `kernel="linear"`) and trained it on the training data (`x_train`, `y_train`). Predictions were made on the
  test set (`x_test`) using `predict()`.

*** Visualization of Decision Boundary:
- Visualized the decision boundary of the Linear SVM using `DecisionBoundaryDisplay.from_estimator()` to depict how the classifier separates the classes
  based on the selected features (`x_train`). This visualization aids in understanding the classification performance visually.

*** Accuracy Calculation:
- Calculated the accuracy of the Linear SVM classifier using `accuracy_score` from `sklearn.metrics`.

*** SVM Model Variation:

*** Non-linear SVM (RBF Kernel):
- Implemented a Non-linear SVM with Radial Basis Function (RBF) kernel (`kernel="rbf"`), adjusted with parameters `C=10` and `gamma=2`. This model was also
    trained, predicted, and evaluated for accuracy similarly to the Linear SVM.

*** Probability Prediction:
- Modified the RBF SVM (`SVC_diag_1`) to predict probabilities using `predict_proba()` to assess the confidence of predictions.

*** Conclusion:
- The SVM classifiers demonstrated effective performance in predicting breast cancer diagnosis based on the clinical features. Both Linear and Non-linear (RBF)
  SVM models achieved respectable accuracy scores on the test set, indicating their ability to generalize well to unseen data.
- The decision boundary visualizations provided insights into how SVM classifiers separate malignant and benign cases based on the selected features.
- Future considerations may involve further hyperparameter tuning, exploring other SVM kernels, or incorporating additional features to potentially enhance
  model performance.

  

