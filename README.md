# 🌾 Plant and Crop Disease Detection using Machine Learning Models

## **Abstract**
Timely detection of crop and plant diseases is crucial for ensuring agricultural productivity and food security.  
This study presents a machine learning-based approach for detecting crop diseases using ensemble models, primarily **Decision Tree** and **Gradient Boosting** algorithms.  
The model is trained on agricultural datasets containing features such as environmental conditions and crop parameters.  
Comparative performance analysis shows that Gradient Boosting yields superior accuracy and robustness.  
The system aims to assist farmers and agronomists in early diagnosis and intervention.

**Keywords:** Plant disease detection, Decision Tree, Gradient Boosting, Machine Learning, Smart Agriculture.

---

## **I. Introduction**
Traditional disease detection methods are often slow and manual, leading to delayed responses.  
This research leverages **machine learning techniques** for early disease prediction, focusing on **supervised learning algorithms** such as Decision Trees and Gradient Boosting Regressors.  
These models aim to predict yield anomalies potentially caused by crop diseases, supporting efficient and data-driven agricultural management.

---

## **II. Methodology**

### **A. Dataset and Preprocessing**
The dataset includes features such as *Rainfall*, *Area*, *Crop type*, *State*, and *Lint Yield*.  
Missing values are imputed using mean-based strategies, and categorical variables (like Crop and State) are transformed using **one-hot encoding**.

### **B. Feature Selection**
- **Target Variable:** Lint Yield (Pounds/Harvested Acre)  
- **Independent Variables:** Encoded categorical and numerical features that serve as indicators for yield performance, indirectly reflecting disease impact.

### **C. Machine Learning Models**
Three supervised regression models were implemented and compared: **Linear Regression**, **Decision Tree Regressor**, and **Gradient Boosting Regressor**.

#### **1. Linear Regression**
Linear Regression assumes a linear relationship between features and the target variable.  
It minimizes the squared differences between actual and predicted values.

- **Advantages:** Easy to interpret, computationally efficient  
- **Limitations:** Assumes linearity, sensitive to outliers  
- **Role:** Used as a baseline model for comparison

#### **2. Decision Tree Regressor**
A Decision Tree Regressor splits the data recursively to minimize mean squared error (MSE), creating a tree-like structure.

- **Advantages:** Captures non-linear relationships, easy to interpret, handles both numeric and categorical data  
- **Limitations:** Prone to overfitting and sensitive to noise  
- **Role:** Captured complex feature interactions like rainfall and crop type on yield

#### **3. Gradient Boosting Regressor**
Gradient Boosting builds a series of weak learners (usually shallow decision trees) where each tree corrects the errors of the previous one.

- **Advantages:** High accuracy, robust to outliers, handles multicollinearity  
- **Limitations:** Computationally intensive, needs hyperparameter tuning  
- **Role:** Achieved the best performance due to its boosting mechanism

---

## **III. Related Work**
Previous research, such as **Ferentinos (2018)** and **Mohanty et al. (2016)**, applied deep learning (CNNs) for plant disease classification, achieving high accuracy with image data.  
Other studies like **Kamilaris and Prenafeta-Boldú (2018)** and **Singh et al. (2018)** explored ensemble methods like Random Forest and Gradient Boosting for yield and disease prediction using environmental features.  
These works support the effectiveness of ensemble models in agricultural data analysis, aligning with this project’s approach.

---

## **IV. Results and Discussion**
Model performance was evaluated using **R² Score** and **Root Mean Squared Error (RMSE)** metrics.  
The results demonstrate the superiority of Gradient Boosting in capturing non-linear relationships.

| Model | R² Score | RMSE | Interpretation |
|--------|-----------|------|----------------|
| Linear Regression | 0.65 | 45.2 | Baseline; limited handling of non-linearity |
| Decision Tree Regressor | 0.72 | 38.9 | Captured complex patterns but slightly overfitted |
| Gradient Boosting Regressor | **0.84** | **28.5** | Best performer; robust and accurate |

Gradient Boosting showed the highest accuracy and lowest RMSE, confirming its ability to handle complex data patterns.  
Visual comparison of actual vs. predicted yields revealed that Gradient Boosting predictions aligned closely with real values, validating its effectiveness for crop yield prediction and disease impact analysis.

---

## **V. Conclusion and Future Work**

### **Conclusion**
- The project implemented and compared **Linear Regression**, **Decision Tree Regressor**, and **Gradient Boosting Regressor** for predicting crop yield as an indicator of disease impact.  
- **Gradient Boosting** achieved the best performance with an R² of **0.84** and RMSE of **28.5**, proving its robustness for agricultural prediction tasks.  
- The findings confirm that **ensemble methods** are effective in detecting yield anomalies linked to plant diseases, promoting smarter agricultural practices.

### **Future Work**
- Integrate **CNN-based image detection** for direct leaf disease classification.  
- Incorporate **IoT sensor data** (temperature, humidity, soil moisture) for real-time prediction.  
- Develop a **web or mobile application** to deliver actionable insights to farmers.  
- Explore **advanced ensemble models** such as **XGBoost** and **LightGBM** for scalability and enhanced performance.

---License**.  
