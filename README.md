[README.md](https://github.com/user-attachments/files/21707088/README.md)
# Crop Yield Prediction

This project predicts **Lint Yield (Pounds/Harvested Acre)** using multiple machine learning models, including **Linear Regression**, **Decision Tree Regression**, and **Gradient Boosting Regression**. It evaluates model performance using **R² score** and **RMSE** and visualizes the results.

## 📂 Project Structure
```
├── SrcCode.py                  # Main Python script
├── dataset.csv                  # Dataset file (not included here)
├── predicted_vs_actual.png      # Visualization of predictions
└── README.md                    # Project documentation
```

## 📋 Features
- Loads and preprocesses agricultural dataset
- Handles missing values
- Encodes categorical features (`State`, `Crop`)
- Splits dataset into **training** and **testing** sets
- Trains and evaluates:
  - Linear Regression
  - Decision Tree Regression
  - Gradient Boosting Regression
- Saves scatter plot of actual vs. predicted yields

## 🛠️ Requirements
Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## 📊 Usage
1. Place `dataset.csv` in the same directory as `SrcCode.py`.
2. Ensure your dataset contains:
   - `Lint Yield (Pounds/Harvested Acre)` as the target column
   - Other features such as `Rainfall`, `Area`, `State`, `Crop`, etc.
3. Run:
```bash
python SrcCode.py
```

## 📈 Output
- **Model Evaluation** (R² and RMSE) printed in console
- **Scatter Plot** saved as `predicted_vs_actual.png`

## 🖼 Example Visualization
![Predicted vs Actual](predicted_vs_actual.png)

## 📜 License
This project is open-source and available under the **MIT License**.
