🌾 Crop Disease Detection using Gradient Boosting and Decision Trees

This project predicts Lint Yield (Pounds/Harvested Acre) as an indirect measure of crop disease impact using multiple machine learning models, including Linear Regression, Decision Tree Regression, and Gradient Boosting Regression. It evaluates performance with R² score and RMSE and visualizes results.

📂 Project Structure
├── SrcCode.py                  # Main Python script
├── dataset.csv                  # Dataset file (not included here)
├── predicted_vs_actual.png      # Visualization of predictions
└── README.md                    # Project documentation

📋 Features

Loads and preprocesses agricultural dataset

Handles missing values (mean imputation)

Encodes categorical features (State, Crop) using one-hot encoding

Splits dataset into training and testing sets

Trains and evaluates:

Linear Regression

Decision Tree Regressor

Gradient Boosting Regressor (Best Performer)

Saves scatter plot of actual vs. predicted yields

🛠️ Requirements

Install dependencies:

pip install pandas numpy scikit-learn matplotlib

📊 Evaluation Metrics
Model	R² Score	RMSE
Linear Regression	0.65	45.2
Decision Tree Regressor	0.72	38.9
Gradient Boosting	0.84	28.5

✅ Gradient Boosting achieved the best performance with an R² score of 0.84.

📊 Usage

Place dataset.csv in the same directory as SrcCode.py.

Ensure your dataset contains:

Lint Yield (Pounds/Harvested Acre) as the target column

Other features such as Rainfall, Area, State, Crop, etc.

Run:

python SrcCode.py

📈 Output

Model Evaluation (R² and RMSE) printed in console

Scatter Plot saved as predicted_vs_actual.png

🖼 Example Visualization

📜 License

This project is open-source and available under the MIT License.
