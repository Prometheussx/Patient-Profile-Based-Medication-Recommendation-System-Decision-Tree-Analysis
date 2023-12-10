# Decision Tree for Drug Recommendation


This project involves a drug recommendation system based on patients' demographic characteristics. The dataset includes characteristics such as age, gender, blood pressure (BP), cholesterol level and sodium-potassium ratio. The project involves building a decision tree using `DecisionTreeClassifier` and making drug recommendations using this tree.


## Libraries Used


- `numpy`
- `pandas`
- `scikit-learn` (DecisionTreeClassifier)
- `matplotlib`


## Data Content


The dataset is read from a CSV file named "drug200.csv". This file contains demographic information of the patients and which drug was recommended.

### Installation

1. Clone the project repository:

```bash
git clone [https://github.com/Prometheussx/Patient-Profile-Based-Medication-Recommendation-System-Decision-Tree-Analysis]
```

2. Ensure you have Python and the required libraries installed.

## Data Preprocessing


- The data set is loaded and the first five observations are displayed.
- Input features (X) and target variable (y) are determined.
- Categorical variables are converted to numerical values.


## Decision Tree Model


- The data set is divided into training and test sets.
- A decision tree model is created using `DecisionTreeClassifier` (entropy criterion, maximum depth: 4).
- The model is trained and predictions are made on the test set.


## Model Evaluation


- The accuracy of the model (`accuracy_score`) is calculated and printed on the screen.
- *Accuracy Score: 0.9833*


## Visualization


- The generated decision tree is visualized and drawn on the screen.

![output](https://github.com/Prometheussx/Patient-Profile-Based-Medication-Recommendation-System-Decision-Tree-Analysis/assets/54312783/2be7dc1e-397c-45a4-9ab5-5437935361b7)

## How to Use


1. Download the project to your computer.
2. Provide the dataset named "drug200.csv".
3. Run the project in Jupyter Notebook or Python environment.


## Requirements


- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## License

This project is released under the [MIT License](https://github.com/Prometheussx/Kaggle-Notebook-Cancer-Prediction-ACC96.5-With-Logistic-Regression/blob/main/LICENSE).


## Author

- Email: [Email_Address](mailto:erdemtahasokullu@gmail.com)
- LinkedIn Profile: [LinkedIn Profile](https://www.linkedin.com/in/erdem-taha-sokullu/)
- GitHub Profile: [GitHub Profile](https://github.com/Prometheussx)
- Kaggle Profile: [@erdemtaha](https://www.kaggle.com/erdemtaha)

Feel free to reach out if you have any questions or need further information about the project.
