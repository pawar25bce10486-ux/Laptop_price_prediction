# Laptop Price Prediction: A Regression Analysis

## Overview
The goal of this project was to create and evaluate multiple **regression models** to accurately predict the price of laptops based on their core specifications. This analysis utilized a filtered and modified subset of the **Laptop Price Prediction using specifications dataset** from Kaggle.

The project implemented and compared **Linear Regression** (single and multiple variables) and **Polynomial Regression** models, including the use of a data **Pipeline** for enhanced feature processing.

A comprehensive multi-variable model achieved an **R-squared (R²)** score of **0.5083** and a **Mean Squared Error (MSE)** of **161,680.57**. This performance demonstrates the model's ability to explain a significant portion of the variance in laptop prices using hardware specifications.

***

## Business Understanding
In the competitive electronics market, accurately **pricing new laptop models** is critical for manufacturers and retailers. Setting an optimal price maximizes profitability while maintaining market share.

The objective of this project is to create a predictable model that quantifies the relationship between internal hardware specifications (e.g., **CPU frequency**, **RAM**, **Storage\_GB\_SSD**) and the final **Price**. This understanding is invaluable for:
* **Informed Pricing Strategies:** Establishing a data-driven basis for pricing new product lines.
* **Value Assessment:** Identifying which components contribute most significantly to the final market value of a laptop.

***

## 📊 Data Understanding
The dataset used is a filtered and modified version of the **Laptop Price Prediction using specifications dataset**, available on the Kaggle website under the Database Contents License (DbCL) v1.0.

The data consists of features covering various aspects of a laptop's hardware and the target variable, **Price** (in USD).

### Key Parameters
The features used in the models are:

| Parameter | Description | Assigned Numerical Value (if applicable) |
| :--- | :--- | :--- |
| `Price` | The price of the laptop (Target Variable) | N/A |
| `Manufacturer` | The company that manufactured the laptop | N/A |
| `Screen_Size_cm` | Size of the laptop screen in cm | N/A |
| `CPU_frequency` | Frequency at which the CPU operates (GHz) | N/A |
| `RAM_GB` | Size of the RAM in GB | N/A |
| `Storage_GB_SSD` | Size of the SSD storage in GB | N/A |
| `Weight_kg` | Weight of the laptop in kgs | N/A |
| `Category` | Laptop type | Gaming (1), Netbook (2), Notebook (3), Ultrabook (4), Workstation (5) |
| `GPU` | GPU manufacturer | AMD (1), Intel (2), NVidia (3) |
| `OS` | Operating system type | Windows (1), Linux (2) |
| `CPU_core` | Processor type | i3 (3), i5 (5), i7 (7) |

### Preprocessing
Categorical features (`Category`, `GPU`, `OS`, `CPU_core`) were converted to **numerical values** through ordinal encoding to be compatible with the regression algorithms.

***

## Modeling and Evaluation

### Objectives
The project focused on the following regression tasks:
* Use **Linear Regression** in one variable (`CPU_frequency`) to fit a simple model.
* Use **Linear Regression** in multiple variables (`Z` features) to fit a complex model.
* Use **Polynomial Regression** in a single variable (`CPU_frequency`) to fit varying degrees (1, 3, 5).
* Create a **Pipeline** for performing Linear Regression using multiple features in polynomial scaling.
* Evaluate the performance of different models on the basis of **MSE** and **R²** parameters.

### Performance Metrics (Multi-Feature Polynomial Pipeline)
The multi-feature polynomial pipeline demonstrated the best fit on the training data:

| Metric | Value |
| :--- | :--- |
| **R-squared (R²)** Score | **0.5083** |
| **Mean Squared Error (MSE)** | **161,680.57** |
| Root Mean Squared Error (RMSE) | 402.10 |

The **R² score of 0.5083** indicates that approximately 50.83% of the variability in laptop price can be explained by the features included in the model.



***

## Conclusion
The regression analysis successfully developed models to predict laptop prices. While the linear and lower-degree polynomial models showed a weak fit, the **multi-variable polynomial pipeline** provided a reasonable first-pass prediction.

### Future Work
To increase the model's R² score and further reduce the MSE, future work should include:
* **Feature Importance:** Determine which specific features in the multi-variable model are most influential.
* **Higher-Degree Polynomials:** Experiment with higher-degree polynomial features in the pipeline.
* **Advanced Algorithms:** Explore non-linear models like **Random Forest Regressor** or **Gradient Boosting** to capture more complex price determinants.
