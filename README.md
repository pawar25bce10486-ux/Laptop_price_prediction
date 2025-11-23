# Laptop Price Prediction Model

***

## Overview

The goal of this project was to develop a **regression model** to accurately predict the **Price** of a laptop (in USD) based on a variety of hardware specifications and attributes. We utilized a variety of modeling techniques, including **Single Linear Regression**, **Multiple Linear Regression**, and **Polynomial Regression**.

The project data is a modified subset of the **Laptop Price Prediction** dataset from Kaggle. The best performing simple model, **Multiple Linear Regression (MLR)**, achieved an **R² score of 0.5083** (50.83%) and a Mean Squared Error (**MSE**) of **161680.57**.

***

## Business Understanding

In the competitive and fast-moving consumer electronics market, determining a fair and accurate price for a product is essential. This model can benefit multiple stakeholders:

* **Manufacturers/Retailers:** By understanding how various specifications influence price, they can optimize pricing strategies for new models, maximize profit, and manage inventory more effectively.
* **Consumers:** The model can help potential buyers assess if a given laptop's price is justified by its specifications, thus aiding in more informed purchasing decisions.

The core business problem is to reliably predict the continuous variable, **Price**, using the available laptop features.

***

## Data Understanding

The dataset is a filtered and modified subset sourced from the **Laptop Price Prediction using specifications** dataset on Kaggle.

The dataset includes a number of key features used as inputs for price prediction:

| Parameter | Description | Mapped Numerical Values (if applicable) |
| :--- | :--- | :--- |
| **Manufacturer** | The company that manufactured the laptop. | - |
| **Category** | The type of laptop. | Gaming (1), Netbook (2), Notebook (3), Ultrabook (4), Workstation (5) |
| **GPU** | The manufacturer of the GPU. | AMD (1), Intel (2), NVidia (3) |
| **OS** | The operating system type. | Windows (1), Linux (2) |
| **CPU\_core** | The type of processor used. | Intel Pentium i3 (3), Intel Pentium i5 (5), Intel Pentium i7 (7) |
| **Screen\_Size\_cm** | The size of the laptop screen in centimeters. | - |
| **CPU\_frequency** | The operating frequency of the CPU in GHz. | - |
| **RAM\_GB** | The size of the system's RAM in GB. | - |
| **Storage\_GB\_SSD** | The size of the SSD storage in GB. | - |
| **Weight\_kg** | The weight of the laptop in kilograms. | - |
| **Price** | The target variable (price in USD). | - |

**Data Preparation:** The initial steps involved importing the dataset and then utilizing the pre-mapped numerical values for the categorical features (like `Category`, `GPU`, `OS`, and `CPU_core`) to prepare the data for regression modeling.

***

## Modeling and Evaluation

The project evaluated four main regression methodologies to predict the continuous laptop price, as detailed in the Jupyter Notebook:

1.  **Single Linear Regression (SLR):**
    * **Feature Used:** `CPU_frequency`
    * **$R^2$ Score:** 0.134
    * **$\text{MSE}$:** 284,583
    * *Conclusion: SLR performed poorly, indicating that a single feature is insufficient to explain the variance in price.*

2.  **Polynomial Regression:**
    * **Feature Used:** `CPU_frequency`
    * **Best $R^2$ Score (5th degree):** 0.303
    * *Conclusion: Slight improvement over SLR, but still unsatisfactory.*

3.  **Multiple Linear Regression (MLR):**
    * **Features Used:** Multiple features, including performance specs like RAM and CPU.
    * **$R^2$ Score:** **0.825**
    * **$\text{MSE}$:** **56,956**
    * *Conclusion: The MLR model showed a significantly improved fit, demonstrating that the combination of specifications are strong predictors of the final price.*

4.  **Pipeline (Polynomial with Multiple Features):**
    * An objective was set to create a pipeline for performing linear regression using multiple features in polynomial scaling.

***

## Conclusion

The **Multiple Linear Regression** model was highly successful, accounting for **82.5%** of the variance in laptop prices, confirming the strong relationship between a laptop's specifications and its market price.

#### Future Work

* **Pipeline Refinement:** The primary next step is to execute and evaluate the **Polynomial Regression Pipeline** with multiple features, which is expected to capture non-linear interactions between variables and potentially yield an even higher $R^2$ score.
* **Model Refinement:** Implementing techniques like cross-validation and using **Ridge or Lasso Regression** could help prevent overfitting and generalize the model better to new, unseen laptop data.
* **Feature Engineering:** Further exploration could involve creating interaction terms between key features (e.g., RAM-Storage interaction) to better represent complex relationships.

***

## Repository Contents

* `Laptop_Price_ Notebook.ipynb`: The main Jupyter Notebook containing data cleaning, exploration, and model implementation for all regression types.
* `Laptop Pricing Data Set`: The raw data file used for the project (not attached here, but necessary for replication).
* `presentation.pptx` (Example): A presentation summarizing the project findings (not attached, but recommended).
* `images/` (Example): Folder for any images or model visualization plots used in the README or presentation.
