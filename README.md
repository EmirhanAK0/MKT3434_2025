## Added Features and Explanations

### 1. Loss Function Selection – Regression

**Description:**  
Users can select different loss functions (MSE, MAE, Huber) for Linear Regression and SVR. The loss is calculated dynamically based on this choice.

**Code:**
```python
self.loss_combo = QComboBox()
self.loss_combo.addItems(["MSE", "MAE", "Huber"])
...
loss_type = self.loss_combo.currentText()
if loss_type == "MSE":
    loss_value = mean_squared_error(self.y_test, y_pred)
elif loss_type == "MAE":
    loss_value = mean_absolute_error(self.y_test, y_pred)
elif loss_type == "Huber":
    loss_value = np.mean(np.where(
        np.abs(self.y_test - y_pred) < 1.0,
        0.5 * (self.y_test - y_pred) ** 2,
        1.0 * (np.abs(self.y_test - y_pred) - 0.5)
    ))
```

### 2. Loss Function Selection – Classification

**Description:**  
For classification models such as Logistic Regression, users can select between Cross-Entropy and Hinge loss functions.

**Code:**
```python
self.classification_loss_combo = QComboBox()
self.classification_loss_combo.addItems(["Cross-Entropy", "Hinge"])
...
if loss_type == "Cross-Entropy":
    y_proba = model.predict_proba(self.X_test)
    loss_value = log_loss(self.y_test, y_proba)
elif loss_type == "Hinge":
    y_pred_bin = model.predict(self.X_test)
    loss_value = hinge_loss(self.y_test, y_pred_bin)
```

### 3. Support Vector Regression (SVR)

**Description:**  
SVR was added to the GUI. It allows the user to customize hyperparameters such as kernel, C, and epsilon.

**Code:**
```python
model = SVR(
    C=param_widgets["C"].value(),
    epsilon=param_widgets["epsilon"].value(),
    kernel=param_widgets["kernel"].currentText()
)
```

### 4. Support Vector Machine (SVM) – Classification

**Description:**  
SVM is supported for classification. The GUI allows kernel, C, and degree values to be set by the user.

**Code:**
```python
model = SVC(
    C=param_widgets["C"].value(),
    kernel=param_widgets["kernel"].currentText(),
    degree=param_widgets["degree"].value()
)
```

### 5. Naive Bayes – Custom Priors and var_smoothing

**Description:**  
GaussianNB was enhanced to allow configuration of `var_smoothing` and optionally accept user-defined class priors.

**Code:**
```python
if prior_type == "Uniform":
    model = GaussianNB(var_smoothing=var_smoothing)
else:
    priors = list(map(float, self.prior_input.text().split(",")))
    model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
```

### 6. Missing Data Handling Options

**Description:**  
The user can choose how to handle missing values: Mean Imputation, Interpolation, Forward Fill, Backward Fill.

**Code:**
```python
if method == "Mean Imputation":
    imputer = SimpleImputer(strategy="mean")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
elif method == "Interpolation":
    return X.interpolate()
elif method == "Forward Fill":
    return X.fillna(method='ffill')
elif method == "Backward Fill":
    return X.fillna(method='bfill')
```

### 7. Show Model Name in Metrics Panel

**Description:**  
The name of the trained model is shown above the metrics for better clarity.

**Code:**
```python
metrics_text = f"Model: {name}\n\n"
```

