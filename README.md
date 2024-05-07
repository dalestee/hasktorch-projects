# Hasktorch Projects

## Setup
- Change the torch paths in `stack.yaml` to the correct paths on your system.

then run:
```bash
stack run main
```

## Linear Regression

**Hyperparameters**
- **Input Layer** 1
- **Hidden Layer** 1
- **Output Layer** 1
- **Learning Rate** 0.01

## Learning Curves

- **GD Optimizer:**
  ![Linear Regression Learning Curve with GD Optimizer](/app/linearRegression/curves/graph-linear-good.png)

## Accuracy
- **GD Optimizer:**
  - Test Set: 1.0
  - Training Set: 0.8

## Titanic

### Titanic Dataset
- You can find the Titanic dataset [here](https://www.kaggle.com/c/titanic/data).

**Hyperparameters:**
- **Input Layer:** 7
- **Hidden Layer:** 21
- **Output Layer:** 1
- **Learning Rate:** 0.01

### Learning Curves
- **GD Optimizer:**
  ![Titanic Learning Curve with GD Optimizer](/app/titanic-mlp/curves/graph-titanic-mse210.8436_GD.png)

- **Adam Optimizer:**
  ![Titanic Learning Curve with Adam Optimizer](/app/titanic-mlp/curves/graph-titanic-mse129.70596_Adam.png)

### Accuracy
- **GD Optimizer:**
  - Kaggle Test Set: 0.62200
  - Training Set: 0.61616

- **Adam Optimizer:**
  - Kaggle Test Set: 0.74641
  - Training Set: 0.79904


