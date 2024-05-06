# Hasktorch Projects

## Setup
- Change the torch paths in `stack.yaml` to the correct paths on your system.

then run:
```bash
stack run main
```

## Titanic

### Titanic Dataset
- You can find the Titanic dataset [here](https://www.kaggle.com/c/titanic/data).

### Learning Curves
- **GD Optimizer:**
  ![Titanic Learning Curve with GD Optimizer](/app/titanic-mlp/curves/graph-titanic-mse210.8436_GD.png)

- **Adam Optimizer:**
  ![Titanic Learning Curve with Adam Optimizer](/app/titanic-mlp/curves/graph-titanic-mse129.70596_Adam.png)

### Accuracy
- **GD Optimizer:**
  - Kaggle Validation Set: 0.62200
  - Training Set: 0.61616

- **Adam Optimizer:**
  - Kaggle Validation Set: 0.74641
  - Training Set: 0.79904


