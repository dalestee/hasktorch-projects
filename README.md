# hasktorch-projects

## setup

change the torch paths in `stack.yaml` to the correct paths on your system.

then run:
```bash
stack run main
```

## Titanic

### titanic dataset: https://www.kaggle.com/c/titanic/data

### learning curve:

- GD optimizer
![titanic learning curve](/src/titanic-mlp/curves/graph-titanic-mse210.8436_GD.png)

- Adam optimizer
![titanic learning curve](/src/titanic-mlp/curves/graph-titanic-mse129.70596_Adam.png)

### accuracy: 
GD optimizer:
- 0.62200 on kaggle validation set
- 0.61616 on training set

Adam optimizer:
- 0.74641 on kaggle validation set
- 0.79904 on training set