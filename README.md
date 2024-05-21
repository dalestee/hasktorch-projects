# Hasktorch Projects

## Setup
- Install [Stack](https://docs.haskellstack.org/en/stable/README/).
- Clone   [Hasktorch](https://github.com/hasktorch/hasktorch.git).
- Clone   [Hasktorch-tools](https://github.com/DaisukeBekki/hasktorch-tools.git).
- Change the torch paths in `stack.yaml` to the correct paths on your system.
- Configure .bashrc
  **Add to your .bashrc:**
    ```bash
    HASKTORCH="~/.local/lib/hasktorch"
    ```

  **In the cloned Hasktorch repository, write this series of command into the terminal:**
    ```bash
    source setenv
    cat $LD_LIBRARY_PATH
    ```
  
  **and copy the output and add it to your .bashrc like this:**
    ```bash
    export LD_LIBRARY_PATH=<LD_LIBRARY_PATH that you copied>
    ```
for linear **regression**:
```bash
stack run linear
```

for **titanic**:
```bash
stack run titanic
```

## Linear Regression

**Hyperparameters**
- **Input Layer**  1
- **Hidden Layer**  1
- **Output Layer**  1
- **Learning Rate** 0.01

## Learning Curves

- **GD Optimizer:**
  ![Linear Regression Learning Curve with GD Optimizer](/app/linearRegression/curves/graph-linear-good.png)

## Accuracy
- **GD Optimizer:**
  - Test Set:     1.0
  - Training Set: 0.8

## Titanic

### Titanic Dataset
- You can find the Titanic dataset [here](https://www.kaggle.com/c/titanic/data).

**Hyperparameters:**
- **Input Layer:**   7
- **Hidden Layer:**  21
- **Output Layer:**  1
- **Learning Rate:** 0.01

### Learning Curves
- **GD Optimizer:**
  ![Titanic Learning Curve with GD Optimizer](/app/titanic-mlp/curves/graph-titanic-mse210.8436_GD.png)

- **Adam Optimizer:**
  ![Titanic Learning Curve with Adam Optimizer](/app/titanic-mlp/curves/graph-titanic-mse129.70596_Adam.png)

### Accuracy
- **GD Optimizer:**
  - Kaggle Test Set: 0.62200
  - Training Set:    0.61616

- **Adam Optimizer:**
  - Kaggle Test Set: 0.74641
  - Training Set:    0.79904

## Cifar
- You can find the cifar dataset [here](https://github.com/hasktorch/hasktorch/blob/master/hasktorch/src/Torch/Vision.hs).
- You can find the cifar competition [here](https://www.kaggle.com/competitions/cifar-10).

**Hyperparameters:**
- **Input Layer:**   3074
- **Hidden Layer:**  256
- **Output Layer:**  256
- **Learning Rate:** 0.001


### Learning Curves
- **Adam Optimizer:**
  ![Cifar Learning Curve with Adam Optimizer](/app/cifar/curves/lossCifar256x256.png)

### Accuracy
- **Adam Optimizer:**
  | Epoch | Loss     | Kaggle Accuracy | ValidData Accuracy | F1 Macro   |
  |-------|--------- |-----------------|--------------------|------------|
  | 50    | 3258.0178| -               | 0.4054             | 0.3384     |
  | 550   | 2613.6038| -               | 0.4978             | 0.4967     |
  | 1050  | 2321.2322| -               | 0.5046             | 0.5016     |
  | 1550  | 2170.7373| -               | 0.5084             | 0.5063     |
  | 2050  | 1906.2094| -               | 0.5088             | 0.5104     |
  | 2550  | 1774.9410| -               | 0.5182             | 0.5184     |
  | 3050  | 1658.4749| -               | 0.5096             | 0.5094     |
  | 3550  | 1482.1902| -               | 0.5166             | 0.5176     |
  | 3850  | 2347.8490| 0.5118          | 0.5100             | 0.5071     |

## Observations
For Multi-class classification it is optimal to feed the data randomly to the model when training

Use the Confusion Matrix to analyze the data

Store the loss value validation on the and the loss value on the training set to analyze the model comparing both curves

When Evaluating also store the Precision and the Recall to compare to the F1 scores

Save the embedding values as parameters

two techniques of tokenization: subword and part of speech