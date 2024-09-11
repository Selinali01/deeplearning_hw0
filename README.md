# LeNet-5 FashionMNIST Experiment

This repository contains an implementation of the LeNet-5 neural network, modified to work with the FashionMNIST dataset. The code supports various regularization techniques such as Dropout, Batch Normalization, and L2 Regularization (Weight Decay). This README provides instructions on how to set up, run, and experiment with different regularization techniques.

## Setup

### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone xxx

### 2. Install Dependencies

```bash
pip install -r requirements.txt


![Dropout Plot](graphs/dropout.png)
![Batchnorm Plot](graphs/batchnorm.png)
![Weight Decay Plot](graphs/weight_decay.png)
![No Regularization Plot](graphs/no_reg.png)


<table border="1" cellpadding="5" cellspacing="0">
  <tr>
    <th>Regularization Method</th>
    <th>Final Train Accuracy</th>
    <th>Final Validation Accuracy</th>
    <th>Final Test Accuracy</th>
  </tr>
  <tr>
    <td>Dropout</td>
    <td>90.09%</td>
    <td>89.17%</td>
    <td>88.68%</td>
  </tr>
  <tr>
    <td>Batchnorm</td>
    <td>94.52%</td>
    <td>89.98%</td>
    <td>89.51%</td>
  </tr>
  <tr>
    <td>Weight Decay</td>
    <td>91.25%</td>
    <td>88.97%</td>
    <td>88.36%</td>
  </tr>
  <tr>
    <td>No Regularization</td>
    <td>92.52%</td>
    <td>90.50%</td>
    <td>89.87%</td>
  </tr>
</table>
