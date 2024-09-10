To run the code, first install all the dependencies in requirements.txt. Then run the code in code/main.py. The original code is set to no regularization trail. To run the code with dropout trail, set use_dropout to True. Similarly, to run the code with batch normalization, set the use_batchnorm to True. To run the code with l2 regularization, set the weight_decay to a nonzero value. To replicate our result, we set the weight_decay to 0.001. 

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
    <td>89.14%</td>
    <td>88.43%</td>
    <td>87.26%</td>
  </tr>
  <tr>
    <td>Batchnorm</td>
    <td>94.60%</td>
    <td>90.42%</td>
    <td>89.58%</td>
  </tr>
  <tr>
    <td>Weight Decay</td>
    <td>91.62%</td>
    <td>89.28%</td>
    <td>88.98%</td>
  </tr>
  <tr>
    <td>No Regularization</td>
    <td>92.12%</td>
    <td>90.12%</td>
    <td>89.34%</td>
  </tr>
</table>
