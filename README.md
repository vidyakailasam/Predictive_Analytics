# Predictive_Analytics

The problem description about the dataset is to predict the price of a used Toyota Corolla based on its specifications. The topic is a regression problem. It involves numerical prediction and continuous values. The selected machine learning models for solving the given problem are:
Multiple linear regression, k-Nearest Neighbors(k-NN) and Tree based models (Decision tree, Random Forest, Gradient Search)

•	Multiple regression is a ML algorithm to predict a dependent variable with two or more predictors. Multiple regression has numerous real-world applications involving relationships between variables, making numerical predictions and time series forecasting. It is simpler to implement and less complex compared to other algorithms.

•	k-NN models are easy to implement and handle non-linearities well. It uses feature similarity and distance measures to find the appropriate matches and predicting the numerical values.

•	Tree-based models perform well for large data sets and require less data preparation than other techniques. Tree based algorithms empower predictive models with high accuracy, stability and ease of interpretation. Unlike linear models, they map non-linear relationships quite well.

The data is been preprocessed in such a way that unnecessary and redundant data are removed. The rows having all their observations as NA values are removed. The ‘Mistlamps’ column had more than half of its observations with null values. So, it has been removed from the dataset. Also, ‘Id’ and ‘Cylinders’ columns are removed as they do not add any prominent value in the numerical prediction of Corolla cars.  Other observations having quite few NA values are dropped. Categorical columns are transformed to binary variables, Eg: ‘Color’ column.  The predictors are normalized using z-score normalization to make it standard and to neutralize the features for training and testing.

For each model, the hyperparameters can be changed:

Multiple Linear Regression:
Fit_intercept=bool

k-NN: 
‘n_neighbors’ is changed from 5 to 7
‘weights’ is chosen to be ‘distance’

Decision tree:
max_depth=5, max_features='auto',max_leaf_nodes=None,min_samples_leaf=1,min_weight_fraction_leaf=0.1,splitter='best'
	
Random forest:
n_estimators=20
max_depth=20

Gradient boosting:
n_estimators=120
max_depth=20
learning_rate=0.005

**Training and Testing performances for each model:**

The metrics for evaluating regression model performance are:
Mean Absolute Error (MAE): Lesser the value of MAE, better the model.
Root Mean Square Error (RMSE): RMSE takes the square of the difference between the predictions and the actual value, hence the value will be greater the MAE value.
Coefficient of determination (R^2): The value of R^2 lies between 0 to 1. The closer the value to one, the model is better. 
Adjusted R^2: For better understanding of the effect of additional variables to the model. When new variable is introduced, the value of R^2 increases, by decreasing the adjusted R^2 value. 

**MLR:**
_Training data:_
                      Mean Error (ME) : 0.0000
       Root Mean Squared Error (RMSE) : 1316.9682
            Mean Absolute Error (MAE) : 983.3779
          Mean Percentage Error (MPE) : -1.1647
Mean Absolute Percentage Error (MAPE) : 9.9056

_Testing data:_
Regression statistics

                      Mean Error (ME) : -62.3109
       Root Mean Squared Error (RMSE) : 1412.6193
            Mean Absolute Error (MAE) : 1041.8620
          Mean Percentage Error (MPE) : -1.5385
Mean Absolute Percentage Error (MAPE) : 10.1586
__
Training data after changing hyperparameters:__
Regression statistics

                      Mean Error (ME) : 0.0000
       Root Mean Squared Error (RMSE) : 1352.8899
            Mean Absolute Error (MAE) : 1004.9636
          Mean Percentage Error (MPE) : -1.2316
Mean Absolute Percentage Error (MAPE) : 10.1860
_
Testing data after changing hyperparameters:_
Regression statistics

                      Mean Error (ME) : -80.5732
       Root Mean Squared Error (RMSE) : 1308.4928
            Mean Absolute Error (MAE) : 979.7928
          Mean Percentage Error (MPE) : -1.6790
Mean Absolute Percentage Error (MAPE) : 9.5058

**k-NN:**

_(Training data):_ Root Mean Squared Error (RMSE):  1405.882

_(After changing hyperparameters):_ Root Mean Squared Error (RMSE):  0.0003

_(Testing data):_ Root Mean Squared Error (RMSE):  1726.5195

_(After changing hyperparameters):_ Root Mean Squared Error (RMSE):  1704.2231

Optimal value for k: 6

**Tree based models:**

**Decision tree: **

_(Training data):_
Mean Absolute Error: 0.0
Mean Squared Error: 0.0
Root Mean Squared Error: 0.0

_(Testing data):_
Mean Absolute Error: 1087.5328638497654
Mean Squared Error: 2060894.2511737088
Root Mean Squared Error: 1435.5815027972842
_
(After changing hyperparameters):_
Mean Absolute Error: 1062.1335547681897
Mean Squared Error: 2414218.6251599602
Root Mean Squared Error: 1553.7756032194482

**Random forest:**
_(Training data):_
Mean Absolute Error: 317.21556898288014
Mean Squared Error: 178859.3858934542
Root Mean Squared Error: 422.9177058169286

_(Testing data):_
Mean Absolute Error: 840.9959859154931
Mean Squared Error: 1223732.729949061
Root Mean Squared Error: 1106.22453866702

_(After changing hyperparameters in train data):_
Mean Absolute Error: 337.67767104982084
Mean Squared Error: 195999.62860715998
Root Mean Squared Error: 442.71845297791685
_
(After changing hyperparameters in test data):_
Mean Absolute Error: 836.2163885893779
Mean Squared Error: 1202481.0898728112
Root Mean Squared Error: 1096.5769876633428

**Gradient boost:**

_(Training data):_
Mean Absolute Error: 641.0212861233148
Mean Squared Error: 679296.6998088812
Root Mean Squared Error: 824.1945764253009

_(Testing data):_
Mean Absolute Error: 807.3522755228529
Mean Squared Error: 1137843.7763212216
Root Mean Squared Error: 1066.6976030352846

_(After changing hyperparameters in train data):_

Mean Absolute Error: 1407.8935043763115
Mean Squared Error: 3848568.6570391175
Root Mean Squared Error: 1961.7769131680384

_(After changing hyperparameters in test data):_

Mean Absolute Error: 1492.8691457965583
Mean Squared Error: 4421584.545737378
Root Mean Squared Error: 2102.7564161684013

**Judging the fit for the models:**

**MLR:**
With the regression statistics, the model is found to be overfitting the training data as the model performs poorly on the testing data and performs better on the traing data. 

**k-NN:**
For-k-NN, the model seems to be overfit as it has higher RMSE in test compared to training data.

**Tree based models:**

Decision tree: the model seems to be overfit as it has higher RMSE in test compared to training data.

Random forest: the model seems to be overfit as it has higher RMSE in test compared to training data.

We got 100% score on training data. On test data we got 7.5% score because we did not provide any tuning parameters while initializing the tree as a result of which algorithm split the training data till the leaf node. Due to which depth of tree increased and our model did the overfitting. So, to solve this problem we used hyper parameter tuning. We used GridSearch for hyper parameters tuning. 

Gradient boost: the model seems to be overfit as it has higher RMSE in test compared to training data

**Inference:**

Gradient boosting machine learning model is recommended to for implementing and deploying it in the real-world performance analysis. The RMSE value is better in Gradient boosting compared to all other models. Tree-based modeling is an excellent alternative to linear regression analysis.It is because the model Can be used for any type of data. Can even handle data that are not normally Are easy to represent visually, making a complex predictive model much easier to interpret. Also, requires little data preparation because variable transformations are unnecessary
