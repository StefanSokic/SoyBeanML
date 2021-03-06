# SoyBeanML
The dataset I am working with gives all kinds of geographical and genetic
data about different varieties of soy beans. I am using various regression machine
learning models to best predict which growing conditions and plant types
produce optimal yields. 
### Progress 
Different attempts at predicting yield are in different branches and in files on the master branch. So far the
completed branches are as follows:
1. **univariate-linear-regression** - Using scikit learn  to predict
   yield based only on temperature.   
   Result: _151.16 mean squared error_
2. **multivariate-linear-regression** - Using scikit learn to predict yield
   based on all given geographical data. This includes features like top-soil,
   soil ph level, irrigation, precipitation, and many others.   
   Result: _102.72 mean square error_
3. **xgboost** - Using xgboost to predict yield
based on all given geographical data with gradient boosting. Xgboost has proven quite successful in recent kaggle competitions.   
Result: _76.47 mean squared error_  
4. **xgboost-with-feature-engineering** - On top of just using xgboost I have one hot encoded a few columns including year, plant variety, and plant family.  
Result: _53.95 mean squared error on only 1/5 of training data_   
  
  Improvement!!
