
## one_hot_encoder
This library helps machine learning practicionnes to easily encode categorical variables when working with pandas datafames. Here are the added values of this library:
- The value returned by the get_dummies method is a pandas dataframe: The sklearn encoders return arrays.
- The get_dummies method ensures uniqueness of column names and returns clear column names so that it's easy to link the category to the column that represents it: That's not the case
of the library category_encoders
- Thanks to the fit method, we always have the test and train data always have the same columns: Pandas get_dummies method has not a fit method,
so if for example test data and train data has different categories for a certain column, you will get different columns.

To use it, run in your terminal:
```
pip install one_hot_encoder
```

If you want to develop and contribute, you are welcome. Here are the different steps to follow:

##### Create a virtualenv
```
mkvirtualenv virtualenv_name
workon virtualenv_name
```
```
# install dependencies
pip install -r dev/requirements.txt
```
##### Run tests
```
pytests
```

#### Use Case:
The use of this library is very easy. There are only two methods to use: fit and get_dummies.


```
from one_hot_encoder.encoder import Encoder

encoder = Encoder()
encoder.fit(train_data)

train_data_with_dummies = encoder.get_dummies(train_data)
test_data_with_dummies = encoder.get_dummies(test_data)
```

Here we suppose that train_data and test_data are pandas Dataframes. With those
few lines. This library is very useful for production environments. As a matter fact:
- You do not need to encode column by column: You encode all the columns you need just once.
- When predicting on one row, you cannot use the panda method get_dummies because you will not get
the same columns which make predictions impossible to do (unless you use some tricky manipulations.)

