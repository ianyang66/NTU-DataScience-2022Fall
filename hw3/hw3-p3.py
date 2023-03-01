import pandas as pd
from matplotlib import pyplot as plt
import pmdarima as pm
from pmdarima.arima.utils import ndiffs
from pmdarima.arima import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# ===========================================================
#                          Load Data
# ===========================================================
# TODO: Load data here.
train_df = pd.read_csv('hw3_Data2/train.csv',
                         index_col="Date", parse_dates=True)
test_df = pd.read_csv('hw3_Data2/test.csv',
                        index_col="Date", parse_dates=True)
all_df = pd.concat([train_df,test_df])

# print("train: ")
# print(train_df)
# print("test: ")
# print(test_df)

# ===========================================================
#                          Load Data
# ===========================================================


# ===========================================================
#                           ARIMA
# =========================================================== 

# d-val
d_val = ndiffs(train_df['Close'], test='adf')
print("d-val: ")
print(d_val)

## fit stepwise auto-ARIMA
#splitting the data to train and test sets based on Ntest value
#last  days


#Define auto-arima to find best model
# model = pm.auto_arima(train_df['Close'],
#                       d = d_val,
#                       start_p = 0,
#                       max_p = 5,
#                       start_q = 0,
#                       max_q = 5,
#                       D=None,
#                       m=25,
#                       stepwise=True,
#                       trace=True)
# print(model.get_params())
# print(model.summary())

y = train_df['Close'].to_numpy()
model1 = ARIMA(order=(0,1,0), seasonal_order = (0, 0, 1, 25))
model1.fit(y)
print(model1.get_params())
print(model1.summary())

# ===========================================================
#                           ARIMA
# =========================================================== 


# ===========================================================
#                       Visualization
# =========================================================== 
def plot_result(model, data, train, test, col_name, Ntest):
    
    params = model.get_params()
    d = params['order'][1]
    
    #In sample data prediction
    train_pred = model.predict_in_sample(start=d, end=-1)

    test_pred, conf = model.predict(n_periods=Ntest, return_conf_int=True)
    #print(len(test_pred))
    
    #plotting real values, fitted values and prediction values
    plt.plot(data[col_name].index, data[col_name], label='Actual Values')
    plt.plot(train.index[d:], train_pred, color='green', label='Fitted Values')
    plt.plot(test.index, test_pred, label='Forecast Values')
    #print(test.index)
    plt.fill_between(test.index, conf[:,0], conf[:,1], color='red', alpha=0.2)
    plt.legend()
    plt.savefig('3-3_result.png')
    #evaluating the model using RMSE and MAE metrics
    y_true = test_df[col_name].values
    mse = mean_squared_error(y_true,test_pred)
    mae = mean_absolute_error(y_true,test_pred)
    return mse, mae

mse , mae = plot_result(model1, all_df, train_df, test_df, 'Close', Ntest=len(test_df['Close']))
print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)
# ===========================================================
#                       Visualization
# =========================================================== 