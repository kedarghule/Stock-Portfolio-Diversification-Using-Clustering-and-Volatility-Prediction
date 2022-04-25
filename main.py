import streamlit as st
import os
import numpy as np
import math
import pandas as pd
import time
from arch import arch_model
import statsmodels.api as sm
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from glob import glob
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

st.set_page_config(
    page_icon='💲',
    layout='wide',
    page_title='Stock Volatility Clustering and Prediction'
)
st.title('Stock Volatility Clustering & Prediction')

# Getting the list of individual stock
path = 'datasets\\individual_stock\\'
master_path = 'datasets\\'
### CHANGE
try:
    path = '/app/stock-portfolio-diversification-using-clustering-and-volatility-prediction/datasets/individual_stock'
    dir_list = os.listdir(path)
except FileNotFoundError:
    dir_list = Path(__file__).parents[0] / 'datasets/individual_stock'
    st.write(dir_list)
    # for filename in Path(__file__).parents[0] / 'datasets\\individual_stock':
    #     dir_list.append(str(filename))
    # dir_list = [str(i) for i in dir_list]
try:
    dir_list = [os.path.splitext(x)[0] for x in dir_list]
except:
    dir_list = [os.path.splitext(str(x))[0] for x in dir_list]
## CHANGE END
# USER: Select desired Stock Ticker Symbol
symbol = st.selectbox("Symbol: ", dir_list)

# Getting the dataset ready
stock_df = pd.read_csv(path + '/{}.csv'.format(symbol))

# Importing constituents Dataframe
## CHANGE
try:
    constituents_df = pd.read_csv(master_path + '\\constituents.csv')
except FileNotFoundError:
    master_path = '/app/stock-portfolio-diversification-using-clustering-and-volatility-prediction/datasets/constituents.csv'
    constituents_df = pd.read_csv(master_path)
## CHANGE END
selected_stock = constituents_df[constituents_df['Symbol'] == symbol]
stock_name = selected_stock.Name.values[0]
stock_sector = selected_stock.Sector.values[0]
stock_sub_sector = selected_stock['Sub-Sector'].values[0]

# Displaying Stock Details
d1, d2, d3 = st.columns(3)
d1.metric(label='Name', value=stock_name)
d2.metric(label='Sector', value=stock_sector)
d3.metric(label='Sub-Sector', value=stock_sub_sector)

with st.spinner('Loading...'):
    p1, p2 = st.columns(2)

    # Plotting the Adjusted Close of the stock
    trend = px.line(stock_df, x='Date', y='Adj Close', template='plotly_white', title='Trend of Adjusted Closing Price')
    trend.update_layout(
        title_x=0.5
    )
    trend.update_traces(line_color='#Ef3b26')
    p1.plotly_chart(trend)

    # Changing Datatype of Date column
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.set_index('Date')
    stock_ticker = stock_df.Name[0]

    # Removing name column
    if 'Name' in stock_df.columns:
        stock_df = stock_df.drop(['Name'], axis=1)

    # Calculating and plotting the realized volatility of the stock
    ret = 100 * (stock_df.pct_change()[1:]['Adj Close'])
    realized_vol = ret.rolling(5).std()
    newnames = {'Adj Close': 'Realized Volatility'}  # For Visualizations
    fig = px.line(realized_vol, template='plotly_white', labels={'value': 'Volatility'},
                  title='Realized Volatility- {}'.format(stock_name))
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                          )
                       )
    fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    p2.plotly_chart(fig)

    # Setting value of n - this is the length of the test data for our ML model
    n = 300
    split_date = ret.iloc[-n:].index

    # Making dictionary to save metric results
    rmse_dict = {}
    mse_dict = {}
    mae_dict = {}
    mape_dict = {}

    q1, q2 = st.columns(2)

    # Volatility Clustering
    retv = ret.values
    vc = px.line(x=stock_df.index[1:], y=ret, template='plotly_white', labels={'x': 'Date', 'y': 'Daily Returns'},
                 title='Volatility clustering of {}'.format(stock_name))
    vc.update_layout(
        title_x=0.5
    )
    q1.plotly_chart(vc)

    # Daily Monthly and Annual Volatility
    period_volatility = {}
    daily_volatility = ret.std()
    print('Daily volatility: ', '{:.2f}%'.format(daily_volatility))
    period_volatility['Daily Volatility'] = round(daily_volatility, 2)
    monthly_volatility = math.sqrt(21) * daily_volatility
    print('Monthly volatility: ', '{:.2f}%'.format(monthly_volatility))
    period_volatility['Monthly Volatility'] = round(monthly_volatility, 2)
    annual_volatility = math.sqrt(252) * daily_volatility
    print('Annual volatility: ', '{:.2f}%'.format(annual_volatility))
    period_volatility['Annual Volatility'] = round(annual_volatility, 2)
    vt = go.Figure(data=[go.Table(header=dict(values=['Period', 'Percentage Volatility'],
                                              fill_color='#0e0101',
                                              font={'color': 'white'}),
                                  cells=dict(values=[list(period_volatility.keys()), list(period_volatility.values())],
                                             fill_color='#D4d0d0'))
                         ])
    q2.write(vt)

    st.subheader('Classical Volatility Models')

    r1, r2 = st.columns(2)
    # ARCH Model - P value is chosen such that BIC is minimized
    arch = arch_model(ret, mean='zero', vol='ARCH', p=1).fit(disp='off')
    # print(arch.summary())
    bic_arch = []
    for p in range(1, 5):
        arch = arch_model(ret, mean='zero', vol='ARCH', p=p).fit(disp='off')
        bic_arch.append(arch.bic)
        if arch.bic == np.min(bic_arch):
            best_param = p
    arch = arch_model(ret, mean='zero', vol='ARCH', p=best_param).fit(disp='off')
    print(arch.summary())
    forecast = arch.forecast(start=split_date[0])
    forecast_arch = forecast
    rmse_arch = np.sqrt(mse(realized_vol[-n:] / 100, np.sqrt(forecast_arch.variance.iloc[-len(split_date):] / 100)))
    # print('The RMSE value of ARCH model is {:.4f}'.format(rmse_arch))
    rmse_dict['ARCH'] = rmse_arch
    mse_arch = mse(realized_vol[-n:] / 100, np.sqrt(forecast_arch.variance.iloc[-len(split_date):] / 100))
    # print('The MSE value of ARCH model is {:.4f}'.format(mse_arch))
    mse_dict['ARCH'] = mse_arch
    mae_arch = mae(realized_vol[-n:] / 100, np.sqrt(forecast_arch.variance.iloc[-len(split_date):] / 100))
    # print('The MAE value of ARCH model is {:.4f}'.format(mae_arch))
    mae_dict['ARCH'] = mae_arch
    arch_forecast_df = forecast_arch.variance.iloc[-len(split_date):] / 100
    arch_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                       title='Volatility Prediction with ARCH')
    arch_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                               legendgroup=newnames[t.name],
                                               hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                               )
                            )
    arch_fig.add_trace(
        go.Scatter(x=arch_forecast_df.index, y=arch_forecast_df['h.1'], mode='lines', name='ARCH Predictions'))
    arch_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    r1.plotly_chart(arch_fig)

    # GARCH Model
    garch = arch_model(ret, mean='zero', vol='GARCH', p=1, o=0, q=1).fit(disp='off')
    # print(garch.summary())
    bic_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            garch = arch_model(ret, mean='zero', vol='GARCH', p=p, o=0, q=q).fit(disp='off')
            bic_garch.append(garch.bic)
            if garch.bic == np.min(bic_garch):
                best_param = p, q
    garch = arch_model(ret, mean='zero', vol='GARCH', p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
    print(garch.summary())
    forecast = garch.forecast(start=split_date[0])
    forecast_garch = forecast
    rmse_garch = np.sqrt(mse(realized_vol[-n:] / 100, np.sqrt(forecast_garch.variance.iloc[-len(split_date):] / 100)))
    # print('The RMSE value of GARCH model is {:.4f}'.format(rmse_garch))
    rmse_dict['GARCH'] = rmse_garch
    mse_garch = mse(realized_vol[-n:] / 100, np.sqrt(forecast_garch.variance.iloc[-len(split_date):] / 100))
    # print('The MSE value of GARCH model is {:.4f}'.format(mse_garch))
    mse_dict['GARCH'] = mse_garch
    mae_garch = mae(realized_vol[-n:] / 100, np.sqrt(forecast_garch.variance.iloc[-len(split_date):] / 100))
    # print('The MAE value of GARCH model is {:.4f}'.format(mae_garch))
    mae_dict['GARCH'] = mae_garch
    garch_forecast_df = forecast_garch.variance.iloc[-len(split_date):] / 100
    garch_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                        title='Volatility Prediction with GARCH')
    garch_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                legendgroup=newnames[t.name],
                                                hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                )
                             )
    garch_fig.add_trace(
        go.Scatter(x=garch_forecast_df.index, y=garch_forecast_df['h.1'], mode='lines', name='GARCH Predictions'))
    garch_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    r2.plotly_chart(garch_fig)

    s1, s2 = st.columns(2)

    # GJR-GARCH Model
    bic_gjr_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            gjrgarch = arch_model(ret, mean='zero', p=p, o=1, q=q).fit(disp='off')
            bic_gjr_garch.append(gjrgarch.bic)
            if gjrgarch.bic == np.min(bic_gjr_garch):
                best_param = p, q
    gjrgarch = arch_model(ret, mean='zero', p=best_param[0], o=1,
                          q=best_param[1]).fit(disp='off')
    # print(gjrgarch.summary())
    forecast = gjrgarch.forecast(start=split_date[0])
    forecast_gjrgarch = forecast
    rmse_gjr_garch = np.sqrt(
        mse(realized_vol[-n:] / 100, np.sqrt(forecast_gjrgarch.variance.iloc[-len(split_date):] / 100)))
    # print('The RMSE value of GJR-GARCH models is {:.4f}'.format(rmse_gjr_garch))
    rmse_dict['GJR-ARCH'] = rmse_gjr_garch
    mse_gjr_garch = mse(realized_vol[-n:] / 100, np.sqrt(forecast_gjrgarch.variance.iloc[-len(split_date):] / 100))
    # print('The MSE value of GJR-GARCH model is {:.4f}'.format(mse_gjr_garch))
    mse_dict['GJR-GARCH'] = mse_gjr_garch
    mae_gjr_garch = mae(realized_vol[-n:] / 100, np.sqrt(forecast_gjrgarch.variance.iloc[-len(split_date):] / 100))
    # print('The MAE value of GJR-GARCH model is {:.4f}'.format(mae_gjr_garch))
    mae_dict['GJR-GARCH'] = mae_gjr_garch
    gjrgarch_forecast_df = forecast_gjrgarch.variance.iloc[-len(split_date):] / 100
    gjr_garch_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                            title='Volatility Prediction with GJR-GARCH')
    gjr_garch_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                    legendgroup=newnames[t.name],
                                                    hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                    )
                                 )
    gjr_garch_fig.add_trace(
        go.Scatter(x=gjrgarch_forecast_df.index, y=gjrgarch_forecast_df['h.1'], mode='lines', name='GJR-GARCH Predictions'))
    gjr_garch_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    s1.plotly_chart(gjr_garch_fig)

    # EGARCH Model
    bic_egarch = []
    for p in range(1, 5):
        for q in range(1, 5):
            egarch = arch_model(ret, mean='zero', vol='EGARCH', p=p, q=q).fit(disp='off')
            bic_egarch.append(egarch.bic)
            if egarch.bic == np.min(bic_egarch):
                best_param = p, q
    egarch = arch_model(ret, mean='zero', vol='EGARCH', p=best_param[0], q=best_param[1]).fit(disp='off')
    # print(egarch.summary())
    forecast = egarch.forecast(start=split_date[0])
    forecast_egarch = forecast
    rmse_egarch = np.sqrt(mse(realized_vol[-n:] / 100, np.sqrt(forecast_egarch.variance.iloc[-len(split_date):] / 100)))
    # print('The RMSE value of EGARCH models is {:.4f}'.format(rmse_egarch))
    rmse_dict['EARCH'] = rmse_egarch
    mse_egarch = mse(realized_vol[-n:] / 100, np.sqrt(forecast_egarch.variance.iloc[-len(split_date):] / 100))
    # print('The MSE value of EGARCH model is {:.4f}'.format(mse_egarch))
    mse_dict['EGARCH'] = mse_egarch
    mae_egarch = mae(realized_vol[-n:] / 100, np.sqrt(forecast_egarch.variance.iloc[-len(split_date):] / 100))
    # print('The MAE value of EGARCH model is {:.4f}'.format(mae_egarch))
    mae_dict['EGARCH'] = mae_egarch
    egarch_forecast_df = forecast_egarch.variance.iloc[-len(split_date):] / 100
    egarch_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                         title='Volatility Prediction with EGARCH')
    egarch_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                 legendgroup=newnames[t.name],
                                                 hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                 )
                              )
    egarch_fig.add_trace(
        go.Scatter(x=egarch_forecast_df.index, y=egarch_forecast_df['h.1'], mode='lines', name='EGARCH Predictions'))
    egarch_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    s2.plotly_chart(egarch_fig)

    st.subheader('Machine Learning Approaches')
    t1, t2 = st.columns(2)

    # SVR-GARCH - Linear
    realized_vol = ret.rolling(5).std()
    realized_vol = pd.DataFrame(realized_vol)
    realized_vol.reset_index(drop=True, inplace=True)
    returns_svm = ret ** 2
    returns_svm = returns_svm.reset_index()
    del returns_svm['Date']
    X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
    X = X[4:].copy()
    X = X.reset_index()
    X.drop('index', axis=1, inplace=True)
    realized_vol = realized_vol.dropna().reset_index()
    realized_vol.drop('index', axis=1, inplace=True)
    svr_lin = SVR(kernel='linear')
    svr_rbf = SVR(kernel='rbf')
    para_grid = {'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
    clf = RandomizedSearchCV(svr_lin, para_grid)
    clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n - 1)].values.reshape(-1, ))
    predict_svr_lin = clf.predict(X.iloc[-n:])
    predict_svr_lin = pd.DataFrame(predict_svr_lin)
    predict_svr_lin.index = ret.iloc[-n:].index
    rmse_svr = np.sqrt(mse(realized_vol.iloc[-n:] / 100, predict_svr_lin / 100))
    # print('The RMSE value of SVR with Linear Kernel is {:.6f}'.format(rmse_svr))
    rmse_dict['SVR-GARCH (Linear)'] = rmse_svr
    mse_svr = mse(realized_vol.iloc[-n:] / 100, predict_svr_lin / 100)
    # print('The MSE value of SVR with Linear Kernel is {:.7f}'.format(mse_svr))
    mse_dict['SVR-GARCH (Linear)'] = mse_svr
    mae_svr = mae(realized_vol.iloc[-n:] / 100, predict_svr_lin / 100)
    # print('The MAE value of SVR with Linear Kernel is {:.6f}'.format(mae_svr))
    mae_dict['SVR-GARCH (Linear)'] = mae_svr
    realized_vol.index = ret.iloc[4:].index
    svr_lin_forecast_df = predict_svr_lin / 100
    svr_lin_forecast_df.columns = ['Preds']
    svr_lin_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                          title='Volatility Prediction with SVR-GARCH (Linear)')
    svr_lin_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                  legendgroup=newnames[t.name],
                                                  hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                  )
                               )
    svr_lin_fig.add_trace(
        go.Scatter(x=svr_lin_forecast_df.index, y=svr_lin_forecast_df['Preds'], mode='lines', name='SVR-GARCH Predictions'))
    svr_lin_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    t1.plotly_chart(svr_lin_fig)

    # SVR GARCH - RBF
    clf = RandomizedSearchCV(svr_rbf, para_grid)
    clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n - 1)].values.reshape(-1, ))
    predict_svr_rbf = clf.predict(X.iloc[-n:])
    predict_svr_rbf = pd.DataFrame(predict_svr_rbf)
    predict_svr_rbf.index = ret.iloc[-n:].index
    rmse_svr_rbf = np.sqrt(mse(realized_vol.iloc[-n:] / 100, predict_svr_rbf / 100))
    # print('The RMSE value of SVR with RBF Kernel is  {:.6f}'.format(rmse_svr_rbf))
    rmse_dict['SVR-GARCH (RBF)'] = rmse_svr_rbf
    mse_svr_rbf = mse(realized_vol.iloc[-n:] / 100, predict_svr_rbf / 100)
    # print('The MSE value of SVR with RBF Kernel is {:.7f}'.format(mse_svr_rbf))
    mse_dict['SVR-GARCH (RBF)'] = mse_svr_rbf
    mae_svr_rbf = mae(realized_vol.iloc[-n:] / 100, predict_svr_rbf / 100)
    # print('The MAE value of SVR with RBF Kernel is {:.6f}'.format(mae_svr_rbf))
    mae_dict['SVR-GARCH (RBF)'] = mae_svr_rbf
    svr_rbf_forecast_df = predict_svr_rbf / 100
    svr_rbf_forecast_df.columns = ['Preds']
    svr_rbf_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                          title='Volatility Prediction with SVR-GARCH (RBF)')
    svr_rbf_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                  legendgroup=newnames[t.name],
                                                  hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                  )
                               )
    svr_rbf_fig.add_trace(
        go.Scatter(x=svr_rbf_forecast_df.index, y=svr_rbf_forecast_df['Preds'], mode='lines', name='SVR-GARCH Predictions'))
    svr_rbf_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    t2.plotly_chart(svr_rbf_fig)

    u1, u2 = st.columns(2)

    # Neural Networks
    model = keras.Sequential(
        [layers.Dense(256, activation="relu"),
         layers.Dense(128, activation="relu"),
         layers.Dense(1, activation="linear"), ])
    model.compile(loss='mse', optimizer='rmsprop')
    epochs_trial = np.arange(100, 400, 4)
    batch_trial = np.arange(100, 400, 4)
    DL_pred = []
    DL_RMSE = []
    DL_MSE = []
    DL_MAE = []
    for i, j, k in zip(range(4), epochs_trial, batch_trial):
        model.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n - 1)].values.reshape(-1, ),
                  batch_size=k, epochs=j, verbose=False)
        DL_predict = model.predict(np.asarray(X.iloc[-n:]))
        DL_RMSE.append(np.sqrt(mse(realized_vol.iloc[-n:] / 100, DL_predict.flatten() / 100)))
        DL_MSE.append(mse(realized_vol.iloc[-n:] / 100, DL_predict.flatten() / 100))
        DL_MAE.append(mae(realized_vol.iloc[-n:] / 100, DL_predict.flatten() / 100))
        DL_pred.append(DL_predict)
        # print('DL_RMSE_{}:{:.6f}'.format(i+1, DL_RMSE[i]))
        # print('DL_MSE_{}:{:.6f}'.format(i+1, DL_MSE[i]))
        # print('DL_MAE_{}:{:.6f}'.format(i+1, DL_MAE[i]))
    DL_predict = pd.DataFrame(DL_pred[DL_RMSE.index(min(DL_RMSE))])
    DL_predict.index = ret.iloc[-n:].index
    rmse_dict['Neural Networks'] = min(DL_RMSE)
    mse_dict['Neural Networks'] = DL_MSE[DL_RMSE.index(min(DL_RMSE))]
    mae_dict['Neural Networks'] = DL_MAE[DL_RMSE.index(min(DL_RMSE))]
    nn_forecast_df = DL_predict / 100
    nn_forecast_df.columns = ['Preds']
    nn_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                     title='Volatility Prediction with Neural Networks')
    nn_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                             legendgroup=newnames[t.name],
                                             hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                             )
                          )
    nn_fig.add_trace(
        go.Scatter(x=nn_forecast_df.index, y=nn_forecast_df['Preds'], mode='lines', name='Neural Network Predictions'))
    nn_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    u1.plotly_chart(nn_fig)

    # HAR-RV Model
    rv_daily = ret.rolling(2).std()
    rv_weekly = ret.rolling(5).std()
    rv_monthly = ret.rolling(22).std()
    rv_df = pd.DataFrame()
    rv_df['Daily'] = rv_daily
    rv_df['Weekly'] = rv_weekly
    rv_df['Month'] = rv_monthly
    rv_df = rv_df.dropna()
    # rv_df
    rv_df["Target"] = rv_df["Daily"].shift(-1)  # We want to predict the RV of the next day.
    rv_df.dropna(inplace=True)
    # Scale the data
    rv_scaled = (rv_df - rv_df.min()) / (rv_df.max() - rv_df.min())
    # Add constant c
    rv_scaled = sm.add_constant(rv_scaled)
    # Split train and test sets
    X = rv_scaled.drop("Target", axis=1)
    y = rv_scaled["Target"]
    split = len(X) - n
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    results = sm.OLS(y_train, X_train).fit()
    # results.summary()
    # Perform out of sample prediction
    y_hat = results.predict(X_test)
    denormalized_y = y_hat * (max(y_test) - min(y_test)) + min(y_test)
    rmse_har_rv = np.sqrt(mse(realized_vol.iloc[-n:] / 100, denormalized_y))
    # print('The RMSE value of HAR-RV Model is  {:.6f}'.format(rmse_har_rv))
    rmse_dict['HAR-RV'] = rmse_har_rv
    mse_har_rv = mse(realized_vol.iloc[-n:] / 100, denormalized_y)
    # print('The MSE value of HAR-RV Model is  {:.6f}'.format(mse_har_rv))
    mse_dict['HAR-RV'] = mse_har_rv
    mae_har_rv = mae(realized_vol.iloc[-n:] / 100, denormalized_y)
    # print('The MAE value of HAR-RV Model is  {:.6f}'.format(mae_har_rv))
    mae_dict['HAR-RV'] = mae_har_rv
    har_rv_fig = px.line(realized_vol / 100, template='plotly_white', labels={'value': 'Volatility'},
                         title='Volatility Prediction with HAR-RV Model')
    har_rv_fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                 legendgroup=newnames[t.name],
                                                 hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                 )
                              )
    har_rv_fig.add_trace(go.Scatter(x=y_test.index, y=denormalized_y, mode='lines', name='HAR-RV Predictions'))
    har_rv_fig.update_layout(
        title_x=0.5,
        legend_title="Legend"
    )
    u2.plotly_chart(har_rv_fig)

    st.subheader('Evaluation Of All Models')

    metric_fig = go.Figure(data=[go.Table(header=dict(values=['Model', 'RMSE Value', 'MSE Value', 'MAE Value'],
                                                      fill_color='#0e0101',
                                                      font={'color': 'white'}),
                                          cells=dict(
                                              values=[list(rmse_dict.keys()),
                                                      [round(num, 7) for num in list(rmse_dict.values())],
                                                      [round(num, 7) for num in list(mse_dict.values())],
                                                      [round(num, 7) for num in list(mae_dict.values())]],
                                              fill_color='#D4d0d0'))
                                 ])
    st.plotly_chart(metric_fig)

    rmse_vals_list = list(rmse_dict.values())
    i = rmse_vals_list.index(min(rmse_vals_list))
    keys_list = list(rmse_dict)
    key = keys_list[i]
    st.text('The model with the best performance (lowest RMSE) is : {}'.format(key))
    time.sleep(5)
st.success('Done!')
