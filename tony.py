import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.ensemble import GradientBoostingRegressor
# import xgboost
import shap
# from sklearn.model_selection import KFold
# import SessionState

@st.cache(persist = True)
def load_data():
    X = pd.read_csv("X.csv")
    X_scaled = pd.read_csv("X_scaled.csv")
    y = pd.read_csv("y.csv")
    X_train = X_scaled[X_scaled["reviews_per_month"] > 0]
    X_test = X_scaled[X_scaled["reviews_per_month"] <= 0]
    X_topup = X_test.sample(n=2928)
    X_train = pd.concat([X_scaled[X_scaled["reviews_per_month"] > 0], X_topup])
    X_test = X_test.drop(X_topup.index)
    y_train = y.loc[X_train.index, :]
    y_test = y.loc[X_test.index, :]
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    df = pd.concat([y, X], axis=1)
    df_test = pd.concat([y_test, X_test], axis=1)
    del X_topup
    return X, y, X_train, y_train, X_test, y_test, df, df_test
    
# @st.cache
# def load_gbm():
    # model_gb =  GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
    # explainer = shap.TreeExplainer(model_gb)
    # shap_values = explainer.shap_values(X_test)
    # return explainer, shap_values
# @st.cache(hash_funcs={'xgboost.XGBRegressor': id})
# def load_xgboost():
    # lr = 0.032
    # n_folds = 10
    # numerical_features =  X_train.select_dtypes(exclude=['object'])
    # kf = KFold(n_folds, shuffle=True, random_state = 2020).get_n_splits(numerical_features)
    # xgb_model = xgboost.XGBRegressor(objective ='reg:squarederror',\
                                            # subsample = 0.6, \
                                            # reg_lambda = 2.5, n_estimators  = 75, max_depth = 13, \
                                            # learning_rate = lr,\
                                            # gamma = 0, colsample_bytree = .9,\
                                            # early_stopping=5)
    # xgb_model.fit(X_train, y_train)
    # explainer = shap.TreeExplainer(xgb_model)
    # shap_values = explainer.shap_values(X_test)
    # return xgb_model

# def st_shap(plot, height=None):
    # shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    # components.html(shap_html, height=height)

st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/NTU%20Logo.png', width = 750)
st.write('''<h1 align=center><font color='Blue'>BC3409</font> - 
<font color='red'>AI in Accounting & Finance</font>''', unsafe_allow_html=True)

use_case = st.sidebar.selectbox("Select use cases", 
("GBM",
"XGBoost"
))

# with st.spinner(''):
_, _, X_train, y_train, X_test, y_test, _, df_test = load_data()
# session_state = SessionState.get(n=0)

if use_case == "GBM":
    st.write("<h2 align=center>Airbnb Optimal Price Recommender - {}</h2>".format(use_case), unsafe_allow_html=True)
    # with st.spinner('Do allow approximately 5s for GBM to be fitted.'):
    f = open("gbmBase.txt")
    gbmBaseValue = float(f.readlines()[0])
    f.close()
    gbmVals = np.load("gbmValues.npy")
    # st.success('Done!')
    # if st.button('Start Training'):
    # with st.spinner('Waiting for models to be fitted'):
        # model_gb = GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
        # explainer = shap.TreeExplainer(model_gb)
        # shap_values = explainer.shap_values(X_test)
    # features = st.radio(
     # "Which features are you interested in?",
    # ('Price Prediction Breakdown', 
    # 'All Prediction Breakdown', 
    # 'Feature Importance Summary', 
    # 'Feature Dependence Plot'))
    # if features == "Price Prediction Breakdown":
        # if st.button('Start Training'):
            # with st.spinner('Waiting for models to be fitted'):
                # model_gb = GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
                # explainer = shap.TreeExplainer(model_gb)
                # shap_values = explainer.shap_values(X_test)
    '''A key part to address the problem of this assignment is to advise property listers of an ‘optimum price’ value given a set of input variables in their pricing decisions. 
    This web app provides a peek at the high interpretability and explainability of using SHAP for both GBM and XGBoost. '''

    '###### *We will be using cached predictions to conserve CPU usage (Heroku dynos) and avoid long train/fitting time.*'
    
    "#### Reproducible code:"
    ''' 
    ~~~
    model_gb =  GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
    explainer = shap.TreeExplainer(model_gb)
    shap_values = explainer.shap_values(X_test)
    ~~~
    '''
    # with st.echo():
        # ''' 
            # model_gb =  GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())\
            # explainer = shap.TreeExplainer(model_gb)\
            # shap_values = explainer.shap_values(X_test)
        # '''
    "## SHAP Explainer"
    # placeholder1 = st.empty()
    # placeholder2 = st.empty()
    n = 0
    n = st.slider("Select test case:", 0, len(df_test), n)
    n = st.number_input("Alternatively, type desired row:", 0, len(df_test), n)
    # "**Original y-value:** ", y_test.iloc[n].values.tolist()[0]
    # "**Predicted y-hat:** ",  (explainer.expected_value + shap_values[n,:].sum())[0]
    # "**Residual MAE:**", abs(y_test.iloc[n].values.tolist()[0] - (explainer.expected_value + shap_values[n,:].sum())[0])
    # n = session_state.n
    results = pd.DataFrame({"Original y-value": y_test.iloc[n].values.tolist()[0], 
                  "Predicted y-hat": (gbmBaseValue + gbmVals[n,:].sum()),
                  "Residual MAE": abs(y_test.iloc[n].values.tolist()[0] - (gbmBaseValue + gbmVals[n,:].sum()))},
                  index = ["Test Case Result: {}".format(n)])
    results
    # st_shap(shap.force_plot(gbmBaseValue, gbmVals[n,:], X_test.iloc[n,:]))
    st.pyplot(shap.force_plot(gbmBaseValue, gbmVals[n,:], X_test.iloc[n,:]),bbox_inches='tight',dpi=300,pad_inches=0)
    "**Breakdown:**"
    breakdownCols = ["Base value"] + X_test.columns.tolist()
    breakdownVals = [gbmBaseValue] + list(gbmVals[n,:])
    breakdown = pd.DataFrame(breakdownVals, index = breakdownCols, columns = ["Value"])
    breakdown
    # "Base value:", explainer.expected_value[0]
    # for i, v in enumerate(shap_values[n,:]):
        # X_test.columns[i],":", v
    # if st.checkbox("Show Raw Data"):
    "## Raw Data"
    "**Sample dataset** (top 5 rows)"
    df_test.iloc[:6, :]
    "### Inspect Test Set"
    nrows = st.slider("Select number of rows:", 0, len(df_test))
    df_test.iloc[:nrows, :]
    # model_gb = load_gbm(X_train, y_train)
    # explainer = shap.TreeExplainer(model_gb)
    # shap_values = explainer.shap_values(X_test)

elif use_case == "XGBoost":
    st.write("<h2 align=center>Airbnb Optimal Price Recommender - {}</h2>".format(use_case), unsafe_allow_html=True)
    '''A key part to address the problem of this assignment is to advise property listers of an ‘optimum price’ value given a set of input variables in their pricing decisions. 
    This web app provides a peek at the high interpretability and explainability of using SHAP for both GBM and XGBoost. '''

    '###### *We will be using cached predictions to conserve CPU usage (Heroku dynos) and avoid long train/fitting time.*'
    # with st.spinner('Do allow approximately 60s for XGBoost model to be fitted.'):
        # lr = 0.032
        # n_folds = 10
        # numerical_features =  X_train.select_dtypes(exclude=['object'])
        # kf = KFold(n_folds, shuffle=True, random_state = 2020).get_n_splits(numerical_features)
        # xgb_model = xgboost.XGBRegressor(objective ='reg:squarederror',\
                                                # subsample = 0.6, \
                                                # reg_lambda = 2.5, n_estimators  = 75, max_depth = 13, \
                                                # learning_rate = lr,\
                                                # gamma = 0, colsample_bytree = .9,\
                                                # early_stopping=5)
        # xgb_model.fit(X_train, y_train)
        # explainer = shap.TreeExplainer(xgb_model)
        # shap_values = explainer.shap_values(X_test)
    f = open("xgboostBase.txt")
    xgbBaseValue = float(f.readlines()[0])
    f.close()
    xgbVals = np.load("xgboostValues.npy")
    st.success('Done!')
    "#### Reproducible code:"
    ''' 
    ~~~
    lr = 0.032
    n_folds = 10
    numerical_features =  X_train.select_dtypes(exclude=['object'])
    kf = KFold(n_folds, shuffle=True, random_state = 2020).get_n_splits(numerical_features)
    xgb_model = xgboost.XGBRegressor(objective ='reg:squarederror',\
                                            subsample = 0.6, \
                                            reg_lambda = 2.5, n_estimators  = 75, max_depth = 13, \
                                            learning_rate = lr,\
                                            gamma = 0, colsample_bytree = .9,\
                                            early_stopping=5)
    xgb_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    ~~~
    '''
    "## SHAP Explainer"
    n = 0
    n = st.slider("Select test case:", 0, len(df_test), n)
    n = st.number_input("Alternatively, type desired row:", 0, len(df_test), n)
    results = pd.DataFrame({"Original y-value": y_test.iloc[n].values.tolist()[0], 
                  "Predicted y-hat": (xgbBaseValue + xgbVals[n,:].sum()),
                  "Residual MAE": abs(y_test.iloc[n].values.tolist()[0] - (xgbBaseValue + xgbVals[n,:].sum()))},
                  index = ["Test Case Result: {}".format(n)])
    results
    st_shap(shap.force_plot(xgbBaseValue, xgbVals[n,:], X_test.iloc[n,:]))
    "**Breakdown:**"
    breakdownCols = ["Base value"] + X_test.columns.tolist()
    breakdownVals = [xgbBaseValue] + list(xgbVals[n,:])
    breakdown = pd.DataFrame(breakdownVals, index = breakdownCols, columns = ["Value"])
    breakdown
    "## Raw Data"
    "**Sample dataset** (top 5 rows)"
    df_test.iloc[:6, :]
    "### Inspect Test Set"
    nrows = st.slider("Select number of rows:", 0, len(df_test))
    df_test.iloc[:nrows, :]
    # number = st.select_slider(
        # 'Select a range of input  ',
        # options=["Sensor {}".format(i) for i in range(1,  31)],
        # value=("Sensor 5", "Sensor 10"))
        
    # inputVars = st.multiselect(
        # 'Further filter input variables:',
        # ["Sensor {}".format(i) for i in range(startInput,  endInput+1)],
        # ["Sensor {}".format(i) for i in range(startInput,  endInput+1)])
    
    # targetVars = st.multiselect(
        # 'Select target variables',
        # ["Pressure {}".format(i) for i in range(1,  3)],
        # ["Pressure {}".format(i) for i in range(1,  3)])

    # if st.checkbox("Plot Correlation Matrix"):
        # with st.spinner('Plotting matrix now...'):
            # start_time = time.time()
            # fig, ax = plt.subplots(figsize=(25, 25))
            # sns.heatmap(df[inputVars + targetVars].corr(),cmap="BrBG",annot=True, fmt = 'g')
            # "**Tip:** To observe graph better, click top-right, settings, wide mode."
            # st.pyplot(fig)
        # st.success('Done! Took {} seconds'.format(round(time.time()-start_time), 2))
        
    # if st.checkbox("Plot Scatterplot Matrix"):
        # with st.spinner('Plotting matrix now...'):
            # start_time = time.time()
            # fig, ax = plt.subplots(figsize=(25, 25))
            # fig = sns.pairplot(df[inputVars + targetVars], diag_kind="kde")
            # "**Tip:** To observe graph better, click top-right, settings, wide mode."
            # st.pyplot(fig)
        # st.success('Done! Took {} seconds'.format(round(time.time()-start_time), 2))

   


        



    # color = st.select_slider(
        # 'Select a color of the rainbow',
        # options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    # st.write('My favorite color is', color)
    # start_color, end_color = st.select_slider(
        # 'Select a range of color wavelength',
        # options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
        # value=('red', 'blue'))

    # number = st.number_input('Insert a number')
    # st.write('The current number is ', number)

# elif use_case == "Detecting anomaly (encoder)":
    # st.write("<h2 align=center>Attila Cybertech - Detecting anomaly (encoder)</h2>", unsafe_allow_html=True)
    # targetVariable = st.sidebar.selectbox("Select target variable", 
    # ("Pressure 1", "Pressure 2"))

    # unsupervised_model = st.radio(
     # "Which model are you interested in?",
    # ('Convolutional', 'LSTM'))

    # TIME_STEPS = 120
    # training_mean = df["Pressure 1"].mean()
    # training_std = df["Pressure 1"].std()
    # df_training_value = (df["Pressure 1"] - training_mean) / training_std
    # def create_sequences(values, time_steps=TIME_STEPS):
    #     output = []
    #     for i in range(len(values) - time_steps):
    #         output.append(values[i : (i + time_steps)])
    #     return np.expand_dims(np.stack(output), axis = 2)
    # X_train = create_sequences(df_training_value.values)

    
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

myWatermark = """
            <style>
            footer:after {
            content:'Tony Ng'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            }
            </style>
            """
st.markdown(myWatermark, unsafe_allow_html=True)