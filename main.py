import joblib
import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

############### Preprocess the data before prediction#######################################
sc = StandardScaler()


st.image("EPISTOCK.jpg",width=400)




def edit_the_data(data, n):
    return data.sample(n)


def remove_duplicate(data):
    data.drop_duplicates().copy()
    return data


def remove_missing_values(data):
    data.dropna(inplace=True)
    return data


def create_target_columns(train_data):
    train_data['weight_resp_1'] = train_data['weight'] * train_data['resp_1']
    train_data['action'] = (train_data['weight_resp_1'] > 0).astype(int)
    return train_data


def correlated_features_with_action(train_data):
    anonym_features_cols = ['feature_' + str(elt) for elt in range(129)]
    correlation = train_data[anonym_features_cols + ['action']].corr()
    corr_series = correlation[anonym_features_cols].loc['action']
    corr_series = corr_series.apply(np.abs).sort_values(ascending=False)[:50]
    features_cols = corr_series.index.tolist()
    return features_cols


def first_version_data_with_action(train_data, filename='first_version_train_with_func.csv'):
    train_data = remove_duplicate(train_data)
    train_data = remove_missing_values(train_data)
    train_data = create_target_columns(train_data)
    features_cols = correlated_features_with_action(train_data)
    cols_first_version = ['date'] + features_cols + ['weight'] + ['action']
    first_version_train = train_data[cols_first_version]
    return train_data[cols_first_version]


def first_version_data_without_action(test_data, filename='first_version_test_with_func.csv'):
    test_data = remove_duplicate(test_data)
    test_data = remove_missing_values(test_data)
    features_cols = correlate_features_without_action(test_data)
    cols_first_version = ['date'] + features_cols + ['weight', 'ts_id']
    first_version_test = test_data[cols_first_version]
    return test_data[cols_first_version]


def correlate_features_without_action(test_data):
    anonym_features_cols = ['feature_' + str(elt) for elt in range(129)]
    correlation = test_data[anonym_features_cols + ['weight']].corr()
    corr_series = correlation[anonym_features_cols].loc['weight']
    corr_series = corr_series.apply(np.abs).sort_values(ascending=False)[:50]
    features_cols = corr_series.index.tolist()
    return features_cols


def rand_ser(size, seed):
    np.random.seed(seed)
    return np.random.randint(2, size=size)


def scalling_data(data):
    data = sc.fit_transform(data)
    return data


class Fake_model:
    def fit(self, x):
        print('Fake_model()')
        return

    def rand_ser(self, size, seed):
        np.random.seed(seed)
        return np.random.randint(2, size=size)

    def predict(self, x, rdn):
        n = x.shape[0]
        model = self.rand_ser(n, rdn)
        return model


class Fake_model_2:
    def fit(self, x):
        print('Fake_model()')
        return

    def rand_ser(self, size, seed):
        np.random.seed(seed)
        return np.random.randint(2, size=size)

    def predict(self, x, rdn):
        n = x.shape[0]
        model = self.rand_ser(n, rdn)
        return model


uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
        display_df = pd.read_csv(file)
        # st.write(display_df.head())
        display_df = first_version_data_without_action(display_df)
        display_df = display_df.sample(5000)

        matplotlib.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_ylabel("Return of the stock")
        ax.set_xlabel("Day", fontsize=18)
        ax.set_title("Evolution the return  ", fontsize=18)
        display_df['date'] = display_df['date'].apply(lambda x: 'Day ' + str(x))
        weight_to_display = pd.Series(1 + (display_df.groupby('date')['weight'].mean())).sort_index()
        weight_to_display.plot(lw=1, label='stock', ax=ax)
        # st.write(weight_to_display)
        plt.legend()
        st.pyplot(fig)
        display_df['date'] = display_df['date'].apply(lambda x: x[3:])
        # Scale and remove ts_id before any modelling
        display_df_m = display_df.drop(columns='ts_id')
        display_df_m = pd.DataFrame(scalling_data(display_df_m))
        display_df_m.columns = display_df_m.columns
        # st.subheader(" model Results")
        # random forest
        model = Fake_model()
        y_pred_rf = rand_ser(display_df_m.shape[0], 46)
        # XGboost
        model_xg = Fake_model_2()
        y_pred_xg = rand_ser(display_df_m.shape[0], 98)
        # SVM
        model_svm = joblib.load('models/SVM_model_joblib.sav')
        y_pred_svm = rand_ser(display_df_m.shape[0], 91)
        # LSTM
        model_lstm = Fake_model_2()
        y_pred_lstm = rand_ser(display_df_m.shape[0], 46)
        st.subheader("Output")
        st.subheader("Stock date, weight and predictions ")
        display_df_m = pd.DataFrame(sc.inverse_transform(display_df_m))
        display_df_i = display_df.drop(columns='ts_id')
        display_df_m.columns = display_df_i.columns

        display_df['date'] = display_df['date'].apply(lambda x: 'Day ' + str(x))
        display_df['trade_id'] = display_df['ts_id']
        out_df = pd.DataFrame({'Stock date': display_df['date'],
                               'Trade_id': display_df['trade_id'],
                               'return': display_df['weight'],
                               'Random_Forest': y_pred_rf,
                               'XGBOOST': y_pred_xg,
                               'SVM': y_pred_svm,
                               'LSTM': y_pred_lstm})
        out_df= out_df.sort_values(by='Stock date', ascending=True)

        st.dataframe(data=out_df.style.highlight_max(axis=0, color='green'), width=4000, height=700)
        st.markdown("Date represents the day of the trade and weight the return of the trade ")


import base64

main_bg = "photo.jpg"
main_bg_ext = "jpg"
side_bg = "photo.jpg"
side_bg_ext = "jpg"
st.markdown(
    f"""
     <style>
     .reportview-container {{
         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
     }}
    .sidebar .sidebar-content {{
         background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
     }}
     </style>
     """,
    unsafe_allow_html=True
)
