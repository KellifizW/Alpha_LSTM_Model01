import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from datetime import datetime, timedelta
import pytz
import pickle
import io
import os
import tempfile

eastern = pytz.timezone('US/Eastern')
current_date = datetime.now(eastern)
st.write(f"當前 TensorFlow 版本: {tf.__version__}，今日美國東部時間: {current_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")

st.markdown("""
### 使用說明
- **訓練模式**：選擇「訓練模式」，輸入股票代碼、時間步長、次數等參數，然後點擊「運行分析」訓練模型並生成預測與回測結果。訓練完成後可下載模型和縮放器。
- **預測模式**：選擇「預測模式」，上載之前訓練好的模型和縮放器文件，輸入股票代碼與日期範圍，點擊「運行預測」生成歷史與未來價格預測。
- **還原狀態**：點擊「還原狀態」清除當前結果並重置進度。
""")

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_h = self.add_weight(name='W_h', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b_h = self.add_weight(name='b_h', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.W_a = self.add_weight(name='W_a', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        h_transformed = K.tanh(K.dot(inputs, self.W_h) + self.b_h)
        e = K.dot(h_transformed, self.W_a)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        output = K.sum(context, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_model(input_shape, model_type="original", learning_rate=0.001):
    if model_type == "lstm_simple":
        model = Sequential()
        model.add(LSTM(150, activation='relu', input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.01))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    else:
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=128, kernel_size=1, activation='relu', padding='same')(inputs)
        x = Bidirectional(LSTM(units=128, activation='tanh', return_sequences=True))(x)
        x = Dropout(0.01)(x)
        x = Attention()(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

def preprocess_data(data, timesteps, predict_days=1, train_split_ratio=0.7, scaler_features=None, scaler_target=None, is_training=True):
    if data.empty:
        raise ValueError("輸入數據為空，無法進行預處理。請檢查數據來源或日期範圍。")
    st.write(f"原始數據長度：{len(data)}")
    data = data.copy()
    data['Yesterday_Close'] = data['Close'].shift(1)
    data['Average'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data.dropna()
    st.write(f"移除 NaN 後數據長度：{len(data)}")
    if len(data) < timesteps + predict_days - 1:
        raise ValueError(f"數據樣本數 ({len(data)}) 小於時間步長加預測天數 ({timesteps + predict_days - 1})，無法生成有效輸入。\n"
                         f"請選擇更長的歷史數據年限（當前為 {len(data)} 個交易日），或減少時間步長（當前為 {timesteps}）或預測天數（當前為 {predict_days}）。")
    features = ['Yesterday_Close', 'Open', 'High', 'Low', 'Average']
    target = 'Close'
    if is_training:
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()
        scaled_features = scaler_features.fit_transform(data[features])
        scaled_target = scaler_target.fit_transform(data[[target]])
    else:
        if scaler_features is None or scaler_target is None:
            raise ValueError("預測模式需要提供訓練時的 scaler_features 和 scaler_target。")
        scaled_features = scaler_features.transform(data[features])
        scaled_target = scaler_target.transform(data[[target]])
    total_samples = len(scaled_features) - timesteps - predict_days + 1
    if total_samples <= 0:
        raise ValueError(f"數據樣本數不足以生成時間序列，總樣本數: {len(scaled_features)}，時間步長: {timesteps}，預測天數: {predict_days}")
    X, y, y_current = [], [], []
    for i in range(total_samples):
        X.append(scaled_features[i:i + timesteps])
        y.append(scaled_target[i + timesteps + predict_days - 1])
        y_current.append(scaled_target[i + timesteps - 1])
    X = np.array(X)
    y = np.array(y)
    y_current = np.array(y_current)
    if is_training:
        data_index = data.index
        train_size = int(total_samples * train_split_ratio)
        test_size = total_samples - train_size
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        y_current_train = y_current[:train_size]
        y_current_test = y_current[train_size:]
        # 調整 test_dates 計算方式，確保包含最後的日期
        test_dates = data_index[timesteps + train_size - 1 + predict_days:timesteps + train_size + test_size + predict_days - 1]
        return X_train, X_test, y_train, y_test, y_current_train, y_current_test, scaler_features, scaler_target, test_dates, data
    else:
        data_index = data.index
        test_dates = data_index[timesteps + predict_days - 1:timesteps + total_samples + predict_days - 1]
        return X, y, y_current, test_dates, data

@tf.function(reduce_retracing=True)
def predict_step(model, x):
    return model(x, training=False)

def backtest(data, predictions, test_dates, period_start, period_end, initial_capital=100000):
    data = data.copy()
    test_size = len(predictions)
    data['Predicted'] = np.nan
    if len(test_dates) != len(predictions):
        raise ValueError(f"test_dates 長度 ({len(test_dates)}) 與 predictions 長度 ({len(predictions)}) 不一致！")
    data.loc[test_dates, 'Predicted'] = predictions.flatten()
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    test_mask = data.index.isin(test_dates)
    data.loc[test_mask, 'EMA12_pred'] = pd.Series(predictions.flatten(), index=test_dates).ewm(span=12, adjust=False).mean()
    data.loc[test_mask, 'EMA26_pred'] = pd.Series(predictions.flatten(), index=test_dates).ewm(span=26, adjust=False).mean()
    data['MACD_pred'] = data['EMA12_pred'] - data['EMA26_pred']
    data['Signal_pred'] = data['MACD_pred'].ewm(span=9, adjust=False).mean()
    position_pred = 0
    capital_pred = initial_capital
    shares_pred = 0
    capital_values_pred = []
    buy_signals_pred = []
    sell_signals_pred = []
    golden_cross_pred = []
    death_cross_pred = []
    position_actual = 0
    capital_actual = initial_capital
    shares_actual = 0
    capital_values_actual = []
    buy_signals_actual = []
    sell_signals_actual = []
    golden_cross_actual = []
    death_cross_actual = []
    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end)
    mask = (test_dates >= period_start) & (test_dates <= period_end)
    filtered_dates = test_dates[mask]
    if len(filtered_dates) == 0:
        st.error("回測時段不在測試數據範圍內！")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    test_start_idx = data.index.get_loc(filtered_dates[0])
    test_end_idx = data.index.get_loc(filtered_dates[-1])
    capital_values_pred = [initial_capital] * test_start_idx
    capital_values_actual = [initial_capital] * test_start_idx
    for i in range(test_start_idx, test_end_idx + 1):
        close_price = data['Close'].iloc[i].item()
        macd_pred = data['MACD_pred'].iloc[i]
        signal_pred = data['Signal_pred'].iloc[i]
        macd = data['MACD'].iloc[i]
        signal = data['Signal'].iloc[i]
        if i > test_start_idx:
            prev_macd_pred = data['MACD_pred'].iloc[i - 1]
            prev_signal_pred = data['Signal_pred'].iloc[i - 1]
            if macd_pred > signal_pred and prev_macd_pred <= prev_signal_pred:
                golden_cross_pred.append((data.index[i], macd_pred))
                if position_pred == 0:
                    shares_pred = capital_pred // close_price
                    capital_pred -= shares_pred * close_price
                    position_pred = 1
                    buy_signals_pred.append((data.index[i], close_price))
            elif macd_pred < signal_pred and prev_macd_pred >= prev_signal_pred:
                death_cross_pred.append((data.index[i], macd_pred))
                if position_pred == 1:
                    capital_pred += shares_pred * close_price
                    position_pred = 0
                    shares_pred = 0
                    sell_signals_pred.append((data.index[i], close_price))
        total_value_pred = capital_pred + (shares_pred * close_price if position_pred > 0 else 0)
        capital_values_pred.append(total_value_pred)
        if i > test_start_idx:
            prev_macd = data['MACD'].iloc[i - 1]
            prev_signal = data['Signal'].iloc[i - 1]
            if macd > signal and prev_macd <= prev_signal:
                golden_cross_actual.append((data.index[i], macd))
                if position_actual == 0:
                    shares_actual = capital_actual // close_price
                    capital_actual -= shares_actual * close_price
                    position_actual = 1
                    buy_signals_actual.append((data.index[i], close_price))
            elif macd < signal and prev_macd >= prev_signal:
                death_cross_actual.append((data.index[i], macd))
                if position_actual == 1:
                    capital_actual += shares_actual * close_price
                    position_actual = 0
                    shares_actual = 0
                    sell_signals_actual.append((data.index[i], close_price))
        total_value_actual = capital_actual + (shares_actual * close_price if position_actual > 0 else 0)
        capital_values_actual.append(total_value_actual)
    capital_values_pred = np.array(capital_values_pred)
    total_return_pred = (capital_values_pred[-1] / capital_values_pred[0] - 1) * 100
    max_return_pred = (max(capital_values_pred) / capital_values_pred[0] - 1) * 100
    min_return_pred = (min(capital_values_pred) / capital_values_pred[0] - 1) * 100
    capital_values_actual = np.array(capital_values_actual)
    total_return_actual = (capital_values_actual[-1] / capital_values_actual[0] - 1) * 100
    max_return_actual = (max(capital_values_actual) / capital_values_actual[0] - 1) * 100
    min_return_actual = (min(capital_values_actual) / capital_values_actual[0] - 1) * 100
    return (
        data,
        capital_values_pred, total_return_pred, max_return_pred, min_return_pred,
        buy_signals_pred, sell_signals_pred, golden_cross_pred, death_cross_pred,
        capital_values_actual, total_return_actual, max_return_actual, min_return_actual,
        buy_signals_actual, sell_signals_actual, golden_cross_actual, death_cross_actual
    )

@st.cache_data
def fetch_stock_data(stock_symbol, start_date, end_date):
    st.write(f"正在下載 {stock_symbol} 的數據，日期範圍：{start_date} 至 {end_date}")
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        st.error(f"無法下載 {stock_symbol} 的數據（{start_date} 至 {end_date}）。請檢查股票代碼或日期範圍是否有效！")
        return None
    data.index = pd.to_datetime(data.index).tz_localize(eastern)
    st.write(f"成功下載數據，總共 {len(data)} 個交易日")
    return data

def train_model(model, X_train, y_train, epochs, _callbacks=None):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, validation_split=0.1, verbose=1, callbacks=_callbacks)
    return model, history.history

def create_price_comparison_chart(dates, actual_prices, predicted_prices, stock_symbol, title, buy_signals_pred=None, sell_signals_pred=None, buy_signals_actual=None, sell_signals_actual=None):
    # 過濾掉 NaN 值，只保留有數據的交易日
    valid_mask = ~np.isnan(actual_prices)  # 假設 actual_prices 和 predicted_prices 的 NaN 位置一致
    valid_dates = dates[valid_mask]
    valid_actual_prices = actual_prices[valid_mask]
    valid_predicted_prices = predicted_prices[valid_mask]

    if len(valid_dates) == 0:
        st.error("數據長度為0，無法繪製圖表！請檢查數據過濾或預測過程。")
        return go.Figure()

    # 將日期轉換為字符串格式
    date_labels = [pd.Timestamp(d).strftime('%Y-%m-%d') if isinstance(d, (pd.Timestamp, datetime)) else d for d in valid_dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid_dates,
        y=valid_actual_prices,
        mode='lines',
        name='實際股價',
        line=dict(color='#1f77b4'),
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=valid_dates,
        y=valid_predicted_prices,
        mode='lines',
        name='預測股價',
        line=dict(color='#ff7f0e', dash='dash'),
        connectgaps=True
    ))

    if buy_signals_pred:
        buy_dates_pred, buy_prices_pred = zip(*buy_signals_pred)
        fig.add_trace(go.Scatter(
            x=buy_dates_pred,
            y=buy_prices_pred,
            mode='markers',
            name='預測 MACD 買入',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
    if sell_signals_pred:
        sell_dates_pred, sell_prices_pred = zip(*sell_signals_pred)
        fig.add_trace(go.Scatter(
            x=sell_dates_pred,
            y=sell_prices_pred,
            mode='markers',
            name='預測 MACD 賣出',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))

    if buy_signals_actual:
        buy_dates_actual, buy_prices_actual = zip(*buy_signals_actual)
        fig.add_trace(go.Scatter(
            x=buy_dates_actual,
            y=buy_prices_actual,
            mode='markers',
            name='實際 MACD 買入',
            marker=dict(symbol='triangle-up', size=10, color='limegreen')
        ))
    if sell_signals_actual:
        sell_dates_actual, sell_prices_actual = zip(*sell_signals_actual)
        fig.add_trace(go.Scatter(
            x=sell_dates_actual,
            y=sell_prices_actual,
            mode='markers',
            name='實際 MACD 賣出',
            marker=dict(symbol='triangle-down', size=10, color='darkred')
        ))

    # 設置 x 軸刻度，顯示每 5 個交易日的日期
    tick_indices = np.arange(0, len(valid_dates), 5)  # 每 5 個交易日顯示一個刻度
    tick_dates = [date_labels[i] for i in tick_indices]

    fig.update_layout(
        title=f'{stock_symbol} - {title}',
        xaxis_title='日期',
        yaxis_title='價格 (USD)',
        hovermode='x unified',
        xaxis=dict(
            tickmode='array',
            tickvals=[valid_dates[i] for i in tick_indices],  # 使用交易日的日期作為刻度值
            ticktext=tick_dates,  # 顯示對應的日期標籤
            tickformat='%Y-%m-%d',
            tickangle=45,
            type='date',  # 確保 x 軸按日期格式處理
            rangeslider=dict(visible=False)
        ),
        template='plotly_white',
        height=600,
        width=1000,
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(args=[{"visible": [True, True, True, True, True, True]}], label="顯示全部", method="update"),
                    dict(args=[{"visible": [True, False, False, False, False, False]}], label="僅實際", method="update"),
                    dict(args=[{"visible": [False, True, False, False, False, False]}], label="僅預測", method="update")
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    return fig

def main():
    st.title("股票價格預測與回測系統 BETA")
    st.write(f"當前 Streamlit 版本: {st.__version__}")

    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'training_progress' not in st.session_state:
        st.session_state['training_progress'] = 40

    mode = st.sidebar.selectbox("選擇模式", ["訓練模式", "預測模式"])

    plotly_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart',
            'height': 600,
            'width': 1000,
            'scale': 1
        },
        'displaylogo': False,
        'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
    }

    if mode == "訓練模式":
        st.markdown("### 訓練模式\n輸入參數並訓練模型，生成預測和回測結果。訓練完成後可下載模型和縮放器。")
        stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL）", value="TSLA")
        timesteps = st.slider("選擇時間步長（歷史數據窗口天數）", min_value=10, max_value=100, value=30, step=10)
        epochs = st.slider("選擇訓練次數（epochs）", min_value=50, max_value=200, value=200, step=50)
        model_type = st.selectbox("選擇模型類型", ["original (CNN-BiLSTM-Attention)", "lstm_simple (單層LSTM 150神經元)"], index=0)
        predict_days = st.selectbox("選擇預測天數（預測未來幾天的收盤價）", [1, 3, 5, 7], index=1)
        data_years = st.selectbox("選擇下載之歷史數據年限", [1, 2, 3], index=2, help="從回測時段結束日期向前倒退的下載數據年限")
        train_test_split = st.selectbox("選擇訓練/測試數據分割比例", ["80%訓練/20%測試", "70%訓練/30%測試"], index=0)
        train_split_ratio = 0.8 if train_test_split.startswith("80%") else 0.7
        learning_rate_options = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
        selected_learning_rate = st.selectbox("選擇 Adam 學習率", options=learning_rate_options, index=3, format_func=lambda x: f"{x:.5f}")
        eastern = pytz.timezone('US/Eastern')
        current_date = datetime.now(eastern).replace(hour=0, minute=0, second=0, microsecond=0)
        periods = []
        end_base_date = current_date
        start_base_date = eastern.localize(datetime(2022, 3, 21))
        while end_base_date > start_base_date:
            period_start = end_base_date - timedelta(days=179)
            if period_start < start_base_date:
                period_start = start_base_date
            periods.append(f"{period_start.strftime('%Y-%m-%d')} to {end_base_date.strftime('%Y-%m-%d')}")
            end_base_date = period_start - timedelta(days=1)
        if not periods:
            st.error("無法生成回測時段選項！請檢查日期範圍設置。")
            return
        period_options = periods[::-1]
        selected_period = st.selectbox("選擇回測時段（6個月）", period_options, index=len(period_options) - 1)

        if st.button("運行分析") and st.session_state['results'] is None:
            start_time = time.time()
            with st.spinner("正在下載數據並訓練模型，請等待..."):
                period_start_str, period_end_str = selected_period.split(" to ")
                period_start = eastern.localize(datetime.strptime(period_start_str, "%Y-%m-%d"))
                period_end = eastern.localize(datetime.strptime(period_end_str, "%Y-%m-%d"))
                data_start = period_end - timedelta(days=365 * data_years)
                data_end = period_end + timedelta(days=1)
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("步驟 1/5: 下載數據...")
                data = fetch_stock_data(stock_symbol, data_start, data_end)
                if data is None:
                    return
                if len(data) < timesteps + predict_days - 1:
                    st.error(
                        f"下載的數據樣本數 ({len(data)}) 小於時間步長加預測天數 ({timesteps + predict_days - 1})。\n"
                        f"請選擇更長的下載歷史數據年限（當前為 {data_years} 年），或減少時間步長（當前為 {timesteps}）或預測天數（當前為 {predict_days}）。")
                    return
                data_temp = data.copy()
                data_temp['Yesterday_Close'] = data_temp['Close'].shift(1)
                data_temp = data_temp.dropna()
                if len(data_temp) < timesteps + predict_days - 1:
                    st.error(
                        f"下載的數據樣本數 ({len(data_temp)}) 小於時間步長加預測天數 ({timesteps + predict_days - 1})。\n"
                        f"請選擇更長的下載歷史數據年限（當前為 {data_years} 年），或減少時間步長（當前為 {timesteps}）或預測天數（當前為 {predict_days}）。")
                    return
                original_data = data.copy()
                actual_start_date = data.index[0].strftime('%Y-%m-%d')
                actual_end_date = data.index[-1].strftime('%Y-%m-%d')
                total_trading_days = len(data)
                if 'Close' not in data.columns:
                    st.error(f"數據中缺少 'Close' 列，無法計算統計特性。數據列: {data.columns.tolist()}")
                    return
                try:
                    close_values = np.asarray(data['Close']).flatten()
                    close_series = pd.Series(close_values, index=data.index)
                    daily_returns = close_series.pct_change().dropna()
                    if daily_returns.empty:
                        raise ValueError("日收益率數據為空，無法計算統計特性")
                    if daily_returns.ndim > 1:
                        daily_returns = daily_returns.squeeze()
                    volatility = daily_returns.std()
                    mean_return = daily_returns.mean()
                    autocorrelation = daily_returns.autocorr()
                except Exception as e:
                    volatility = mean_return = autocorrelation = "N/A"
                    st.warning(f"警告：無法計算統計特性，錯誤: {str(e)}")
                progress_bar.progress(20)
                status_text.text("步驟 2/5: 預處理數據...")
                X_train, X_test, y_train, y_test, y_current_train, y_current_test, scaler_features, scaler_target, test_dates, full_data = preprocess_data(
                    data, timesteps, predict_days, train_split_ratio, is_training=True
                )
                total_samples = len(X_train) + len(X_test)
                train_samples = len(X_train)
                test_samples = len(X_test)
                train_date_range = f"{full_data.index[timesteps].strftime('%Y-%m-%d')} to {full_data.index[timesteps + train_samples - 1].strftime('%Y-%m-%d')}"
                test_date_range = f"{test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}"
                progress_bar.progress(40)
                status_text.text("步驟 3/5: 訓練模型...")
                model_type_selected = "original" if model_type.startswith("original") else "lstm_simple"
                model = build_model(input_shape=(timesteps, X_train.shape[2]), model_type=model_type_selected,
                                    learning_rate=selected_learning_rate)
                total_params = model.count_params()
                st.write(f"模型參數總數: {total_params:,}")
                st.subheader("運算記錄")
                st.write(
                    f"正在下載的股票歷史數據日期範圍: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")
                st.write(f"實際已下載的數據範圍: {actual_start_date} to {actual_end_date}")
                st.write(f"總共交易日: {total_trading_days}")
                st.write(f"總樣本數: {total_samples}")
                st.write(f"訓練樣本數: {train_samples}")
                st.write(f"測試樣本數: {test_samples}")
                st.write(f"訓練數據範圍: {train_date_range}")
                st.write(f"測試數據範圍: {test_date_range}")
                mean_display = f"{mean_return:.6f}" if isinstance(mean_return, (int, float)) else mean_return
                volatility_display = f"{volatility:.6f}" if isinstance(volatility, (int, float)) else volatility
                autocorrelation_display = f"{autocorrelation:.6f}" if isinstance(autocorrelation,
                                                                                 (int, float)) else autocorrelation
                st.write(
                    f"數據統計特性 - 日收益率均值: {mean_display}, 波動率: {volatility_display}, 自相關係數: {autocorrelation_display}")
                progress_per_epoch = 20 / epochs

                def update_progress(epoch, logs):
                    st.session_state['training_progress'] = min(60, st.session_state[
                        'training_progress'] + progress_per_epoch)
                    progress_bar.progress(int(st.session_state['training_progress']))
                    status_text.text(f"步驟 3/5: 訓練模型 - Epoch {epoch + 1}/{epochs} (損失: {logs.get('loss'):.4f})")

                callback = [LambdaCallback(on_epoch_end=update_progress)]
                model, history = train_model(model, X_train, y_train, epochs, _callbacks=callback)
                progress_bar.progress(60)
                status_text.text("步驟 4/5: 進行價格預測...")
                X_test, y_test, y_current_test, test_dates, full_data = preprocess_data(
                    original_data, timesteps, predict_days, train_split_ratio, scaler_features, scaler_target,
                    is_training=False
                )
                predictions = predict_step(model, X_test)
                # 插入的驗證代碼：計算預測值與 y_current_test 和 y_test 的 MAE
                print(f"MAE with y_current: {mean_absolute_error(y_current_test, predictions)}")
                print(f"MAE with y: {mean_absolute_error(y_test, predictions)}")
                # 繼續反向縮放
                predictions = scaler_target.inverse_transform(predictions)
                y_current_test = scaler_target.inverse_transform(y_current_test)
                progress_bar.progress(80)
                status_text.text("步驟 5/5: 執行回測...")
                result = backtest(full_data, predictions, test_dates, period_start, period_end)
                if result[0] is None:
                    return
                (
                    full_data,
                    capital_values_pred, total_return_pred, max_return_pred, min_return_pred,
                    buy_signals_pred, sell_signals_pred, golden_cross_pred, death_cross_pred,
                    capital_values_actual, total_return_actual, max_return_actual, min_return_actual,
                    buy_signals_actual, sell_signals_actual, golden_cross_actual, death_cross_actual
                ) = result
                end_time = time.time()
                elapsed_time = end_time - start_time
                test_dates = test_dates
                period_start = pd.to_datetime(period_start)
                period_end = pd.to_datetime(period_end)

                # 創建完整的日期範圍（包括非交易日）
                full_date_range = pd.date_range(start=period_start, end=period_end, freq='D', tz=eastern)
                # 過濾 test_dates，確保包含在回測時段內的日期
                mask = (test_dates >= period_start) & (test_dates <= period_end)
                filtered_dates = test_dates[mask]
                if len(filtered_dates) == 0:
                    st.error(
                        f"過濾後的日期範圍為空！請檢查回測時段 ({period_start} 到 {period_end}) 是否在測試數據範圍 ({test_dates[0]} 到 {test_dates[-1]}) 內。")
                    return

                # 獲取實際價格和預測價格
                filtered_y_test = full_data.loc[filtered_dates, 'Close'].values
                filtered_predictions = predictions[mask]
                if filtered_predictions.ndim > 1:
                    filtered_predictions = filtered_predictions.flatten()

                # 將 filtered_dates 擴展到完整的日期範圍
                extended_dates = full_date_range
                extended_y_test = np.full(len(extended_dates), np.nan)
                extended_predictions = np.full(len(extended_dates), np.nan)

                # 將 filtered_dates 的數據映射到 extended_dates
                date_to_idx = {d.date(): i for i, d in enumerate(extended_dates)}
                for i, date in enumerate(filtered_dates):
                    idx = date_to_idx.get(date.date())
                    if idx is not None:
                        extended_y_test[idx] = filtered_y_test[i]
                        extended_predictions[idx] = filtered_predictions[i]

                # 調整 buy_signals 和 sell_signals 的日期，使其匹配 extended_dates
                def adjust_signals(signals, date_to_idx):
                    adjusted_signals = []
                    for date, price in signals:
                        idx = date_to_idx.get(date.date())
                        if idx is not None:
                            adjusted_signals.append((extended_dates[idx], price))
                    return adjusted_signals

                buy_signals_pred = adjust_signals(buy_signals_pred, date_to_idx)
                sell_signals_pred = adjust_signals(sell_signals_pred, date_to_idx)
                buy_signals_actual = adjust_signals(buy_signals_actual, date_to_idx)
                sell_signals_actual = adjust_signals(sell_signals_actual, date_to_idx)

                # 繪製圖表
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Training Loss'))
                fig_loss.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'))
                fig_loss.update_layout(title='訓練與驗證損失曲線', xaxis_title='Epoch', yaxis_title='Loss', height=400,
                                       width=600)

                data_backtest = full_data.loc[period_start:period_end].copy()
                golden_x_pred, golden_y_pred = zip(*golden_cross_pred) if golden_cross_pred else ([], [])
                death_x_pred, death_y_pred = zip(*death_cross_pred) if death_cross_pred else ([], [])
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data_backtest.index, y=data_backtest['MACD_pred'], mode='lines',
                                              name='MACD Line (Predicted)'))
                fig_macd.add_trace(go.Scatter(x=data_backtest.index, y=data_backtest['Signal_pred'], mode='lines',
                                              name='Signal Line (Predicted)'))
                fig_macd.add_trace(
                    go.Scatter(x=[data_backtest.index[0], data_backtest.index[-1]], y=[0, 0], mode='lines',
                               name='Zero Line', line=dict(dash='dash')))
                fig_macd.add_trace(go.Scatter(x=golden_x_pred, y=golden_y_pred, mode='markers', name='Golden Cross',
                                              marker=dict(symbol='circle', size=10, color='green')))
                fig_macd.add_trace(go.Scatter(x=death_x_pred, y=death_y_pred, mode='markers', name='Death Cross',
                                              marker=dict(symbol='circle', size=10, color='red')))
                fig_macd.update_layout(title=f'{stock_symbol} MACD Analysis (Predicted) ({selected_period})',
                                       xaxis_title='Date', yaxis_title='MACD Value', height=400, width=600)

                fig_macd_compare = go.Figure()
                fig_macd_compare.add_trace(go.Scatter(x=data_backtest.index, y=data_backtest['MACD_pred'], mode='lines',
                                                      name='MACD (Predicted)'))
                fig_macd_compare.add_trace(
                    go.Scatter(x=data_backtest.index, y=data_backtest['Signal_pred'], mode='lines',
                               name='Signal (Predicted)'))
                fig_macd_compare.add_trace(
                    go.Scatter(x=data_backtest.index, y=data_backtest['MACD'], mode='lines', name='MACD (Actual)',
                               line=dict(dash='dash')))
                fig_macd_compare.add_trace(
                    go.Scatter(x=data_backtest.index, y=data_backtest['Signal'], mode='lines', name='Signal (Actual)',
                               line=dict(dash='dash')))
                fig_macd_compare.add_trace(
                    go.Scatter(x=[data_backtest.index[0], data_backtest.index[-1]], y=[0, 0], mode='lines',
                               name='Zero Line', line=dict(dash='dot')))
                golden_x_actual, golden_y_actual = zip(*golden_cross_actual) if golden_cross_actual else ([], [])
                death_x_actual, death_y_actual = zip(*death_cross_actual) if death_cross_actual else ([], [])
                fig_macd_compare.add_trace(
                    go.Scatter(x=golden_x_pred, y=golden_y_pred, mode='markers', name='Golden Cross (Pred)',
                               marker=dict(symbol='circle', size=10, color='green')))
                fig_macd_compare.add_trace(
                    go.Scatter(x=death_x_pred, y=death_y_pred, mode='markers', name='Death Cross (Pred)',
                               marker=dict(symbol='circle', size=10, color='red')))
                fig_macd_compare.add_trace(
                    go.Scatter(x=golden_x_actual, y=golden_y_actual, mode='markers', name='Golden Cross (Actual)',
                               marker=dict(symbol='triangle-up', size=10, color='green')))
                fig_macd_compare.add_trace(
                    go.Scatter(x=death_x_actual, y=death_y_actual, mode='markers', name='Death Cross (Actual)',
                               marker=dict(symbol='triangle-down', size=10, color='red')))
                fig_macd_compare.update_layout(
                    title=f'{stock_symbol} MACD Comparison (Predicted vs Actual) ({selected_period})',
                    xaxis_title='Date', yaxis_title='MACD Value', height=400, width=600)

                fig_price = create_price_comparison_chart(
                    extended_dates, extended_y_test, extended_predictions, stock_symbol,
                    f"實際 vs 預測股價比較 ({selected_period})",
                    buy_signals_pred=buy_signals_pred, sell_signals_pred=sell_signals_pred,
                    buy_signals_actual=buy_signals_actual, sell_signals_actual=sell_signals_actual
                )

                mae = mean_absolute_error(filtered_y_test, filtered_predictions)
                rmse = np.sqrt(mean_squared_error(filtered_y_test, filtered_predictions))
                r2 = r2_score(filtered_y_test, filtered_predictions)
                mape = np.mean(np.abs((filtered_y_test - filtered_predictions) / filtered_y_test)) * 100

                st.session_state['results'] = {
                    'model': model,
                    'scaler_features': scaler_features,
                    'scaler_target': scaler_target,
                    'fig_loss': fig_loss,
                    'fig_macd': fig_macd,
                    'fig_macd_compare': fig_macd_compare,
                    'fig_price': fig_price,
                    'capital_values_pred': capital_values_pred,
                    'total_return_pred': total_return_pred,
                    'max_return_pred': max_return_pred,
                    'min_return_pred': min_return_pred,
                    'buy_signals_pred': buy_signals_pred,
                    'sell_signals_pred': sell_signals_pred,
                    'capital_values_actual': capital_values_actual,
                    'total_return_actual': total_return_actual,
                    'max_return_actual': max_return_actual,
                    'min_return_actual': min_return_actual,
                    'buy_signals_actual': buy_signals_actual,
                    'sell_signals_actual': sell_signals_actual,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'elapsed_time': elapsed_time,
                    'stock_symbol': stock_symbol,
                    'selected_period': selected_period,
                    'history': history
                }

                progress_bar.progress(100)
                status_text.text("分析完成！")

        if st.session_state['results'] is not None:
            results = st.session_state['results']
            stock_symbol = results['stock_symbol']
            selected_period = results['selected_period']

            st.subheader("下載訓練結果")
            temp_model_path = "temp_model.keras"
            results['model'].save(temp_model_path)
            with open(temp_model_path, "rb") as f:
                model_buffer = io.BytesIO(f.read())
            st.download_button(label="下載訓練好的模型", data=model_buffer, file_name=f"{stock_symbol}_lstm_model.keras", mime="application/octet-stream")
            os.remove(temp_model_path)
            scaler_features_buffer = io.BytesIO()
            pickle.dump(results['scaler_features'], scaler_features_buffer)
            scaler_features_buffer.seek(0)
            st.download_button(label="下載特徵縮放器", data=scaler_features_buffer, file_name=f"{stock_symbol}_scaler_features.pkl", mime="application/octet-stream")
            scaler_target_buffer = io.BytesIO()
            pickle.dump(results['scaler_target'], scaler_target_buffer)
            scaler_target_buffer.seek(0)
            st.download_button(label="下載目標縮放器", data=scaler_target_buffer, file_name=f"{stock_symbol}_scaler_target.pkl", mime="application/octet-stream")

            st.subheader(f"{stock_symbol} 分析結果（{selected_period}）")
            st.subheader("圖表分析")
            st.plotly_chart(results['fig_price'], use_container_width=True, config=plotly_config)
            st.plotly_chart(results['fig_loss'], use_container_width=True, config=plotly_config)
            st.plotly_chart(results['fig_macd'], use_container_width=True, config=plotly_config)
            st.plotly_chart(results['fig_macd_compare'], use_container_width=True, config=plotly_config)

            st.subheader("回測結果比較")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**基於預測 MACD 的策略**")
                st.write(f"初始資金: $100,000")
                st.write(f"最終資金: ${results['capital_values_pred'][-1]:.2f}")
                st.write(f"總回報率: {results['total_return_pred']:.2f}%")
                st.write(f"最大回報率: {results['max_return_pred']:.2f}%")
                st.write(f"最小回報率: {results['min_return_pred']:.2f}%")
                st.write(f"買入交易次數: {len(results['buy_signals_pred'])}")
                st.write(f"賣出交易次數: {len(results['sell_signals_pred'])}")
            with col2:
                st.write("**基於實際 MACD 的策略**")
                st.write(f"初始資金: $100,000")
                st.write(f"最終資金: ${results['capital_values_actual'][-1]:.2f}")
                st.write(f"總回報率: {results['total_return_actual']:.2f}%")
                st.write(f"最大回報率: {results['max_return_actual']:.2f}%")
                st.write(f"最小回報率: {results['min_return_actual']:.2f}%")
                st.write(f"買入交易次數: {len(results['buy_signals_actual'])}")
                st.write(f"賣出交易次數: {len(results['sell_signals_actual'])}")

            st.subheader("買賣記錄對比")
            st.write("**基於預測 MACD 的策略**")
            pred_records = []
            for date, price in results['buy_signals_pred']:
                pred_records.append({'日期': date.strftime('%Y-%m-%d'), '類型': '買入', '股價': f"{price:.2f}"})
            for date, price in results['sell_signals_pred']:
                pred_records.append({'日期': date.strftime('%Y-%m-%d'), '類型': '賣出', '股價': f"{price:.2f}"})
            pred_records = sorted(pred_records, key=lambda x: x['日期'])
            if pred_records:
                df_pred = pd.DataFrame(pred_records)
                st.table(df_pred)
            else:
                st.write("無預測 MACD 策略的買賣記錄可顯示。")

            st.write("**基於實際 MACD 的策略**")
            actual_records = []
            for date, price in results['buy_signals_actual']:
                actual_records.append({'日期': date.strftime('%Y-%m-%d'), '類型': '買入', '股價': f"{price:.2f}"})
            for date, price in results['sell_signals_actual']:
                actual_records.append({'日期': date.strftime('%Y-%m-%d'), '類型': '賣出', '股價': f"{price:.2f}"})
            actual_records = sorted(actual_records, key=lambda x: x['日期'])
            if actual_records:
                df_actual = pd.DataFrame(actual_records)
                st.table(df_actual)
            else:
                st.write("無實際 MACD 策略的買賣記錄可顯示。")

            st.subheader("模型評估指標")
            st.write(f"MAE: {results['mae']:.4f}")
            st.write(f"RMSE: {results['rmse']:.4f}")
            st.write(f"R²: {results['r2']:.4f}")
            st.write(f"MAPE: {results['mape']:.2f}%")
            st.write(f"總耗時: {results['elapsed_time']:.2f} 秒")

    elif mode == "預測模式":
        st.markdown("### 預測模式\n上載保存的模型和縮放器（.keras 格式），下載新數據並進行股價預測（包括未來 N 天）。")
        stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL）", value="TSLA")
        timesteps = st.slider("選擇時間步長（需與訓練時一致）", min_value=10, max_value=100, value=30, step=10)
        predict_days = st.selectbox("選擇預測天數（需與訓練時一致）", [1, 3, 5, 7], index=1)
        eastern = pytz.timezone('US/Eastern')
        current_date = datetime.now(eastern).replace(hour=0, minute=0, second=0, microsecond=0)
        default_start_date = current_date - timedelta(days=90)
        default_end_date = current_date - timedelta(days=1)
        start_date = st.date_input("選擇歷史數據開始日期", value=default_start_date, max_value=current_date)
        end_date = st.date_input("選擇歷史數據結束日期", value=default_end_date, max_value=current_date)
        future_days = st.selectbox("選擇未來預測天數", [1, 5], index=0)
        model_file = st.file_uploader("上載模型文件 (.keras)", type=["keras"])
        scaler_features_file = st.file_uploader("上載特徵縮放器 (.pkl)", type=["pkl"])
        scaler_target_file = st.file_uploader("上載目標縮放器 (.pkl)", type=["pkl"])

        if st.button("運行預測") and model_file and scaler_features_file and scaler_target_file:
            with st.spinner("正在載入模型並預測（包括未來預測）..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_model:
                    tmp_model.write(model_file.read())
                    tmp_model_path = tmp_model.name
                custom_objects = {"Attention": Attention, "mse": tf.keras.losses.MeanSquaredError(), "MeanSquaredError": tf.keras.losses.MeanSquaredError}
                model = load_model(tmp_model_path, custom_objects=custom_objects)
                os.unlink(tmp_model_path)
                scaler_features = pickle.load(scaler_features_file)
                scaler_target = pickle.load(scaler_target_file)
                start_date = eastern.localize(datetime.combine(start_date, datetime.min.time()))
                end_date = eastern.localize(datetime.combine(end_date, datetime.min.time()))
                data = fetch_stock_data(stock_symbol, start_date, end_date)
                if data is None:
                    return
                try:
                    X_new, y_new, y_current_new, new_dates, full_data = preprocess_data(data, timesteps, predict_days,
                                                                         scaler_features=scaler_features,
                                                                         scaler_target=scaler_target, is_training=False)
                except ValueError as e:
                    st.error(str(e))
                    return
                historical_predictions = predict_step(model, X_new)
                historical_predictions = scaler_target.inverse_transform(historical_predictions)
                y_current_new = scaler_target.inverse_transform(y_current_new)
                future_predictions = []
                last_sequence = X_new[-1].copy()
                last_close = float(full_data['Close'].iloc[-1])
                last_open = float(full_data['Open'].iloc[-1])
                last_high = float(full_data['High'].iloc[-1])
                last_low = float(full_data['Low'].iloc[-1])
                for _ in range(future_days):
                    pred = predict_step(model, last_sequence[np.newaxis, :])
                    pred_price = scaler_target.inverse_transform(pred)[0, 0]
                    future_predictions.append(pred_price)
                    new_features = [last_close, last_open, last_high, last_low, (last_high + last_low + pred_price) / 3]
                    scaled_new_features = scaler_features.transform([new_features])
                    last_sequence = np.roll(last_sequence, -1, axis=0)
                    last_sequence[-1] = scaled_new_features[0]
                    last_close = pred_price
                all_dates = np.concatenate([new_dates, pd.date_range(start=end_date + timedelta(days=1), periods=future_days, tz=eastern)])
                all_actual = np.concatenate([full_data.loc[new_dates, 'Close'].values, [np.nan] * future_days])
                all_predicted = np.concatenate([historical_predictions.flatten(), future_predictions])
                fig_price = create_price_comparison_chart(all_dates, all_actual, all_predicted, stock_symbol,
                                                         f"歷史與未來預測股價比較 ({start_date.strftime('%Y-%m-%d')} 至 {all_dates[-1].strftime('%Y-%m-%d')})")
                if len(y_current_new) > 0:
                    mae = mean_absolute_error(y_current_new, historical_predictions)
                    rmse = np.sqrt(mean_squared_error(y_current_new, historical_predictions))
                    r2 = r2_score(y_current_new, historical_predictions)
                    mape = np.mean(np.abs((y_current_new - historical_predictions) / y_current_new)) * 100
                    st.subheader("歷史數據預測評估指標")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")
                    st.write(f"R²: {r2:.4f}")
                    st.write(f"MAPE: {mape:.2f}%")
                st.subheader("圖表分析")
                st.plotly_chart(fig_price, use_container_width=True, config=plotly_config)
                st.subheader("未來預測價格")
                future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=future_days, tz=eastern)
                for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
                    st.write(f"日期: {date.strftime('%Y-%m-%d')}，預測價格: {price:.2f}")

    if st.button("還原狀態"):
        if 'results' in st.session_state:
            del st.session_state['results']
        st.session_state['training_progress'] = 40
        st.rerun()

if __name__ == "__main__":
    main()
