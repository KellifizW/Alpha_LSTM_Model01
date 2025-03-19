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

# 顯示 TensorFlow 版本並加上當日美國時間
eastern = pytz.timezone('US/Eastern')
current_date = datetime.now(eastern)
st.write(f"當前 TensorFlow 版本: {tf.__version__}，今日美國東部時間: {current_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# 自訂 Attention 層（保持不變）
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

# 構建模型函數（修正為接受 learning_rate）
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

# 數據預處理（保持不變）
def preprocess_data(data, timesteps, scaler_features=None, scaler_target=None, is_training=True):
    if data.empty:
        raise ValueError("輸入數據為空，無法進行預處理。請檢查數據來源或日期範圍。")

    data['Yesterday_Close'] = data['Close'].shift(1)
    data['Average'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data.dropna()

    if len(data) < timesteps:
        raise ValueError(f"數據樣本數 ({len(data)}) 小於時間步長 ({timesteps})，無法生成有效輸入。")

    features = ['Yesterday_Close', 'Open', 'High', 'Low', 'Average']
    target = 'Close'

    if is_training:
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()
        scaled_features = scaler_features.fit_transform(data[features])
        scaled_target = scaler_target.fit_transform(data[[target]])
    else:
        scaled_features = scaler_features.transform(data[features])
        scaled_target = scaler_target.transform(data[[target]])

    total_samples = len(scaled_features) - timesteps
    if total_samples <= 0:
        raise ValueError(f"數據樣本數不足以生成時間序列，總樣本數: {len(scaled_features)}，時間步長: {timesteps}")

    X, y = [], []
    for i in range(total_samples):
        X.append(scaled_features[i:i + timesteps])
        y.append(scaled_target[i + timesteps])

    X = np.array(X)
    y = np.array(y)

    if is_training:
        data_index = pd.to_datetime(data.index)
        train_size = int(total_samples * 0.7)
        test_size = total_samples - train_size
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        test_dates = data_index[timesteps + train_size:timesteps + train_size + test_size]
        return X_train, X_test, y_train, y_test, scaler_features, scaler_target, test_dates, data
    else:
        return X, y, data.index[timesteps:], data

# 預測函數
@tf.function(reduce_retracing=True)
def predict_step(model, x):
    return model(x, training=False)

# 回測與交易策略（保持不變）
def backtest(data, predictions, test_dates, period_start, period_end, initial_capital=100000):
    data = data.copy()
    test_size = len(predictions)
    data['Predicted'] = np.nan
    data.iloc[-test_size:, data.columns.get_loc('Predicted')] = predictions.flatten()

    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    test_mask = data.index.isin(test_dates)
    data.loc[test_mask, 'EMA12_pred'] = pd.Series(predictions.flatten(), index=test_dates).ewm(span=12, adjust=False).mean()
    data.loc[test_mask, 'EMA26_pred'] = pd.Series(predictions.flatten(), index=test_dates).ewm(span=26, adjust=False).mean()
    data['MACD_pred'] = data['EMA12_pred'] - data['EMA26_pred']
    data['Signal_pred'] = data['MACD_pred'].ewm(span=9, adjust=False).mean()

    position = 0
    capital = initial_capital
    shares = 0
    capital_values = []
    buy_signals = []
    sell_signals = []
    golden_cross = []
    death_cross = []

    test_dates = pd.to_datetime(test_dates)
    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end)

    mask = (test_dates >= period_start) & (test_dates <= period_end)
    filtered_dates = test_dates[mask]

    if len(filtered_dates) == 0:
        st.error("回測時段不在測試數據範圍內！")
        return None, None, None, None, None, None, None, None, None

    test_start_idx = data.index.get_loc(filtered_dates[0])
    test_end_idx = data.index.get_loc(filtered_dates[-1])

    capital_values = [initial_capital] * test_start_idx

    for i in range(test_start_idx, test_end_idx + 1):
        close_price = data['Close'].iloc[i].item()
        macd_pred = data['MACD_pred'].iloc[i]
        signal_pred = data['Signal_pred'].iloc[i]
        pred_price = data['Predicted'].iloc[i] if not pd.isna(data['Predicted'].iloc[i]) else close_price

        if i == test_start_idx or i == test_end_idx:
            st.write(f"日期: {data.index[i]}, MACD_pred: {macd_pred:.4f}, Signal_pred: {signal_pred:.4f}")

        if i > test_start_idx:
            if pd.notna(macd_pred) and pd.notna(signal_pred):
                prev_macd = data['MACD_pred'].iloc[i - 1]
                prev_signal = data['Signal_pred'].iloc[i - 1]
                if macd_pred > signal_pred and prev_macd <= prev_signal:
                    golden_cross.append((data.index[i], macd_pred))
                elif macd_pred < signal_pred and prev_macd >= prev_signal:
                    death_cross.append((data.index[i], macd_pred))

        if position == 1 and pred_price < close_price * 0.98:
            capital += shares * close_price
            position = 0
            shares = 0
            sell_signals.append((data.index[i], close_price))
            st.write(f"止損賣出: {data.index[i]}, 價格: {close_price:.2f}")

        elif macd_pred > signal_pred and i > test_start_idx and data['MACD_pred'].iloc[i - 1] <= data['Signal_pred'].iloc[i - 1]:
            if position == 0:
                shares = capital // close_price
                capital -= shares * close_price
                position = 1
                buy_signals.append((data.index[i], close_price))
                st.write(f"買入: {data.index[i]}, 價格: {close_price:.2f}")

        elif macd_pred < signal_pred and i > test_start_idx and data['MACD_pred'].iloc[i - 1] >= data['Signal_pred'].iloc[i - 1]:
            if position == 1:
                capital += shares * close_price
                position = 0
                shares = 0
                sell_signals.append((data.index[i], close_price))
                st.write(f"賣出: {data.index[i]}, 價格: {close_price:.2f}")

        total_value = capital + (shares * close_price if position > 0 else 0)
        capital_values.append(total_value)

    capital_values = np.array(capital_values)
    total_return = (capital_values[-1] / capital_values[0] - 1) * 100
    max_return = (max(capital_values) / capital_values[0] - 1) * 100
    min_return = (min(capital_values) / capital_values[0] - 1) * 100

    return data, capital_values, total_return, max_return, min_return, buy_signals, sell_signals, golden_cross, death_cross

# 主程式
def main():
    st.title("股票價格預測與回測系統 BETA")
    st.write(f"當前 Streamlit 版本: {st.__version__}")

    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'training_progress' not in st.session_state:
        st.session_state['training_progress'] = 40

    mode = st.sidebar.selectbox("選擇模式", ["訓練模式", "預測模式"])

    if mode == "訓練模式":
        st.markdown("""
        ### 訓練模式
        輸入參數並訓練模型，生成預測和回測結果。訓練完成後可下載模型和縮放器。
        """)

        stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL）", value="TSLA")
        timesteps = st.slider("選擇時間步長（歷史數據窗口天數）", min_value=10, max_value=100, value=30, step=10)
        epochs = st.slider("選擇訓練次數（epochs）", min_value=50, max_value=200, value=200, step=50)
        model_type = st.selectbox("選擇模型類型", ["original (CNN-BiLSTM-Attention)", "lstm_simple (單層LSTM 150神經元)"], index=0)
        
        learning_rate_options = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
        selected_learning_rate = st.selectbox(
            "選擇 Adam 學習率",
            options=learning_rate_options,
            index=3,
            format_func=lambda x: f"{x:.5f}",
            help="選擇 Adam 優化器的學習率，影響模型訓練速度和收斂性。"
        )

        eastern = pytz.timezone('US/Eastern')
        current_date = datetime.now(eastern).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = current_date - timedelta(days=1095)
        periods = []
        temp_end_date = current_date

        while temp_end_date >= start_date:
            period_start = temp_end_date - timedelta(days=179)
            if period_start < start_date:
                period_start = start_date
            periods.append(f"{period_start.strftime('%Y-%m-%d')} to {temp_end_date.strftime('%Y-%m-%d')}")
            temp_end_date = period_start - timedelta(days=1)

        if not periods:
            st.error("無法生成回測時段選項！請檢查日期範圍設置。")
            return

        selected_period = st.selectbox("選擇回測時段（6個月，最近 3 年）", periods[::-1])

        if st.button("運行分析") and st.session_state['results'] is None:
            start_time = time.time()

            with st.spinner("正在下載數據並訓練模型，請等待..."):
                period_start_str, period_end_str = selected_period.split(" to ")
                period_start = datetime.strptime(period_start_str, "%Y-%m-%d")
                period_end = datetime.strptime(period_end_str, "%Y-%m-%d")
                data_start = period_start - timedelta(days=1095)
                data_end = period_end + timedelta(days=1)

                ticker = yf.Ticker(stock_symbol)
                try:
                    first_trade_date = pd.to_datetime(ticker.info.get('firstTradeDateEpochUtc', 0), unit='s')
                    if first_trade_date > pd.to_datetime(data_start):
                        st.error(f"股票 {stock_symbol} 上市日期為 {first_trade_date.strftime('%Y-%m-%d')}，無法提供 {data_start.strftime('%Y-%m-%d')} 之前的數據！")
                        return
                except Exception as e:
                    st.warning(f"無法獲取股票 {stock_symbol} 的上市日期，繼續下載數據...（錯誤：{e}）")

                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("步驟 1/5: 下載數據...")
                data = yf.download(stock_symbol, start=data_start, end=data_end)

                if data.empty:
                    st.error("無法獲取此代碼的數據。請檢查股票代碼或時段！")
                    return

                actual_start_date = data.index[0].strftime('%Y-%m-%d')
                actual_end_date = data.index[-1].strftime('%Y-%m-%d')
                total_trading_days = len(data)

                # 計算數據統計特性（修復）
                st.write(f"調試信息 - data 列名: {data.columns.tolist()}")
                if ('Close', stock_symbol) not in data.columns:
                    st.error(f"數據中缺少 ('Close', '{stock_symbol}') 列，無法計算統計特性。數據列: {data.columns.tolist()}")
                    return
                # 明確提取 'Close' 列為 Series
                daily_returns = data[('Close', stock_symbol)].pct_change().dropna()
                st.write(f"調試信息 - daily_returns 類型: {type(daily_returns)}, 長度: {len(daily_returns)}")
                st.write(f"調試信息 - daily_returns 前5行: {daily_returns.head().tolist()}")
                try:
                    volatility = daily_returns.std()
                    mean_return = daily_returns.mean()
                    autocorrelation = daily_returns.autocorr()
                except Exception as e:
                    volatility = mean_return = autocorrelation = "N/A"
                    st.warning(f"警告：無法計算統計特性，錯誤: {str(e)} (daily_returns 長度: {len(daily_returns)})")

                progress_bar.progress(20)
                status_text.text("步驟 2/5: 預處理數據...")
                X_train, X_test, y_train, y_test, scaler_features, scaler_target, test_dates, full_data = preprocess_data(data, timesteps, is_training=True)

                total_samples = len(X_train) + len(X_test)
                train_samples = len(X_train)
                test_samples = len(X_test)
                train_date_range = f"{full_data.index[timesteps].strftime('%Y-%m-%d')} to {full_data.index[timesteps + train_samples - 1].strftime('%Y-%m-%d')}"
                test_date_range = f"{test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}"

                progress_bar.progress(40)
                status_text.text("步驟 3/5: 訓練模型...")
                model_type_selected = "original" if model_type.startswith("original") else "lstm_simple"
                model = build_model(input_shape=(timesteps, X_train.shape[2]), model_type=model_type_selected, learning_rate=selected_learning_rate)

                st.write("模型結構：")
                model_summary = io.StringIO()
                model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
                st.text(model_summary.getvalue())

                st.subheader("運算記錄")
                st.write(f"正在下載的股票歷史數據日期範圍: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")
                st.write(f"實際已下載的數據範圍: {actual_start_date} to {actual_end_date}")
                st.write(f"總共交易日: {total_trading_days}")
                st.write(f"總樣本數: {total_samples}")
                st.write(f"訓練樣本數: {train_samples}")
                st.write(f"測試樣本數: {test_samples}")
                st.write(f"訓練數據範圍: {train_date_range}")
                st.write(f"測試數據範圍: {test_date_range}")
                mean_display = f"{mean_return:.6f}" if isinstance(mean_return, (int, float)) else mean_return
                volatility_display = f"{volatility:.6f}" if isinstance(volatility, (int, float)) else volatility
                autocorrelation_display = f"{autocorrelation:.6f}" if isinstance(autocorrelation, (int, float)) else autocorrelation
                st.write(f"數據統計特性 - 日收益率均值: {mean_display}, 波動率: {volatility_display}, 自相關係數: {autocorrelation_display}")

                progress_per_epoch = 20 / epochs
                def update_progress(epoch, logs):
                    st.session_state['training_progress'] = min(60, st.session_state['training_progress'] + progress_per_epoch)
                    progress_bar.progress(int(st.session_state['training_progress']))
                    status_text.text(f"步驟 3/5: 訓練模型 - Epoch {epoch + 1}/{epochs} (損失: {logs.get('loss'):.4f})")

                epoch_callback = LambdaCallback(on_epoch_end=update_progress)
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, validation_split=0.1, verbose=1, callbacks=[epoch_callback])

                progress_bar.progress(60)
                status_text.text("步驟 4/5: 進行價格預測...")
                predictions = predict_step(model, X_test)
                predictions = scaler_target.inverse_transform(predictions)
                y_test = scaler_target.inverse_transform(y_test)

                progress_bar.progress(80)
                status_text.text("步驟 5/5: 執行回測...")
                result = backtest(full_data, predictions, test_dates, period_start, period_end)
                if result[0] is None:
                    return
                full_data, capital_values, total_return, max_return, min_return, buy_signals, sell_signals, golden_cross, death_cross = result

                end_time = time.time()
                elapsed_time = end_time - start_time

                test_dates = pd.to_datetime(test_dates)
                period_start = pd.to_datetime(period_start)
                period_end = pd.to_datetime(period_end)
                mask = (test_dates >= period_start) & (test_dates <= period_end)
                filtered_dates = test_dates[mask]
                filtered_y_test = y_test[mask]
                filtered_predictions = predictions[mask]

                # 生成價格圖表
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=filtered_dates, y=filtered_y_test.flatten(), mode='lines', name='Actual Price'))
                fig_price.add_trace(go.Scatter(x=filtered_dates, y=filtered_predictions.flatten(), mode='lines', name='Predicted Price'))
                buy_x, buy_y = zip(*buy_signals) if buy_signals else ([], [])
                sell_x, sell_y = zip(*sell_signals) if sell_signals else ([], [])
                fig_price.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))
                fig_price.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')))
                fig_price.update_layout(
                    title=f'{stock_symbol} Actual vs Predicted Prices ({selected_period})',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=600,
                    width=1000
                )

                # 生成損失圖表
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
                fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
                fig_loss.update_layout(title='訓練與驗證損失曲線', xaxis_title='Epoch', yaxis_title='Loss')

                # 生成MACD圖表
                data_backtest = full_data.loc[period_start:period_end].copy()
                golden_x, golden_y = zip(*golden_cross) if golden_cross else ([], [])
                death_x, death_y = zip(*death_cross) if death_cross else ([], [])
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data_backtest.index, y=data_backtest['MACD_pred'], mode='lines', name='MACD Line (Predicted)'))
                fig_macd.add_trace(go.Scatter(x=data_backtest.index, y=data_backtest['Signal_pred'], mode='lines', name='Signal Line (Predicted)'))
                fig_macd.add_trace(go.Scatter(x=[data_backtest.index[0], data_backtest.index[-1]], y=[0, 0], mode='lines', name='Zero Line', line=dict(dash='dash')))
                fig_macd.add_trace(go.Scatter(x=golden_x, y=golden_y, mode='markers', name='Golden Cross', marker=dict(symbol='circle', size=10, color='green')))
                fig_macd.add_trace(go.Scatter(x=death_x, y=death_y, mode='markers', name='Death Cross', marker=dict(symbol='circle', size=10, color='red')))
                fig_macd.update_layout(
                    title=f'{stock_symbol} MACD Analysis ({selected_period})',
                    xaxis_title='Date',
                    yaxis_title='MACD Value',
                    height=600,
                    width=1000
                )

                # 計算評估指標
                mae = mean_absolute_error(filtered_y_test, filtered_predictions)
                rmse = np.sqrt(mean_squared_error(filtered_y_test, filtered_predictions))
                r2 = r2_score(filtered_y_test, filtered_predictions)
                mape = np.mean(np.abs((filtered_y_test - filtered_predictions) / filtered_y_test)) * 100

                st.session_state['results'] = {
                    'model': model,
                    'scaler_features': scaler_features,
                    'scaler_target': scaler_target,
                    'fig_price': fig_price,
                    'fig_loss': fig_loss,
                    'fig_macd': fig_macd,
                    'capital_values': capital_values,
                    'total_return': total_return,
                    'max_return': max_return,
                    'min_return': min_return,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'elapsed_time': elapsed_time,
                    'stock_symbol': stock_symbol,
                    'selected_period': selected_period,
                    'history': history.history
                }

        if st.session_state['results'] is not None:
            results = st.session_state['results']
            stock_symbol = results['stock_symbol']
            selected_period = results['selected_period']

            st.subheader("下載訓練結果")
            temp_model_path = "temp_model.keras"
            results['model'].save(temp_model_path)
            with open(temp_model_path, "rb") as f:
                model_buffer = io.BytesIO(f.read())
            st.download_button(
                label="下載訓練好的模型",
                data=model_buffer,
                file_name=f"{stock_symbol}_lstm_model.keras",
                mime="application/octet-stream"
            )
            os.remove(temp_model_path)

            scaler_features_buffer = io.BytesIO()
            pickle.dump(results['scaler_features'], scaler_features_buffer)
            scaler_features_buffer.seek(0)
            st.download_button(
                label="下載特徵縮放器",
                data=scaler_features_buffer,
                file_name=f"{stock_symbol}_scaler_features.pkl",
                mime="application/octet-stream"
            )

            scaler_target_buffer = io.BytesIO()
            pickle.dump(results['scaler_target'], scaler_target_buffer)
            scaler_target_buffer.seek(0)
            st.download_button(
                label="下載目標縮放器",
                data=scaler_target_buffer,
                file_name=f"{stock_symbol}_scaler_target.pkl",
                mime="application/octet-stream"
            )

            st.subheader(f"{stock_symbol} 分析結果（{selected_period}）")
            st.plotly_chart(results['fig_price'], use_container_width=True)
            st.subheader("訓練與驗證損失曲線")
            st.plotly_chart(results['fig_loss'], use_container_width=True)
            st.subheader("MACD 分析")
            st.plotly_chart(results['fig_macd'], use_container_width=True)

            st.subheader("回測結果")
            st.write(f"初始資金: $100,000")
            st.write(f"最終資金: ${results['capital_values'][-1]:.2f}")
            st.write(f"總回報率: {results['total_return']:.2f}%")
            st.write(f"最大回報率: {results['max_return']:.2f}%")
            st.write(f"最小回報率: {results['min_return']:.2f}%")
            st.write(f"買入交易次數: {len(results['buy_signals'])}")
            st.write(f"賣出交易次數: {len(results['sell_signals'])}")

            st.subheader("模型評估指標")
            st.write(f"MAE: {results['mae']:.4f}")
            st.write(f"RMSE: {results['rmse']:.4f}")
            st.write(f"R²: {results['r2']:.4f}")
            st.write(f"MAPE: {results['mape']:.2f}%")
            st.write(f"總耗時: {results['elapsed_time']:.2f} 秒")

    elif mode == "預測模式":
        st.markdown("""
        ### 預測模式
        上載保存的模型和縮放器，下載新數據並進行股價預測（包括未來 N 天）。
        """)

        stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL）", value="TSLA")
        timesteps = st.slider("選擇時間步長（需與訓練時一致）", min_value=10, max_value=100, value=30, step=10)
        eastern = pytz.timezone('US/Eastern')
        current_date = datetime.now(eastern).replace(hour=0, minute=0, second=0, microsecond=0)
        default_start_date = current_date - timedelta(days=90)
        default_end_date = current_date - timedelta(days=1)
        start_date = st.date_input("選擇歷史數據開始日期", value=default_start_date, max_value=current_date)
        end_date = st.date_input("選擇歷史數據結束日期", value=default_end_date, max_value=current_date)
        future_days = st.selectbox("選擇未來預測天數", [1, 5], index=0)  # 用戶選擇 1 天或 5 天

        model_file = st.file_uploader("上載模型文件 (.h5)", type=["h5"])
        scaler_features_file = st.file_uploader("上載特徵縮放器 (.pkl)", type=["pkl"])
        scaler_target_file = st.file_uploader("上載目標縮放器 (.pkl)", type=["pkl"])

        if st.button("運行預測") and model_file and scaler_features_file and scaler_target_file:
            with st.spinner("正在載入模型並預測（包括未來預測）..."):
                # 載入模型
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model:
                    tmp_model.write(model_file.read())
                    tmp_model_path = tmp_model.name
                
                custom_objects = {
                    "Attention": Attention,
                    "mse": tf.keras.losses.MeanSquaredError(),
                    "MeanSquaredError": tf.keras.losses.MeanSquaredError
                }
                model = load_model(tmp_model_path, custom_objects=custom_objects)
                os.unlink(tmp_model_path)

                # 載入縮放器
                scaler_features = pickle.load(scaler_features_file)
                scaler_target = pickle.load(scaler_target_file)

                # 下載歷史數據
                data = yf.download(stock_symbol, start=start_date, end=end_date)
                if data.empty:
                    st.error(f"無法下載 {stock_symbol} 的數據（{start_date} 至 {end_date}）。請檢查股票代碼或日期範圍是否有效！")
                    return

                # 預處理歷史數據
                try:
                    X_new, y_new, new_dates, full_data = preprocess_data(data, timesteps, scaler_features, scaler_target, is_training=False)
                except ValueError as e:
                    st.error(str(e))
                    return

                # 預測歷史數據
                historical_predictions = predict_step(model, X_new)
                historical_predictions = scaler_target.inverse_transform(historical_predictions)
                y_new = scaler_target.inverse_transform(y_new)

                # 未來預測
                future_predictions = []
                last_sequence = X_new[-1]  # 最後一個歷史序列
                # 提取最後一天的純量值
                last_close = float(full_data['Close'].iloc[-1])
                last_open = float(full_data['Open'].iloc[-1])
                last_high = float(full_data['High'].iloc[-1])
                last_low = float(full_data['Low'].iloc[-1])

                for _ in range(future_days):
                    pred = predict_step(model, last_sequence[np.newaxis, :])
                    pred_price = scaler_target.inverse_transform(pred)[0, 0]
                    future_predictions.append(pred_price)
                    # 構造新特徵，使用純量
                    new_features = [
                        last_close,  # Yesterday_Close
                        last_open,   # Open（假設與最後一天相同）
                        last_high,   # High（假設與最後一天相同）
                        last_low,    # Low（假設與最後一天相同）
                        pred_price   # Average（使用預測的 Close）
                    ]
                    scaled_new_features = scaler_features.transform([new_features])[0]
                    # 更新序列
                    last_sequence = np.roll(last_sequence, -1, axis=0)
                    last_sequence[-1] = scaled_new_features
                    # 更新 last_close 為下一輪的 Yesterday_Close
                    last_close = pred_price

                # 合併歷史和未來數據
                future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=future_days)
                all_dates = np.concatenate([new_dates, future_dates])
                all_predictions = np.concatenate([historical_predictions.flatten(), future_predictions])

                # 顯示結果
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=new_dates, y=y_new.flatten(), mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=all_dates, y=all_predictions, mode='lines', name='Predicted Price'))
                fig.add_vline(x=end_date, line_dash="dash", line_color="red", name="Prediction Start")
                fig.update_layout(
                    title=f'{stock_symbol} 預測結果 ({start_date} 至 {end_date} + 未來 {future_days} 天)',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=600,
                    width=1000
                )
                st.plotly_chart(fig, use_container_width=True)

                # 顯示歷史數據的評估指標
                if len(y_new) > 0:
                    mae = mean_absolute_error(y_new, historical_predictions)
                    rmse = np.sqrt(mean_squared_error(y_new, historical_predictions))
                    r2 = r2_score(y_new, historical_predictions)
                    mape = np.mean(np.abs((y_new - historical_predictions) / y_new)) * 100
                    st.subheader("歷史數據預測評估指標")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")
                    st.write(f"R²: {r2:.4f}")
                    st.write(f"MAPE: {mape:.2f}%")
                
                st.subheader("未來預測價格")
                for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
                    st.write(f"日期: {date.strftime('%Y-%m-%d')}，預測價格: {price:.2f}")

    # 還原狀態
    if st.button("還原狀態"):
        st.session_state['results'] = None
        st.session_state['training_progress'] = 40
        st.rerun()

if __name__ == "__main__":
    main()
