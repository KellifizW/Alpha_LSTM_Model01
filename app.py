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
import pickle
import io
import os
import tempfile

# 自訂 Attention 層
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

# 構建模型函數
def build_model(input_shape, model_type="original"):
    if model_type == "lstm_simple":
        model = Sequential()
        model.add(LSTM(150, activation='relu', input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.01))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=128, kernel_size=1, activation='relu', padding='same')(inputs)
        x = Bidirectional(LSTM(units=128, activation='tanh', return_sequences=True))(x)
        x = Dropout(0.01)(x)
        x = Attention()(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 數據預處理（支援訓練和預測）
def preprocess_data(data, timesteps, scaler_features=None, scaler_target=None, is_training=True):
    data['Yesterday_Close'] = data['Close'].shift(1)
    data['Average'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data.dropna()

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

# 回測與交易策略
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

    # 初始化 session_state
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'training_progress' not in st.session_state:
        st.session_state['training_progress'] = 40

    # 模式選擇
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

        current_date = datetime(2025, 3, 17)
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

                progress_bar.progress(20)
                status_text.text("步驟 2/5: 預處理數據...")
                X_train, X_test, y_train, y_test, scaler_features, scaler_target, test_dates, full_data = preprocess_data(data, timesteps, is_training=True)

                progress_bar.progress(40)
                status_text.text("步驟 3/5: 訓練模型...")
                model_type_selected = "original" if model_type.startswith("original") else "lstm_simple"
                model = build_model(input_shape=(timesteps, X_train.shape[2]), model_type=model_type_selected)

                st.write("模型結構：")
                model_summary = io.StringIO()
                model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
                st.text(model_summary.getvalue())

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
                    'selected_period': selected_period
                }

        # 顯示訓練結果
        if st.session_state['results'] is not None:
            results = st.session_state['results']
            stock_symbol = results['stock_symbol']
            selected_period = results['selected_period']

            st.subheader("下載訓練結果")
            temp_model_path = "temp_model.h5"
            results['model'].save(temp_model_path)
            with open(temp_model_path, "rb") as f:
                model_buffer = io.BytesIO(f.read())
            st.download_button(
                label="下載訓練好的模型",
                data=model_buffer,
                file_name=f"{stock_symbol}_lstm_model.h5",
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
        上載保存的模型和縮放器，下載新數據並進行股價預測。
        """)

        stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL）", value="TSLA")
        timesteps = st.slider("選擇時間步長（需與訓練時一致）", min_value=10, max_value=100, value=30, step=10)
        start_date = st.date_input("選擇新數據開始日期", value=datetime(2025, 1, 1))
        end_date = st.date_input("選擇新數據結束日期", value=datetime(2025, 3, 17))

        model_file = st.file_uploader("上載模型文件 (.h5)", type=["h5"])
        scaler_features_file = st.file_uploader("上載特徵縮放器 (.pkl)", type=["pkl"])
        scaler_target_file = st.file_uploader("上載目標縮放器 (.pkl)", type=["pkl"])

        if st.button("運行預測") and model_file and scaler_features_file and scaler_target_file:
            with st.spinner("正在載入模型並預測..."):
                # 載入模型（使用臨時檔案）
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model:
                    tmp_model.write(model_file.read())
                    tmp_model_path = tmp_model.name
                model = load_model(tmp_model_path, custom_objects={"Attention": Attention})
                os.unlink(tmp_model_path)  # 刪除臨時檔案

                # 載入縮放器（直接從文件流讀取）
                scaler_features = pickle.load(scaler_features_file)
                scaler_target = pickle.load(scaler_target_file)

                # 下載新數據
                data = yf.download(stock_symbol, start=start_date, end=end_date)
                if data.empty:
                    st.error("無法下載新數據，請檢查股票代碼或日期範圍！")
                    return

                # 預處理新數據
                X_new, y_new, new_dates, full_data = preprocess_data(data, timesteps, scaler_features, scaler_target, is_training=False)

                # 進行預測
                predictions = predict_step(model, X_new)
                predictions = scaler_target.inverse_transform(predictions)
                y_new = scaler_target.inverse_transform(y_new)

                # 顯示結果
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=new_dates, y=y_new.flatten(), mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=new_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
                fig.update_layout(
                    title=f'{stock_symbol} 預測結果 ({start_date} to {end_date})',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=600,
                    width=1000
                )
                st.plotly_chart(fig, use_container_width=True)

                # 顯示評估指標（如果有實際數據）
                if len(y_new) > 0:
                    mae = mean_absolute_error(y_new, predictions)
                    rmse = np.sqrt(mean_squared_error(y_new, predictions))
                    r2 = r2_score(y_new, predictions)
                    mape = np.mean(np.abs((y_new - predictions) / y_new)) * 100
                    st.subheader("預測評估指標")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")
                    st.write(f"R²: {r2:.4f}")
                    st.write(f"MAPE: {mape:.2f}%")

    # 還原狀態
    if st.button("還原狀態"):
        st.session_state['results'] = None
        st.session_state['training_progress'] = 40
        st.rerun()

if __name__ == "__main__":
    main()
