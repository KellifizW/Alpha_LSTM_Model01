import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from datetime import datetime, timedelta

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

# 數據預處理
def preprocess_data(data, timesteps):
    data['Yesterday_Close'] = data['Close'].shift(1)
    data['Average'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data.dropna()

    features = ['Yesterday_Close', 'Open', 'High', 'Low', 'Average']
    target = 'Close'

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(data[features])
    scaled_target = scaler_target.fit_transform(data[[target]])

    data_index = pd.to_datetime(data.index)
    st.write(f"數據範圍：{data_index[0]} 至 {data_index[-1]}，總共 {len(data_index)} 個交易日")

    total_samples = len(scaled_features) - timesteps
    train_size = int(total_samples * 0.7)
    test_size = total_samples - train_size

    st.write(f"總樣本數：{total_samples}，訓練樣本數：{train_size}，測試樣本數：{test_size}")

    X, y = [], []
    for i in range(total_samples):
        X.append(scaled_features[i:i + timesteps])
        y.append(scaled_target[i + timesteps])

    X = np.array(X)
    y = np.array(y)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    test_dates = data_index[timesteps + train_size:timesteps + train_size + test_size]

    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.write(f"訓練數據範圍：{data_index[0]} 至 {data_index[train_size + timesteps - 1]}")
    st.write(f"測試數據範圍：{data_index[train_size + timesteps]} 至 {data_index[-1]}")

    return X_train, X_test, y_train, y_test, scaler_target, test_dates, data

# 預測函數
@tf.function(reduce_retracing=True)
def predict_step(model, x):
    st.write(f"X_test shape before prediction: {x.shape}")
    return model(x, training=False)

# 回測與交易策略
def backtest(data, predictions, test_dates, period_start, period_end, initial_capital=100000):
    data = data.copy()
    test_size = len(predictions)
    data['Predicted'] = np.nan
    data.iloc[-test_size:, data.columns.get_loc('Predicted')] = predictions.flatten()

    # 原有MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # 預測MACD
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

    st.markdown("""
    ### 功能與限制
    本程式使用深度學習模型預測股票價格，並基於MACD策略進行模擬交易回測。
    - **功能**：輸入股票代碼並選擇回測時段，預測未來價格並查看回測結果。
    - **限制**：預測結果僅供參考，不構成投資建議。模型訓練需要大量計算，可能需要3-5分鐘。
    - **注意**：程式根據選擇的回測時段動態下載前3年的歷史數據（約1095天），按 70%/30% 分割為訓練集和測試集。回測時段必須在測試集範圍內。
    """)

    stock_symbol = st.text_input("輸入股票代碼（例如：TSLA, AAPL，依據Yahoo Finance）", value="TSLA")
    timesteps = st.slider("選擇時間步長（歷史數據窗口天數）", min_value=10, max_value=100, value=30, step=10)
    epochs = st.slider("選擇訓練次數（epochs）", min_value=50, max_value=200, value=200, step=50)
    model_type = st.selectbox("選擇模型類型", ["original (CNN-BiLSTM-Attention)", "lstm_simple (單層LSTM 150神經元)"], index=0)
    st.markdown("**提示**：更高的訓練次數（epochs）可能提高模型準確度，但會增加訓練時間。單層LSTM更高效但可能影響長期依賴捕捉。")

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

    if st.button("運行分析"):
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
                    st.error(f"股票 {stock_symbol} 上市日期為 {first_trade_date.strftime('%Y-%m-%d')}，無法提供 {data_start.strftime('%Y-%m-%d')} 之前的數據！請選擇更晚的回測時段。")
                    return
            except Exception as e:
                st.warning(f"無法獲取股票 {stock_symbol} 的上市日期，繼續下載數據...（錯誤：{e}）")

            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("步驟 1/5: 下載數據...")
            st.write(f"正在下載 {stock_symbol} 的歷史數據：{data_start.strftime('%Y-%m-%d')} 至 {data_end.strftime('%Y-%m-%d')}...")
            data = yf.download(stock_symbol, start=data_start, end=data_end)

            if data.empty:
                st.error("無法獲取此代碼的數據。請檢查股票代碼或時段！")
                return

            progress_bar.progress(20)
            status_text.text("步驟 2/5: 預處理數據...")
            X_train, X_test, y_train, y_test, scaler_target, test_dates, full_data = preprocess_data(data, timesteps)
            
            progress_bar.progress(40)
            status_text.text("步驟 3/5: 訓練模型（這可能需要幾分鐘）...")
            model_type_selected = "original" if model_type.startswith("original") else "lstm_simple"
            model = build_model(input_shape=(timesteps, X_train.shape[2]), model_type=model_type_selected)

            # 打印模型結構
            st.write("模型結構：")
            model.summary(print_fn=lambda x: st.write(x))

            # 添加進度回調
            progress_per_epoch = 20 / epochs
            if 'training_progress' not in st.session_state:
                st.session_state['training_progress'] = 40

            def update_progress(epoch, logs):
                st.session_state['training_progress'] = min(60, st.session_state['training_progress'] + progress_per_epoch)
                progress_bar.progress(int(st.session_state['training_progress']))
                status_text.text(f"步驟 3/5: 訓練模型 - Epoch {epoch + 1}/{epochs} (損失: {logs.get('loss'):.4f})")

            epoch_callback = LambdaCallback(on_epoch_end=update_progress)

            # 訓練並檢查損失
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, validation_split=0.1, verbose=1, callbacks=[epoch_callback])
            st.write("訓練完成！")
            st.write(f"最終訓練損失: {history.history['loss'][-1]:.4f}")
            st.write(f"最終驗證損失: {history.history['val_loss'][-1]:.4f}")

            # 繪製損失曲線
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
            fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
            fig_loss.update_layout(title='訓練與驗證損失曲線', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig_loss)

            progress_bar.progress(60)
            status_text.text("步驟 4/5: 進行價格預測...")
            predictions = predict_step(model, X_test)
            predictions = scaler_target.inverse_transform(predictions)
            y_test = scaler_target.inverse_transform(y_test)

            # 檢查預測結果
            st.write(f"預測結果樣本（前5個）：{predictions[:5].flatten()}")
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                st.error("預測結果中包含無效值（NaN或Inf），模型訓練可能有問題！")

            progress_bar.progress(80)
            status_text.text("步驟 5/5: 執行回測...")
            result = backtest(full_data, predictions, test_dates, period_start, period_end)
            if result[0] is None:
                return
            full_data, capital_values, total_return, max_return, min_return, buy_signals, sell_signals, golden_cross, death_cross = result
            progress_bar.progress(100)
            status_text.text("完成！正在生成結果...")

        end_time = time.time()
        elapsed_time = end_time - start_time

        test_dates = pd.to_datetime(test_dates)
        period_start = pd.to_datetime(period_start)
        period_end = pd.to_datetime(period_end)
        mask = (test_dates >= period_start) & (test_dates <= period_end)
        filtered_dates = test_dates[mask]
        filtered_y_test = y_test[mask]
        filtered_predictions = predictions[mask]

        st.subheader(f"{stock_symbol} 分析結果（{selected_period}）")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_y_test.flatten(), mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_predictions.flatten(), mode='lines', name='Predicted Price'))
        buy_signals = [(d, p) for d, p in buy_signals if period_start <= d <= period_end]
        sell_signals = [(d, p) for d, p in sell_signals if period_start <= d <= period_end]
        buy_x, buy_y = zip(*buy_signals) if buy_signals else ([], [])
        sell_x, sell_y = zip(*sell_signals) if sell_signals else ([], [])
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')))
        fig.update_layout(
            title=f'{stock_symbol} Actual vs Predicted Prices ({selected_period})',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            height=600,
            width=1000
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("MACD 分析（回測期間，基於預測價格）")
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
            title=f'{stock_symbol} MACD Analysis (Based on Predicted Prices, {selected_period})',
            xaxis_title='Date',
            yaxis_title='MACD Value',
            legend_title='Legend',
            height=600,
            width=1000
        )
        st.plotly_chart(fig_macd, use_container_width=True)

        st.subheader("回測結果")
        st.write(f"初始資金: $100,000")
        st.write(f"最終資金: ${capital_values[-1]:.2f}")
        st.write(f"總回報率: {total_return:.2f}%")
        st.write(f"最大回報率: {max_return:.2f}%")
        st.write(f"最小回報率: {min_return:.2f}%")
        st.write(f"買入交易次數: {len(buy_signals)}")
        st.write(f"賣出交易次數: {len(sell_signals)}")

        st.subheader("模型評估指標")
        mae = mean_absolute_error(filtered_y_test, filtered_predictions)
        rmse = np.sqrt(mean_squared_error(filtered_y_test, filtered_predictions))
        r2 = r2_score(filtered_y_test, filtered_predictions)
        mape = np.mean(np.abs((filtered_y_test - filtered_predictions) / filtered_y_test)) * 100

        st.write(f"平均絕對誤差 (MAE): {mae:.4f}")
        st.markdown("*MAE表示預測值與實際值的平均絕對差異，數值越小表示預測越準確*")
        st.write(f"均方根誤差 (RMSE): {rmse:.4f}")
        st.markdown("*RMSE是預測誤差的平方根，對大誤差更敏感，數值越小表示模型表現越好*")
        st.write(f"決定係數 (R²): {r2:.4f}")
        st.markdown("*R²表示模型解釋數據變化的能力，範圍0-1，越接近1表示模型擬合越好*")
        st.write(f"平均絕對百分比誤差 (MAPE): {mape:.2f}%")
        st.markdown("*MAPE表示預測誤差的百分比，數值越小表示預測精度越高*")

        st.subheader("運行時間")
        st.write(f"總耗時: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
