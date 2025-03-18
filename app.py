import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Layer, Input
from tensorflow.keras.backend import tanh, dot, sigmoid, softmax, sum as K_sum
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 自訂 Attention 層（保持不變）
class Attention(Layer):
    # ... (與原程式相同)
    pass

# 構建模型
def build_model(input_shape):
    # ... (與原程式相同)
    pass

# 數據預處理與預測準備
def preprocess_data(data, timesteps):
    data['Yesterday_Close'] = data['Close'].shift(1)
    data['Average'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data.dropna()

    features = ['Yesterday_Close', 'Open', 'High', 'Low', 'Average']
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(data[features])
    scaled_target = scaler_target.fit_transform(data[['Close']])

    X = []
    for i in range(len(scaled_features) - timesteps):
        X.append(scaled_features[i:i + timesteps])

    X = np.array(X)
    X_last = X[-1:]  # 最後一個窗口用於預測
    return X, X_last, scaler_features, scaler_target, data.index[-1]

# 預測未來 5 天
def forecast_next_5_days(model, X_last, scaler_features, scaler_target, last_features, timesteps, forecast_days=5):
    forecast = []
    current_input = X_last.copy()

    for _ in range(forecast_days):
        pred = model.predict(current_input, verbose=0)
        pred_price = scaler_target.inverse_transform(pred)[0, 0]
        forecast.append(pred_price)

        # 更新輸入數據：移除最早一天，加入新預測
        new_features = np.array([[
            last_features['Yesterday_Close'][-1],  # 昨日收盤價
            pred_price,  # 假設開盤價=預測價
            pred_price,  # 假設最高價=預測價
            pred_price,  # 假設最低價=預測價
            pred_price   # 假設平均價=預測價
        ]])
        new_scaled_features = scaler_features.transform(new_features)
        current_input = np.append(current_input[:, 1:, :], new_scaled_features.reshape(1, 1, -1), axis=1)
        last_features['Yesterday_Close'][-1] = pred_price

    return forecast

# 主程式
def main():
    st.title("股票價格預測系統")

    st.markdown("""
    ### 程式功能與限制
    本程式使用深度學習模型預測股票未來 5 天的價格走勢。
    - **功能**：輸入股票代碼，程式自動下載最近 3 年的歷史數據，預測未來 5 天股價。
    - **限制**：預測結果僅供參考，不保證準確性。訓練時間約需 1-2 分鐘，請耐心等候。
    """)

    stock_symbol = st.text_input("請輸入股票代碼（例如 TSLA, AAPL）", value="TSLA")

    if st.button("運行預測"):
        with st.spinner("正在下載數據並訓練模型，請耐心等候 1-2 分鐘..."):
            # 自動計算下載時段
            end_date = datetime.today()
            start_date = end_date - timedelta(days=1095)  # 約 3 年
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            if data.empty:
                st.error("無法獲取該股票數據，請檢查股票代碼是否正確！")
                return
            if len(data) < 60:
                st.error("歷史數據不足，至少需要 60 天數據！")
                return

            # 數據預處理
            timesteps = 60
            X, X_last, scaler_features, scaler_target, last_date = preprocess_data(data, timesteps)
            y = scaler_target.transform(data[['Close']])[timesteps:]

            # 訓練模型
            model = build_model(input_shape=(timesteps, X.shape[2]))
            model.fit(X, y, epochs=200, batch_size=256, validation_split=0.1, verbose=0)

            # 預測未來 5 天
            last_features = data[features].tail(1)
            forecast = forecast_next_5_days(model, X_last, scaler_features, scaler_target, last_features, timesteps)

            # 生成未來日期
            future_dates = [last_date + timedelta(days=i+1) for i in range(5)]

            # 顯示結果
            st.subheader(f"{stock_symbol} 未來 5 天預測")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index[-60:], data['Close'][-60:], label='Historical Price')
            ax.plot(future_dates, forecast, label='Forecasted Price', linestyle='--')
            ax.set_title(f'{stock_symbol} Price Forecast (Next 5 Days)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

            st.write("預測價格：")
            for date, price in zip(future_dates, forecast):
                st.write(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

if __name__ == "__main__":
    main()