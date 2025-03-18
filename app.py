# 修改後的模型構建函數
def build_model(input_shape, model_type="original"):
    if model_type == "lstm_simple":
        model = Sequential()
        model.add(LSTM(150, activation='relu', input_shape=input_shape, return_sequences=False))  # 明確設置非序列輸出
        model.add(Dropout(0.01))
        model.add(Dense(32, activation='relu'))  # 新增一層，提升預測波動性
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

# 修改後的回測函數，增加調試與數據對齊
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

    # 預測MACD，限定在測試集範圍
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

    test_dates = pd.to_datetime(test_dates)
    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end)

    mask = (test_dates >= period_start) & (test_dates <= period_end)
    filtered_dates = test_dates[mask]

    if len(filtered_dates) == 0:
        st.error("回測時段不在測試數據範圍內！")
        return None, None, None, None, None, None

    test_start_idx = data.index.get_loc(filtered_dates[0])
    test_end_idx = data.index.get_loc(filtered_dates[-1])

    capital_values = [initial_capital] * test_start_idx

    for i in range(test_start_idx, test_end_idx + 1):
        close_price = data['Close'].iloc[i].item()
        macd_pred = data['MACD_pred'].iloc[i]
        signal_pred = data['Signal_pred'].iloc[i]
        pred_price = data['Predicted'].iloc[i] if not pd.isna(data['Predicted'].iloc[i]) else close_price

        # 調試訊息
        if i == test_start_idx or i == test_end_idx:
            st.write(f"日期: {data.index[i]}, MACD_pred: {macd_pred:.4f}, Signal_pred: {signal_pred:.4f}, Pred_price: {pred_price:.2f}, Close: {close_price:.2f}")

        # 止損規則
        if position == 1 and pred_price < close_price * 0.98:
            capital += shares * close_price
            position = 0
            shares = 0
            sell_signals.append((data.index[i], close_price))
            st.write(f"止損賣出: {data.index[i]}, 價格: {close_price:.2f}")

        # 買入信號
        elif macd_pred > signal_pred and i > 0 and data['MACD_pred'].iloc[i - 1] <= data['Signal_pred'].iloc[i - 1]:
            if position == 0:
                shares = capital // close_price
                capital -= shares * close_price
                position = 1
                buy_signals.append((data.index[i], close_price))
                st.write(f"買入: {data.index[i]}, 價格: {close_price:.2f}")

        # 賣出信號
        elif macd_pred < signal_pred and i > 0 and data['MACD_pred'].iloc[i - 1] >= data['Signal_pred'].iloc[i - 1]:
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

    return capital_values, total_return, max_return, min_return, buy_signals, sell_signals

# 主程式中更新MACD圖表（部分代碼）
st.subheader("MACD 分析（回測期間，基於預測價格）")
data_backtest = full_data.loc[period_start:period_end].copy()
data_backtest['EMA12_pred'] = pd.Series(predictions.flatten()[-len(test_dates):])[mask].ewm(span=12, adjust=False).mean()
data_backtest['EMA26_pred'] = pd.Series(predictions.flatten()[-len(test_dates):])[mask].ewm(span=26, adjust=False).mean()
data_backtest['MACD_pred'] = data_backtest['EMA12_pred'] - data_backtest['EMA26_pred']
data_backtest['Signal_pred'] = data_backtest['MACD_pred'].ewm(span=9, adjust=False).mean()
