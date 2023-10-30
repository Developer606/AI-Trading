import numpy as np
import tensorflow as tf
import yfinance as yf
import time
from datetime import datetime
# Load the saved DQN model
def stock_trading_inference(symbol,balance):

 # Replace with the stock symbol you want to trade

    # Load the saved model
    loaded_model = tf.keras.models.load_model('dqn_stock_trading_model')

    # Define a function to make trading decisions
    def make_trading_decision(model, state):
        # Ensure the state has the same shape and format as the model expects
        state = np.array([state])

        # Use the model to predict the Q-values for each action
        q_values = model.predict(state)

        # Choose the action with the highest Q-value (greedy action)
        action = np.argmax(q_values)

        return action
    live_data = yf.download(symbol, period="1d")
    Action=""
    try:
        latest_close_price = live_data['Close'].iloc[-1]
        latest_open_price = live_data['Open'].iloc[-1]
        latest_volume = live_data['Volume'].iloc[-1]
        current_datetime = datetime.now()
        current_time = int(current_datetime.strftime("%I%M"))
        # current_time = time.localtime()

        # Make a trading decision using the loaded model
        # Example state format: [balance, stock_owned, stock_price, open_price, volume, current_time]
        state = [float(balance),2, latest_close_price, latest_open_price, latest_volume, current_time]
        decision = make_trading_decision(loaded_model, state)
        # print(decision)
        if decision == 0:
             Action= "Hold"
        elif decision == 1:
             Action=  "Buy"
        elif decision == 2:
            Action= "Sell"
        else:
            print("Invalid Action")

    # # Wait for some time before making the next decision (e.g., wait for 1 minute)
    except:
        print("data not fatch prooerly")
    time.sleep(3)
    return Action