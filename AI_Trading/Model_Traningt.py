import numpy as np
import tensorflow as tf
import gym
from gym import spaces
from collections import deque
import random
import yfinance as yf
from time import sleep


def traning():
    sleep(86400)
    # Define trading session timings (NYSE timings as an example)
    market_open_time = 930  # Market opens at 9:30 AM
    market_close_time = 1600  # Market closes at 4:00 PM

    # Define a custom trading environment with 'Volume' and 'Open' prices as observations
    while True:
        class StockTradingEnv(gym.Env):
            def __init__(self, data, initial_balance=10000, max_steps=1000):
                super(StockTradingEnv, self).__init__()
                self.data = data
                self.max_steps = max_steps
                self.current_step = 0

                self.initial_balance = initial_balance
                self.balance = initial_balance
                self.stock_owned = 0
                self.stock_price = 0
                self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
                # Including 'Volume' and 'Open' as observations
                self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,))
                # Market opening time (e.g., 9:30 AM)
                self.market_open_time = market_open_time
                # Market closing time (e.g., 4:00 PM)
                self.market_close_time = market_close_time
                # Initialize current time to market open time
                self.current_time = self.market_open_time

            def reset(self):
                self.current_step = 0
                self.balance = self.initial_balance
                self.stock_owned = 0
                self.stock_price = self.data['Close'].values[self.current_step]
                self.volume = self.data['Volume'].values[self.current_step]
                self.open_price = self.data['Open'].values[self.current_step]
                # Reset current time to market open time
                self.current_time = self.market_open_time
                return np.array([self.balance, self.stock_owned, self.stock_price, self.open_price, self.volume, self.current_time])

            def step(self, action):
                self.current_step += 1
                if self.current_step >= len(self.data) or self.current_step >= self.max_steps:
                    return np.array([self.balance, self.stock_owned, self.stock_price, self.open_price, self.volume, self.current_time]), 0, True, {}

                prev_value = self.balance + (self.stock_owned * self.stock_price)

                # Check if it's time to close the market
                if self.current_time >= self.market_close_time:
                    action = 0  # Force "Hold" action when market is about to close

                if action == 0:  # Hold
                    pass
                elif action == 1:  # Buy
                    max_buyable = int(self.balance / self.stock_price)
                    # Limit the maximum number of shares to buy
                    shares_bought = min(max_buyable, 10)
                    cost = shares_bought * self.stock_price
                    self.balance -= cost
                    self.stock_owned += shares_bought
                elif action == 2:  # Sell
                    max_sellable = self.stock_owned
                    # Limit the maximum number of shares to sell
                    shares_sold = min(max_sellable, 10)
                    revenue = shares_sold * self.stock_price
                    self.balance += revenue
                    self.stock_owned -= shares_sold

                self.stock_price = self.data['Close'].values[self.current_step]
                self.volume = self.data['Volume'].values[self.current_step]
                self.open_price = self.data['Open'].values[self.current_step]

                current_value = self.balance + \
                    (self.stock_owned * self.stock_price)
                reward = current_value - prev_value

                # Update the current time
                self.current_time += 1  # Increment time by 1 (e.g., 1 minute)

                return np.array([self.balance, self.stock_owned, self.stock_price, self.open_price, self.volume, self.current_time]), reward, False, {}

        # Fetch stock price data using yfinance ,period="60d",interval='5m'
        symbol = "TATAMOTORS.NS"
        # start_date = "2010-01-01"
        # end_date = "2023-10-03"
        # data = yf.download(symbol, start=start_date, end=end_date)
        data = yf.download(symbol, period="60d", interval='5m')

        # Define the DQN model
        class DQN(tf.keras.models.Model):
            def __init__(self, num_actions):
                super(DQN, self).__init__()
                self.dense1 = tf.keras.layers.Dense(32, activation='relu')
                self.dense2 = tf.keras.layers.Dense(32, activation='relu')
                self.output_layer = tf.keras.layers.Dense(
                    num_actions, activation='linear')

            def call(self, state):
                x = self.dense1(state)
                x = self.dense2(x)
                return self.output_layer(x)

        # Define the replay buffer for experience replay
        class ReplayBuffer:
            def __init__(self, capacity):
                self.buffer = deque(maxlen=capacity)

            def add(self, experience):
                self.buffer.append(experience)

            def sample(self, batch_size):
                return random.sample(self.buffer, batch_size)

        # Hyperparameters
        learning_rate = 0.001
        discount_factor = 0.99
        batch_size = 64
        epsilon = 0.1
        num_episodes = 300
        max_steps = 150
        log_interval = 1  # Log episode information every 10 episodes

        # Create the trading environment
        env = StockTradingEnv(data, max_steps=max_steps)

        # Create the DQN model and target model
        num_actions = env.action_space.n
        model = DQN(num_actions)
        target_model = DQN(num_actions)
        target_model.set_weights(model.get_weights())

        # Define the optimizer and loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Initialize the replay buffer
        replay_buffer = ReplayBuffer(capacity=10000)

        # Create a list to store episode rewards
        episode_rewards = []

        # Training loop
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            for step in range(max_steps):
                # Epsilon-greedy policy
                if np.random.rand() < epsilon:
                    action = np.random.choice(num_actions)
                else:
                    q_values = model.predict(np.array([state]))
                    action = np.argmax(q_values)

                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                replay_buffer.add((state, action, reward, next_state, done))
                state = next_state

                # Update the DQN model
                if len(replay_buffer.buffer) >= batch_size:
                    minibatch = replay_buffer.sample(batch_size)
                    states, actions, rewards, next_states, dones = zip(*minibatch)
                    states = np.array(states)
                    next_states = np.array(next_states)

                    with tf.GradientTape() as tape:
                        q_values = model(states)
                        target_q_values = target_model(next_states)

                        target = q_values.numpy()
                        for i in range(batch_size):
                            if dones[i]:
                                target[i][actions[i]] = rewards[i]
                            else:
                                target[i][actions[i]] = rewards[i] + \
                                    discount_factor * np.max(target_q_values[i])

                        loss = loss_fn(target, q_values)

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))

            # Update the target model periodically
            if episode % 10 == 0:
                target_model.set_weights(model.get_weights())

            episode_rewards.append(total_reward)

            # Log episode information
            # if episode % log_interval == 0:
            #     print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        # Save the trained model
        model.save('dqn_stock_trading_model')
        print("model training complete")
