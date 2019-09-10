import gym
import tensorflow as tf
import numpy as np
import random
import math

env = gym.make('CartPole-v0')
# observation = env.reset()


# NN placehoders
input_X = tf.placeholder(shape=(None, 4), dtype=tf.float32)
actual_Y = tf.placeholder(shape=(None, 2), dtype=tf.float32)

# NN Variables
weight1 = tf.Variable(tf.truncated_normal(
    shape=(4, 20), stddev=0.05, dtype=tf.float32))
bias1 = tf.Variable(tf.zeros(shape=(20), dtype=tf.float32))

weight2 = tf.Variable(tf.truncated_normal(
    shape=(20, 2), stddev=0.05, dtype=tf.float32))
bias2 = tf.Variable(tf.zeros(shape=(2), dtype=tf.float32))

# outputs
hidden_1 = tf.nn.relu(tf.add(tf.matmul(input_X, weight1), bias1))
output = tf.add(tf.matmul(hidden_1, weight2), bias2)

# memory and memory size for Replay
memory = []
memory_size = 10000
sample_size = 5000
batch_size = 20
learning_rate = 0.05
gamma = 0.9
decay = 0.9
num_states = 4
num_action = 2
max_decay = 0.95
min_decay = 0.2
# max_rewards = -999
max_iterations = 20

# loss and optimizer
loss = tf.losses.mean_squared_error(actual_Y, output)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()


def get_samples():
    if len(memory) < sample_size:
        return random.sample(memory, len(memory))
    else:
        print("Sampled")
        return random.sample(memory, sample_size)


def replay(sess, epsilon):
    sample = get_samples()
    current_states = [np.zeros(num_states) if val[0]
                      is None else val[0] for val in sample]
    next_states = [np.zeros(num_states) if val[1]
                   is None else val[1] for val in sample]

    # print(current_states)
    # print(next_states)
    q_s_a = sess.run(output, feed_dict={input_X: current_states})
    q_s_a_d = sess.run(output, feed_dict={input_X: next_states})

    # print(q_s_a)
    # print(q_s_a_d)
    x = np.zeros((len(sample), num_states))
    y = np.zeros((len(sample), num_action))

    for i, b in enumerate(sample):
        state, next_state, action, reward = b[0], b[1], b[2], b[3]
        current_q = q_s_a[i]
        if next_state is None:
            current_q[action] = reward
        else:
            current_q[action] = reward + epsilon * np.amax(q_s_a_d[i])
        x[i] = state
        y[i] = current_q
        # decay = decay/2

    sess.run(optimizer, feed_dict={actual_Y: y,
                                   input_X: x})


def add_to_memory(data):
    if len(memory) > memory_size:
        memory.pop()
    else:
        # print("Adding")
        memory.append(data)


with tf.Session() as sess:
    sess.run(init)
    iteration = 0
    for step in range(batch_size):
        max_rewards = 0
        prev_state = env.reset()
        epsilon = min_decay + (max_decay - min_decay) * math.exp(-decay * step)
        while True:
            env.render()
            action = np.argmax(sess.run(output, feed_dict={input_X: np.array(
                prev_state).reshape(1, 4)}))  # your agent here (this takes random actions)
            print(action)
            next_state, reward, done, info = env.step(action)
            # print(next_state[0])
            if next_state[3] > -0.5 and next_state[3] < 0.5:
                reward += 200
            else:
                reward -= 100

            add_to_memory([prev_state, next_state, action, reward])
            prev_state = next_state
            replay(sess, epsilon)
            max_rewards += reward
            if done:
                print("Max Reward: ", max_rewards)
                break
        print("Next Batch")
