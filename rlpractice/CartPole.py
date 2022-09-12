import ray
import ray.rllib.agents.ppo as ppo
import pandas as pd
import json
import matplotlib.pyplot as plt

info = ray.init(ignore_reinit_error=True)
SELECT_ENV = 'CartPole-v1'  # Specifies the OpenAI Gym environment
N_ITER = 100  # Number of training runs.

config = ppo.DEFAULT_CONFIG.copy()  # PPO's default configuration. See the next code cell.
config["log_level"] = "WARN"  # Suppress too many messages, but try "INFO" to see what can be printed.

# Other settings we might adjust:
config["num_workers"] = 8
config["num_sgd_iter"] = 10  # Number of SGD (stochastic gradient descent) iterations per training minibatch.
# I.e., for each minibatch of data, do this many passes over it to train.
config["sgd_minibatch_size"] = 250  # The amount of data records per minibatch
config["model"]["fcnet_hiddens"] = [32,32]  #
config["num_cpus_per_worker"] = 1

agent = ppo.PPOTrainer(config, env=SELECT_ENV)

results = []
episode_data = []
episode_json = []

for n in range(N_ITER):
    result = agent.train()
    results.append(result)

    episode = {'n': n,
               'episode_reward_min': result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max': result['episode_reward_max'],
               'episode_len_mean': result['episode_len_mean']}

    episode_data.append(episode)
    episode_json.append(json.dumps(episode))

    print(
        f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')


df = pd.DataFrame(data=episode_data)
df.plot(x="n", y=["episode_reward_mean", "episode_reward_min", "episode_reward_max"], kind='line', secondary_y=True)

plt.show()
ray.shutdown()
