import gym
import torch as th
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import preprocess_obs

from lta_ppo import LTA_PPO
from lta_features import LTAExtractor
from BoSEnv import RepeatedBoSEnv

def bach_partner(obs):
    return 0

def eval_model(model, env, n_eval=50):
    obs = env.reset()
    n_correct = 0
    rewards = []
    for i in range(n_eval):
        obs_tensor = obs_as_tensor(np.array([obs]), model.device)
        actions, values, log_probs = model.policy(obs_tensor)

        preprocessed_obs = preprocess_obs(obs_tensor, model.observation_space)
        human_pred = model.policy.features_extractor.human(preprocessed_obs)

        next_obs, rew, done, _ = env.step(actions[0])

        next_h_action = next_obs[1]
        pred_action = th.argmax(human_pred)

        if next_h_action == pred_action:
            n_correct += 1

        rewards.append(rew)
        obs = next_obs
        if done:
            obs = env.reset()

    return n_correct / n_eval, np.mean(rewards)


if __name__ == "__main__":
    env = RepeatedBoSEnv(bach_partner, 10)
    model = LTA_PPO(
        policy=ActorCriticPolicy, 
        env=env,
        policy_kwargs={"features_extractor_class": LTAExtractor,
                       "features_extractor_kwargs": {"features_dim": 16, "n_actions": 2}}
    )

    acc, avg_rew = eval_model(model, env)
    print("----------------------------------------------------------")
    print("Before training")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")
    print()

    model.learn(total_timesteps=10000)

    # measure accuracy of human prediction model
    acc, avg_rew = eval_model(model, env)
    print("----------------------------------------------------------")
    print("After training")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")


