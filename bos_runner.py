import gym
import torch as th
from torch.nn import functional as F
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import preprocess_obs

from lta_ppo import LTA_PPO
from lta_features import LTAExtractor
from BoSEnv import RepeatedBoSEnv

def bach_partner(obs):
    return 0

def stravinsky_partner(obs):
    return 1

def helpful_partner(obs):
    # always picks what the robot picked last time
    return obs[0]

def adversarial_partner(obs):
    # always picks the opposite of what the robot picked last time
    return not obs[0]

def eval_model(model, env, n_eval=50, adapt=False):
    obs = env.reset()
    n_correct = 0
    rewards = []

    # keep track of observations and human actions during evaluation
    observations = []
    human_actions = []
    ce_loss = th.nn.CrossEntropyLoss()
    if adapt:
        # unfreeze human prediction model, freeze everything else
        for param in model.policy.parameters():
            param.requires_grad = False
        for (name, param) in model.policy.features_extractor.named_parameters():
            if name[:5] == "human":
                param.requires_grad = True

    for i in range(n_eval):
        obs_tensor = obs_as_tensor(np.array([obs]), model.device)
        actions, values, log_probs = model.policy(obs_tensor)

        preprocessed_obs = preprocess_obs(obs_tensor, model.observation_space)
        human_pred = model.policy.features_extractor.human(preprocessed_obs)

        next_obs, rew, done, _ = env.step(actions[0])

        next_h_action = next_obs[1]
        pred_action = th.argmax(human_pred)

        # store data for adaptation
        observations.append(preprocessed_obs)
        human_actions.append(next_h_action)

        if next_h_action == pred_action:
            n_correct += 1

        rewards.append(rew)
        obs = next_obs
        if done:
            obs = env.reset()

        # run adaptation
        if adapt and i >= 1:
            obs_tensor = th.concat(observations, 0)
            pred_actions = model.policy.features_extractor.human(obs_tensor[:-1,:])
            next_actions = F.one_hot(obs_tensor[:,1][1:].long(), num_classes=2).float()

            # compute cross entropy loss
            loss = ce_loss(pred_actions, next_actions)

            # Optimization step
            model.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

    return n_correct / n_eval, np.mean(rewards)


if __name__ == "__main__":
    th.manual_seed(0)
    np.random.seed(0)

    env = RepeatedBoSEnv(helpful_partner, 20)
    model = LTA_PPO(
        policy=ActorCriticPolicy, 
        env=env,
        policy_kwargs={"features_extractor_class": LTAExtractor,
                       "features_extractor_kwargs": {"features_dim": 16, 
                                                     "n_actions": 2,
                                                     "human_pred": True}}
    )
    n_eval = 100

    acc, avg_rew = eval_model(model, env, n_eval=n_eval)
    print("----------------------------------------------------------")
    print("Before training")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")
    print()

    model.learn(total_timesteps=15000)

    # measure accuracy of human prediction model
    acc, avg_rew = eval_model(model, env, n_eval=n_eval)
    print("----------------------------------------------------------")
    print("After training")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")
    print()

    # test on different human partner policy than training
    env2 = RepeatedBoSEnv(stravinsky_partner, 20)
    # measure accuracy of human prediction model
    acc, avg_rew = eval_model(model, env2, n_eval=1000, adapt=False)
    print("----------------------------------------------------------")
    print("After training (different human partner)")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")
    # TODO: figure out why adaptation doesn't increase prediction accuracy even
    # over a long horizon

