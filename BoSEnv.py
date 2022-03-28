import gym
from gym import spaces

class RepeatedBoSEnv(gym.Env):
    def __init__(self, partner_policy, horizon):
        super(RepeatedBoSEnv, self).__init__()
        """Implementation of a repeated simultaneous game (Bach or Stravinsky)
        
        partner_policy: function that determines the policy of the partner agent
        horizon: number of games to play
        """
        self.partner_policy = partner_policy
        self.horizon = horizon
        self.action_space = spaces.Discrete(2)
        # observation is the actions of both agents in the previous game
        self.observation_space = spaces.MultiDiscrete([2, 2])
        # state is the actions of both agents in the previous game
        # initialize state randomly since there was no previous game
        self.state = self.observation_space.sample()
        self.game_num = 0

    def step(self, action):
        self.game_num += 1

        partner_action = self.partner_policy(self.state)

        if action == 0 and partner_action == 0:
            # both agents decide on Bach
            reward = 3
        elif action == 0 and partner_action == 1:
            # ego agent decides on Bach and partner on Stravinsky
            reward = 1
        elif action == 1 and partner_action == 0:
            # ego agent decides on Stravinsky and partner on Bach
            reward = 0
        elif action == 1 and partner_action == 1:
            # both agents decide on Stravinsky
            reward = 2

        observation = (action, partner_action)

        return observation, reward, (self.game_num >= self.horizon), {}

    def reset(self):
        self.game_num = 0
        state = self.observation_space.sample()
        self.state = state

        return state


