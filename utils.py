def bach_partner(obs):
    return 0

def stravinsky_partner(obs):
    return 1

def helpful_partner(obs):
    # always picks what the robot picked last time
    return obs[0]

def adversarial_partner(obs):
    # always picks the opposite of what the robot picked last time
    if obs[0] == 1:
        return 0
    return 1
