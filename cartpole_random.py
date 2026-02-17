import gym
import random

def Random_Games(): 
    # Each of this episode is in its own game
    for episode in range(100):
        env.reset()
    
        # this is each frame, up t0 500... but we wont make it that far with random
        for t in range(500):

            # This will display the environment, 
            # but it wont be very fun to watch, because it will be very fast and random
            env.render()

            # This will just create a random sample action in any environment.
            # In this environment, the action can be 0 or 1, which is either left or right. (probably want it to hover in equilibrium, so ~0.5 yea?)
            action = env.action_space.sample()

            # This executes the environment with an action,
            # and returns the observation of the environment, the reward, 
            # if the environment is over (done), and other info.
            next_state, reward, done, info =  env.step(action)

            # print onto one line
            print(t, next_state, reward, done, info, action)
            if done:
                break

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    Random_Games()