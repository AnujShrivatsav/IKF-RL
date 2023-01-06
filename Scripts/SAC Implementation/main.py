import panda_gym
import gym
import numpy as np
from sac import Agent
from utils import plot_learning_curve
from gym import wrappers

if __name__ == "__main__":
    env = gym.make("PandaReachDense-v2")
    # 9
    agent = Agent(input_dims=[9], env=env, n_actions=env.action_space.shape[0])
    n_games = 2500
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = "PandaReach.png"
    filename1 = "PandaReach_Actor_loss.png"
    filename2 = "PandaReach_Critic_loss.png"
    figure_file = "plots/" + filename
    figure_file1 = "plots/" + filename1
    figure_file2 = "plots/" + filename2

    best_score = env.reward_range[0]
    score_history, actor_loss, critic_loss = [], [], []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")
    for i in range(n_games):
        observation = env.reset()
        observation = np.concatenate(
            [observation["observation"], observation["desired_goal"]]
        )
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = np.concatenate(
                [observation_["observation"], observation_["desired_goal"]]
            )
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        actor_loss.append(agent.actor_loss)
        critic_loss.append(agent.critic_loss)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print("episode ", i, "score %.1f" % score, "avg_score %.1f" % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, "Rewards", figure_file, WINDOW=100)
        plot_learning_curve(x, actor_loss, "Actor Loss", figure_file1, WINDOW=50)
        plot_learning_curve(x, critic_loss, "Critic Loss", figure_file2, WINDOW=25)
