import copy
import os
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect

def collect_data(env, num_episodes=500, episode_length=50):
    trajs_state = []
    trajs_imgs = []
    trajs_action = []
    goals = []          # last state of the trajectory
    desired_goals = []  # where the env randomly sampled for the goal
    for i in tqdm(range(num_episodes)):
        traj_state = []
        traj_imgs = []
        traj_action = []

        s_t = env.reset()
        traj_state.append(s_t['observation'].copy())
        for t in range(episode_length):
            a_t = env.expert_action()
            s_tp1, r_t, done, info = env.step(a_t)
            img_t = env.render()
            s_t = s_tp1

            traj_state.append(s_t['observation'].copy())
            traj_imgs.append(np.expand_dims(img_t.copy(), axis=0))
            traj_action.append(a_t.copy())

        trajs_state.append(np.vstack(traj_state))
        trajs_imgs.append(np.vstack(traj_imgs))
        trajs_action.append(np.vstack(traj_action))
        goals.append(s_t['achieved_goal'].copy())
        desired_goals.append(s_t['desired_goal'].copy())
    return trajs_state, trajs_imgs, trajs_action, goals, desired_goals

class PointEnv(gym.Env):
    def __init__(self):
        self.position = np.array([0.0, 0.0])  # x, y
        self.max_velocity = 0.1
        self.episode_length = 50

        self.goal = np.array([-1.0, -1.0])    # x, y
        self.start = np.array([-1.0, -1.0])    # x, y

        self.observation_space = gym.spaces.Box(-np.inf * np.ones(2).astype(np.float32), np.inf * np.ones(2).astype(np.float32))
        self.action_space = gym.spaces.Box(-np.ones(2).astype(np.float32), np.ones(2).astype(np.float32))

        self.dt = 0.1

        # plotting
        self.xmin = -1
        self.xmax = 1
        self.ymin = 1
        self.ymax = -1


    def reset(self):
        self.position = np.zeros(2)
        self.start = np.zeros(2)
        self.goal = np.random.rand(2) * 2.0 - 1.0
        return self.get_obs()

    def get_obs(self):
        obs = {
            'observation': self.position.copy(),
            'achieved_goal': self.position.copy(),
            'desired_goal': self.goal,
        }
        return obs

    def render(self):
        w, h = figaspect(1.0)
        fig = Figure(figsize=(w,h))
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.scatter(x=self.position[0], y=self.position[1], c='blue', marker='o', s=100)
        ax.scatter(x=self.start[0], y=self.start[1], c='green', marker='v', s=100)
        ax.scatter(x=self.goal[0], y=self.goal[1], c='red', marker='^', s=100)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.tight_layout()

        canvas.draw()       # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        return img

    def expert_action(self):
        heading = np.arctan2(self.goal[1] - self.position[1], self.goal[0] - self.position[0])
        distance = np.linalg.norm(self.goal - self.position)
        orig_distance = np.linalg.norm(self.goal - self.start)
        size_of_step = (orig_distance / self.episode_length) / self.max_velocity # divide by max velocity because it will be cancelled out later
        action = np.array([
            np.cos(heading),
            np.sin(heading),
        ]) * size_of_step

        return action

    def step(self, action):
        self.position += self.max_velocity * action

        return self.get_obs(), self.compute_reward(self.position, self.goal), False, {}

    def compute_reward(self, achieved_goal, desired_goal, info = None):
        return -np.linalg.norm(achieved_goal - desired_goal)

    def close(self):
        self.reset()


def generate_dataset():
    env = PointEnv()
    num_episodes=200
    save_path = os.path.expanduser("~/localdata/vid_classifier/pointmass.pkl")
    trajs_state, trajs_imgs, trajs_action, goals, desired_goals = collect_data(env, num_episodes, env.episode_length)

    pickle_dict = {
        "state": trajs_state,
        "imgs": trajs_imgs,
        "actions": trajs_action,
        "goals": goals,
        "desired_goals": desired_goals,
    }
    pickle.dump(pickle_dict, open(save_path, "wb"))

if __name__ == "__main__":
    generate_dataset()
    exit()

    env = PointEnv()

    traj = []

    obs = env.reset()
    traj.append(obs['observation'])
    plt.figure()
    for i in range(env.episode_length):
        # action = env.action_space.sample()
        action = env.expert_action()
        obs, r, done, _ = env.step(action)
        img = env.render()
        plt.imshow(img)
        plt.show()
        traj.append(obs['observation'])
    traj_np = np.vstack(traj)
    goal = obs['desired_goal']

    plt.figure()
    plt.plot(traj_np[:, 0], traj_np[:, 1], marker='o', color='green')
    plt.plot(goal[0], goal[1], marker='x', color='red')
    plt.show()

    # collect_data(env, 100, env.episode_length)

    env.close()
