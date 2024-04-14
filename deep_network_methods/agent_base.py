from tensorflow import keras
from keras import __version__
keras.__version__ = __version__

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class AgentBase:

    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.memory = SequentialMemory(limit=50000, window_length=1)
        self.policy = BoltzmannQPolicy()
        self.dqn = DQNAgent(model=self.model, nb_actions=self.env.action_space.n, memory=self.memory,
                            nb_steps_warmup=10,
                            target_model_update=1e-2, policy=self.policy)
        self.dqn.compile(optimizer='adam', metrics=['mae'])

    def train(self, nb_steps):
        self.dqn.fit(self.env, nb_steps=nb_steps, visualize=False, verbose=1)

    def test(self, nb_episodes):
        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=False)

    def save(self, filename):
        self.dqn.save_weights(filename, overwrite=True)

    def load(self, filename):
        self.dqn.load_weights(filename)
