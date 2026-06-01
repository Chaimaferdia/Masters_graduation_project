import gymnasium as gym

from sb3_contrib import MaskablePPO
from timetable_env import TimetableEnv


class CustomActionMasker(gym.Wrapper):

    def __init__(self, env, mask_fn):
        super().__init__(env)
        self.mask_fn = mask_fn

    def action_masks(self):
        return self.mask_fn(self.env)


def make_env():
    """
    Builds scheduling environment.
    """
    pass


policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256, 128],
        vf=[256, 256, 128],
    )
)


def train():

    env = make_env()

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=2e-4,
        gamma=0.995,
        n_steps=4096,
        clip_range=0.2,
        ent_coef=0.03,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    model.learn(
        total_timesteps=1_000_000
    )

    model.save("ppo_timetable_final")


if __name__ == "__main__":
    train()