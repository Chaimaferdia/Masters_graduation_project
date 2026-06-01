import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TimetableEnv(gym.Env):

    def __init__(
        self,
        teachers,
        rooms,
        student_groups,
        courses,
        days_per_week,
        periods_per_day,
    ):
        super().__init__()

        self.teachers = teachers
        self.rooms = rooms
        self.student_groups = student_groups
        self.courses = courses

        self.days_per_week = days_per_week
        self.periods_per_day = periods_per_day

        self.action_space = spaces.Discrete(32)

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(5, 7, len(student_groups)),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(32,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, seed=None, options=None):
        self.current_course_index = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "grid": np.zeros(
                (5, 7, len(self.student_groups)),
                dtype=np.float32,
            ),
            "action_mask": self.get_action_mask(),
        }

    def _action_to_time(self, action):
        """
        Converts action index into
        timetable position.
        """
        pass

    def get_action_mask(self):
        """
        Invalid Action Masking.
        Prevents selecting actions
        violating hard constraints.
        """
        return np.ones(32, dtype=np.float32)

    def _find_available_room(self, session, day, period):
        """
        Finds a suitable room
        according to session type.
        """
        pass

    def step(self, action):
        """
        1. Validate action
        2. Schedule session
        3. Compute reward
        4. Move to next session
        """
        terminated = False
        reward = 0.0

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {},
        )