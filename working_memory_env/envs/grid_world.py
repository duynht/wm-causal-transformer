import numpy as np
# import pygame

from gymnasium import spaces            
from gymnasium.utils import seeding

from typing import Optional

from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv, Grid, COLORS
# from stimuli import Circle, Square, Triangle
from minigrid.core.world_object import WorldObj
from minigrid.utils.window import Window
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    downsample,
    highlight_img
)

from sympy import divisors

TILE_PIXELS = 32

class DMTSGridEnv(MiniGridEnv):

    def __init__(
        self, 
        grid_size: int = None,
        width: int = None,
        height: int = None, 
        max_delay: Optional[int] = 5, 
        render_mode: Optional[str] = None,
        tile_size: int = TILE_PIXELS, 
    ):
        self.obj_types = ["circle", "square", "triangle"]
        self.obj_list = self._rand_subset(self.obj_types, 2)
        self.target_type = self.obj_list[0]

        self.asked = False

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = self._rand_int(3, 3 + max_delay + 1)

        self.action_space = spaces.Discrete(self.width * self.height + 1)
        self.pending_action = self.width * self.height

        # Observations are dictionaries containing an image of the grid
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.width * tile_size, self.height * tile_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "asked": spaces.Discrete(2),
                "goal": spaces.Discrete(self.width * self.height + 1),
            }
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.window: Window = None

        # Current grid
        self.grid = DMTSGrid(width, height)
       
        # Rendering attributes
        self.render_mode = render_mode
        self.tile_size = tile_size
        self.tile_factors = divisors(tile_size)[:-1]

        self.agent_pos = None
        self.agent_dir = None
        self.goal_pos = self.pending_action

    def _gen_obj(self, obj_type, is_goal=False):
        if obj_type == "circle":
            obj = Circle(self._rand_color(), self._rand_elem(self.tile_factors), is_goal)
        
        elif obj_type == "square":
            obj = Square(self._rand_color(), self._rand_elem(self.tile_factors), is_goal)

        elif obj_type == "triangle":
            obj = Triangle(self._rand_color(), self._rand_elem(self.tile_factors), is_goal)

        else:
            raise ValueError(
                "{} object type given. Object type can only be of values circle, square, and triangle.".format(
                    obj_type
                )
            )

        return obj

    def _gen_grid(self, width, height, empty=False):
        self.grid = DMTSGrid(width, height)
        if not empty:
            if self.asked:
                for obj_type in self.obj_list:
                    obj = self._gen_obj(obj_type, is_goal=(obj_type == self.target_type))
                    x, y = self.place_obj(obj)
                    self.goal_pos = y * self.height + x
            else:
                self.place_obj(self._gen_obj(self.target_type))

    def get_frame(self, tile_size):
        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
        )

        return img

    # TODO: The agent's view should be omnipotent
    def gen_obs(self):
        """
        Generate the agent's view
        """

        # image = self.grid.encode()
        image = self.get_frame(tile_size=self.tile_size)
        
        # Observations are dictionaries containing:
        # - an image (omnipotent view of the environment)
        # - task trigger

        obs = {"image": image, "asked": self.asked, "goal": self.goal_pos}

        return obs

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1

    def step(self, action):
        if self.step_count == 0:
            self._gen_grid(self.width, self.height)

        self.step_count += 1

        if self.step_count == self.max_steps - 1:
            self.asked = True
            self._gen_grid(self.width, self.height)

        reward = 0
        terminated = False
        truncated = False
        if action != self.pending_action:
            self.agent_pos = ((action % self.grid.height), (action // self.grid.height))
            self.agent_dir = True
            selected_cell = self.grid.get(*self.agent_pos)

            if selected_cell is not None and selected_cell.is_goal:
                terminated = True
                reward = self._reward()

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.goal_pos = self.pending_action

        self.max_steps = self._rand_int(3, 9)

        self.obj_list = self._rand_subset(self.obj_types, 2)
        self.target_type = self.obj_list[0]

        self.asked = False

        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height, empty=True)

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def render(self):

        img = self.get_frame(tile_size=self.tile_size)

        if self.render_mode == "human":
            if self.window is None:
                self.window = Window("minigrid")
                self.window.show(block=False)
            # self.window.set_caption(self.mission)
            self.window.show_img(img)
        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            self.window.close()

class DMTSGrid(Grid):
        def __init__(self, width, height):
            super().__init__(width, height)

        @classmethod
        def render_tile(
            cls, obj, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3
        ):
            """
            Render a tile and cache the result
            """

            # Hash map lookup key for the cache
            key = (agent_dir, highlight, tile_size)
            key = (obj.color * 10, obj.type * 10, obj.scale * 10) + key if obj else key

            if key in cls.tile_cache:
                return cls.tile_cache[key]

            # img = np.zeros(
            #     shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
            # )
            img = np.full(
                shape=(tile_size * subdivs, tile_size * subdivs, 3),
                fill_value=255,
                dtype=np.uint8
            )

            # Draw the grid lines (top and left edges)
            # fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
            # fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
            fill_coords(img, point_in_rect(0, 0.031, 0, 1), (0, 0, 0))
            fill_coords(img, point_in_rect(0, 1, 0, 0.031), (0, 0, 0))

            if obj is not None:
                obj.render(img)

            if agent_dir is not None:
                highlight_img(img, color=COLORS["green"])

            # Highlight the cell if needed
            if highlight:
                highlight_img(img)

            # Downsample the image to perform supersampling/anti-aliasing
            img = downsample(img, subdivs)

            # Cache the rendered tile
            cls.tile_cache[key] = img

            return img

        def render(self, tile_size, agent_pos, agent_dir=None, highlight_mask=None):
            """
            Render this grid at a given scale
            :param r: target renderer object
            :param tile_size: tile size in pixels
            """

            if highlight_mask is None:
                highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

            # Compute the total grid size
            width_px = self.width * tile_size
            height_px = self.height * tile_size

            img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

            # Render the grid
            for j in range(0, self.height):
                for i in range(0, self.width):
                    cell = self.get(i, j)

                    agent_here = np.array_equal(agent_pos, (i, j))
                    tile_img = DMTSGrid.render_tile(
                        cell,
                        agent_dir=agent_dir if agent_here else None,
                        highlight=highlight_mask[i, j],
                        tile_size=tile_size,
                    )

                    ymin = j * tile_size
                    ymax = (j + 1) * tile_size
                    xmin = i * tile_size
                    xmax = (i + 1) * tile_size
                    img[ymin:ymax, xmin:xmax, :] = tile_img

            return img


OBJECT_TO_IDX.update(
    {
        "square": 11,
        "circle": 12,
        "triangle": 13
    }
)

class Square(WorldObj):
    def __init__(self, color, scale=1., is_goal=False):
        super().__init__("square", color)
        self.scale = scale
        self.is_goal = is_goal

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), COLORS[self.color])
        downsample(img, self.scale)

class Triangle(WorldObj):
    def __init__(self, color, scale=1., is_goal=False):
        super().__init__("triangle", color)
        self.scale = scale
        self.is_goal = is_goal

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(
            img,
            point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            ),
            COLORS[self.color],
        )
        
        downsample(img, self.scale)

class Circle(WorldObj):
    def __init__(self, color, scale=1., is_goal=False):
        super().__init__("square", color)
        self.scale = scale
        self.is_goal = is_goal

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
        downsample(img, self.scale)