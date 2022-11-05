# from minigrid.core.world_object import WorldObj

# from minigrid.core.constants import COLORS
# from minigrid.utils.rendering import (
#     fill_coords,
#     point_in_circle,
#     point_in_rect,
#     point_in_triangle,
#     downsample,
# )



# class Square(WorldObj):
#     def __init__(self, color, scale=1., is_goal=False):
#         super().__init__("square", color)
#         self.scale = scale
#         self.is_goal = is_goal

#     def can_overlap(self):
#         return True

#     def render(self, img):
#         fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), COLORS[self.color])
#         downsample(img, 1/self.scale)

# class Triangle(WorldObj):
#     def __init__(self, color, scale=1., is_goal=False):
#         super().__init__("triangle", color)
#         self.scale = scale
#         self.is_goal = is_goal

#     def can_overlap(self):
#         return True

#     def render(self, img):
#         fill_coords(
#             img,
#             point_in_triangle(
#                 (0.12, 0.19),
#                 (0.87, 0.50),
#                 (0.12, 0.81),
#             ),
#             COLORS[self.color],
#         )
        
#         downsample(img, 1/self.scale)

# class Circle(WorldObj):
#     def __init__(self, color, scale=1., is_goal=False):
#         super().__init__("square", color)
#         self.scale = scale
#         self.is_goal = is_goal

#     def can_overlap(self):
#         return True

#     def render(self, img):
#         fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
#         downsample(img, 1/self.scale)