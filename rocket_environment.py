import colorsys
import pygame
import numpy as np
from enum import IntEnum, Enum

from pygame import Vector2
from perlin_numpy import generate_perlin_noise_2d
import math

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700

OBSTACLE_WIDTH = 80
OBSTACLE_GAP = 250


class Action(IntEnum):
    RELEASED = 0
    PRESSED = 1

class ObstacleType(Enum):
    TOP = 0
    BOTTOM = 1


class Player:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, 58, 18)
        self.vel = 0
        self.gravity = 0.3
        self.thrust = -0.8
        self.image = None
        self.rotation = 0

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def update(self, action: Action):
        if action == Action.PRESSED:
            self.vel += self.thrust
        self.vel += self.gravity
        self.rect.y += int(self.vel)
        self.rotation = -min(max(self.vel * 4, -45),45)

        # Clamp to screen
        if self.rect.top < 0:
            self.rect.top = 0
            self.vel = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.vel = 0

    def load_resources(self):
        self.image = pygame.image.load("assets/jet.png")
        self.image = pygame.transform.scale(self.image, (self.rect.width, self.rect.height))

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, rotated_rect)




class Obstacle:

    def __init__(self, point1: Vector2, point2: Vector2, type: ObstacleType, total_x):
        self.point1 = point1
        self.point2 = point2
        self.type = type
        self.total_x = total_x



    def draw(self, screen):
        laser_color = (int(103), int( 103), int( 90))
        if self.type == ObstacleType.TOP:
            points = [self.point1, self.point2, Vector2(self.point2.x, 0), Vector2(self.point1.x, 0)]
            pygame.draw.polygon(screen, laser_color, points)
        else:
            points = [self.point1, self.point2, Vector2(self.point2.x, SCREEN_HEIGHT),
                      Vector2(self.point1.x, SCREEN_HEIGHT)]
            pygame.draw.polygon(screen, laser_color, points)

    def collides_with_rect(self, rect):
        return rect.clipline(self.point1, self.point2)

    def load_resources(self):
        # self.image = pygame.image.load("assets/stone.jpg")
        # self.image = pygame.transform.scale(self.image, (self.rect.width, self.rect.height))
        # self.image = pygame.transform.flip(self.image, True, False)
        pass


class RocketEnvironment:
    def __init__(self, graphics_on=True, steps_before_truncation=4000):
        self.background_color = (30, 30, 30)
        self.player = None
        self.obstacles = []

        self.terminated = False
        self.truncated = False
        self.graphics_on = graphics_on
        self.steps_since_episode = 0
        self.steps_before_truncation = steps_before_truncation

        self.noise = None
        self.noise_size = 1024
        self.total_level = 0
        self.reset()
        self.laser_hue = 0.0

    def reset(self):
        self.noise = generate_perlin_noise_2d((self.noise_size, self.noise_size), (4, 2), tileable=(True, True))
        self.terminated = False
        self.truncated = False
        self.steps_since_episode = 0
        self.obstacles = []
        self.generate_random_obstacles(keep_middle_clear = True, n=50)
        self.total_level = 0

        self.player = Player()
        self.player.set_position(300, SCREEN_HEIGHT / 2)
        if self.graphics_on:
            self.load_resources()
        return self.get_state()

    def step(self, action):
        self.player.update(action)
        for obstacle in self.obstacles:
            if obstacle.collides_with_rect(self.player.rect):
                self.terminated = True

        # delete old obstacles
        if self.obstacles[0].point1.x < -OBSTACLE_WIDTH:
            self.obstacles.pop(0)

        # generate new obstacles
        RENDER_OFFSET = 100
        if self.obstacles[-1].point1.x + OBSTACLE_WIDTH - RENDER_OFFSET < SCREEN_WIDTH:
            self.generate_random_obstacles(offset_x = SCREEN_WIDTH + RENDER_OFFSET)

        self.move_obstacles()

        reward = 1 if not self.terminated else -100
        if self.steps_since_episode > self.steps_before_truncation:
            self.truncated = True

        self.steps_since_episode += 1
        return self.get_state(), reward, self.terminated, self.truncated

    def vertical_ray_segment_intersection_down(self,point, p_seg1, p_seg2):
        px,py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2
        # Check if segment crosses vertical line x = px
        if (x1 - px) * (x2 - px) > 0:
            return None  # Both endpoints on same side of vertical line

        # Handle vertical segment
        if x1 == x2:
            if x1 != px:
                return None
            y_top = min(y1, y2)
            y_bottom = max(y1, y2)
            if py <= y_bottom:
                return max(py, y_top) - py
            return None

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_intersect = m * px + b

        if min(y1, y2) <= y_intersect <= max(y1, y2) and y_intersect >= py:
            return y_intersect - py
        return None

    def vertical_ray_segment_intersection_up(self,point, p_seg1, p_seg2):
        px, py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2
        if (x1 - px) * (x2 - px) > 0:
            return None

        if x1 == x2:
            if x1 != px:
                return None
            y_top = min(y1, y2)
            y_bottom = max(y1, y2)
            if py >= y_top:
                return py - min(py, y_bottom)
            return None

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_intersect = m * px + b

        if min(y1, y2) <= y_intersect <= max(y1, y2) and y_intersect <= py:
            return py - y_intersect
        return None
    def ray_segment_intersection(self,point, angle_deg, p_seg1, p_seg2):
        px, py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2

        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        rx, ry = dx, dy
        sx, sy = x2 - x1, y2 - y1
        denom = rx * sy - ry * sx
        if abs(denom) < 1e-10:
            return None  # Parallel, no intersection

        # Solve for t and u
        t_num = (x1 - px) * sy - (y1 - py) * sx
        u_num = (x1 - px) * ry - (y1 - py) * rx
        t = t_num / denom
        u = u_num / denom

        if t >= 0 and 0 <= u <= 1:
            intersection_x = px + t * rx
            intersection_y = py + t * ry
            distance = math.hypot(intersection_x - px, intersection_y - py)
            return distance
        return None

    def get_obstacle_distances(self):
        min_dist_down = float('inf');
        min_dist_up = float('inf');
        min_dist_right = float('inf');
        for obstacle in self.obstacles:
            curr_dist_down = self.vertical_ray_segment_intersection_down(self.player.rect.center, obstacle.point1,
                                                                         obstacle.point2)
            curr_dist_up = self.vertical_ray_segment_intersection_up(self.player.rect.center, obstacle.point1,
                                                                     obstacle.point2)

            curr_dist_right = self.ray_segment_intersection(self.player.rect.center,0, obstacle.point1,
                                                                     obstacle.point2)
            if curr_dist_down != None and curr_dist_down < min_dist_down:
                min_dist_down = curr_dist_down

            if curr_dist_up != None and curr_dist_up < min_dist_up:
                min_dist_up = curr_dist_up

            if curr_dist_right != None and curr_dist_right< min_dist_right:
                min_dist_right= curr_dist_right
        return min_dist_down, min_dist_up, min_dist_right


    def get_state(self):
        # y pos and vel are normalized
        dist_obst_down, dist_obst_up,dist_obst_right = self.get_obstacle_distances()
        # clamp to avoide infinity
        dist_obst_down = min(dist_obst_down, SCREEN_HEIGHT)
        dist_obst_up = min(dist_obst_up , SCREEN_HEIGHT)
        dist_obst_right = min(dist_obst_right, SCREEN_WIDTH)

        state = np.array([
            self.player.rect.y / SCREEN_HEIGHT,
            self.player.vel / 10.0,
            dist_obst_down / SCREEN_HEIGHT,
            dist_obst_up / SCREEN_HEIGHT,
            dist_obst_right / SCREEN_WIDTH,
        ], dtype=np.float32)

        return state

    def line_to_draw(self,point,angle_deg, p_seg1, p_seg2):
        px, py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2

        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        rx, ry = dx, dy
        sx, sy = x2 - x1, y2 - y1
        denom = rx * sy - ry * sx
        if abs(denom) < 1e-10:
            return (None,None),None  # Parallel, no intersection

        # Solve for t and u
        t_num = (x1 - px) * sy - (y1 - py) * sx
        u_num = (x1 - px) * ry - (y1 - py) * rx
        t = t_num / denom
        u = u_num / denom

        if t >= 0 and 0 <= u <= 1:
            intersection_x = px + t * rx
            intersection_y = py + t * ry
            distance = math.hypot(intersection_x - px, intersection_y - py)
            return (intersection_x, intersection_y), distance

        return (None,None),None

    def draw(self,screen):
        self.laser_hue = (self.laser_hue + 0.005) % 1.0
        assert self.graphics_on
        screen.fill(self.background_color)
        min_distances = dict()
        min_distances[45]=(float('inf'),(None,None))
        min_distances[-45]=(float('inf'),(None,None))
        min_distances[0]=(float('inf'),(None,None))
        r, g, b = colorsys.hsv_to_rgb(self.laser_hue, 1.0, 1.0)
        laser_color = (int(r * 255), int(g * 255), int(b * 255))
        pygame.draw.line(screen, laser_color, (self.player.rect.centerx,0), (self.player.rect.centerx,SCREEN_HEIGHT), 2)
        for obstacle in self.obstacles:
            obstacle.draw(screen)
            #pygame.draw.circle(screen, laser_color, obstacle.point1, 10)
            for angle in min_distances:
                (x, y),dist = self.line_to_draw(self.player.rect.center, angle, obstacle.point1, obstacle.point2)
                if x != None and y != None:
                   if dist < min_distances[angle][0]:
                       min_distances[angle] = (dist,(x,y))


        for angle in min_distances:
            if min_distances[angle][1] != (None,None):
                pygame.draw.line(screen, laser_color, self.player.rect.center, min_distances[angle][1], 2)
        if min_distances[0][1] == (None,None):
            pygame.draw.line(screen, laser_color, self.player.rect.center, (SCREEN_WIDTH,self.player.rect.centery), 2)
        pygame.draw.line(screen, laser_color, self.player.rect.center, (SCREEN_WIDTH,self.player.rect.centery), 2)

        self.player.draw(screen)
        pygame.display.flip()

    def close(self):
        if self.graphics_on:
            pygame.quit()

    def generate_random_obstacles(self, n=20, offset_x = 0, keep_middle_clear = False):
        last_x = None
        last_y = None

        total_x = 0
        if len(self.obstacles) > 0:
            self.total_level += 1
            total_x = self.obstacles[-2].total_x - OBSTACLE_WIDTH # -2 because the last is the bottom one and the second last is the top one
            last_x = self.obstacles[-2].point2.x
            last_y = self.obstacles[-2].point2.y

        for i in range(n):
            offset_obstacles_y = np.interp(i, [0, n], [150, 5]) if keep_middle_clear else 5
            obstacle_gap = np.interp(i, [0, n], [self.get_obstacle_size(self.total_level - 1), self.get_obstacle_size(self.total_level)])
            total_x = (total_x + OBSTACLE_WIDTH) % self.noise_size
            x = offset_x + i * OBSTACLE_WIDTH
            y = np.interp(self.noise[total_x, total_x], [-1, 1], [offset_obstacles_y, SCREEN_HEIGHT - offset_obstacles_y - obstacle_gap])
            if last_x is not None and last_y is not None:
                obstacle = Obstacle(Vector2(last_x, last_y), Vector2(x, y), ObstacleType.TOP, total_x)
                self.obstacles.append(obstacle)
                obstacle = Obstacle(Vector2(last_x, last_y + obstacle_gap), Vector2(x, y + obstacle_gap), ObstacleType.BOTTOM,
                                    total_x)
                self.obstacles.append(obstacle)
            last_x = x
            last_y = y
    def get_obstacle_size(self, level):
        return max(-10 * level + 300, 150)

    def move_obstacles(self, speed = 5):
        for obstacle in self.obstacles:
            obstacle.point1.x -= speed
            obstacle.point2.x -= speed

    def load_resources(self):
        self.player.load_resources()
