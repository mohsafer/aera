"""Interactive pygame viewer — watch the world (and a training run) live.

Used two ways:
  1. attached to a Trainer: `viewer.tick(env)` is called every env step; the
     viewer throttles itself to `fps` and processes input (pause, cameras).
  2. standalone playback: `Viewer.play(env, policy)` runs the env loop here,
     with a random or trained policy — for students to just watch.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

import pygame

from .camera import Camera
from .hud import Chart, draw_feed, draw_help, draw_minimap, draw_vitals
from .renderer3d import render_scene


class Viewer:
    def __init__(self, size=(1280, 720), fps=30, caption="AERA — wake up, learn, survive"):
        import os
        if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
            raise RuntimeError("Viewer needs a display; use --record for headless.")
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 14)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 17, bold=True)
        self.camera = Camera("orbit")
        self.paused = False
        self.step_once = False
        self.show_help = False
        self.quit_requested = False
        self.chart = Chart()
        self.feed: deque = deque(maxlen=40)
        self.events: deque = deque(maxlen=40)
        self._last_render = 0.0
        self._sim_rate = 20.0     # standalone-playback steps per second
        self._acc = 0.0

    # ---------------------------------------------------- trainer hook
    def tick(self, env, episode_ret: float | None = None) -> bool:
        """Called once per env step. Returns False when the user quit."""
        now = pygame.time.get_ticks() / 1000.0
        for ev in getattr(env, "last_events", ()):
            self.events.append(f"[ep {env.episode}] {ev}")
        self._handle_input(env)
        if now - self._last_render >= 1.0 / self.fps:
            self._last_render = now
            self._draw(env)
        if episode_ret is not None:
            self.chart.push(episode_ret)
        return not self.quit_requested

    # ------------------------------------------------------ standalone
    def play(self, env, policy=None, seed: int = 0):
        """Run the env loop here: policy=None → random actions; pass a
        `aera.rl.ppo_numpy.PPO` (or any object with .act(obs)) for playback."""
        obs, _ = env.reset(seed=seed)
        running = True
        while running and not self.quit_requested:
            self.clock.tick(60)
            self._acc += self.clock.get_time() / 1000.0
            step_dt = 1.0 / self._sim_rate
            act_now = (not self.paused) or self.step_once
            while self._acc >= step_dt and act_now:
                self._acc -= step_dt
                a = (policy.act(obs, deterministic=True)[0]
                     if policy is not None else env.action_space.sample())
                obs, _, term, trunc, info = env.step(a)
                for ev in info["events"]:
                    self.events.append(f"[ep {env.episode}] {ev}")
                if term or trunc:
                    ret = info["episode"]["ret"]
                    self.chart.push(ret)
                    obs, _ = env.reset()
                if self.step_once:
                    self.step_once = False
                    break
            self._handle_input(env)
            self._draw(env)
        pygame.quit()

    # ----------------------------------------------------------- internals
    def _handle_input(self, env):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.quit_requested = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.step_once = True
                elif event.key == pygame.K_v:
                    self.camera.cycle_mode()
                elif event.key == pygame.K_f:
                    self.camera.follow = not self.camera.follow
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_LEFT:
                    self.camera.yaw -= 0.15
                elif event.key == pygame.K_RIGHT:
                    self.camera.yaw += 0.15
                elif event.key == pygame.K_UP:
                    self.camera.pitch = min(1.4, self.camera.pitch + 0.1)
                elif event.key == pygame.K_DOWN:
                    self.camera.pitch = max(0.08, self.camera.pitch - 0.1)
            elif event.type == pygame.MOUSEWHEEL:
                self.camera.dist = min(40.0, max(2.0, self.camera.dist - event.y * 1.5))
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                self.camera.yaw -= event.rel[0] * 0.005
                self.camera.pitch = min(1.4, max(0.05, self.camera.pitch + event.rel[1] * 0.004))

    def _draw(self, env):
        W, H = self.screen.get_size()
        panel_w = 360
        vw, vh = W - panel_w, H

        # chunky 3D: render at 1/2 window size, nearest-upscale for the look
        small = render_scene(env.world, [env.agent], self.camera,
                             (max(64, vw // 2), max(64, vh // 2)),
                             time_s=pygame.time.get_ticks() / 1000.0)
        frame = pygame.transform.scale(small, (vw, vh))
        self.screen.blit(frame, (0, 0))

        x = vw + 10
        mp = min(panel_w - 20, 250)
        draw_vitals(self.screen, env, pygame.Rect(x, 10, panel_w - 20, 150),
                    self.font, self.big)
        draw_minimap(self.screen, env.world, env.agent,
                     pygame.Rect(x, 170, mp, mp), self.font)
        self.chart.draw(self.screen, pygame.Rect(x, 180 + mp, panel_w - 20, 110))
        feed_y = 300 + mp
        draw_feed(self.screen, self.events,
                  pygame.Rect(x, feed_y, panel_w - 20, H - feed_y - 10),
                  self.font)
        mode = f"cam: {self.camera.mode}  dist {self.camera.dist:.0f}  {'[paused]' if self.paused else ''}"
        self.screen.blit(self.font.render(mode, True, (232, 230, 220)), (12, 10))
        if self.show_help:
            draw_help(self.screen, self.font)
        pygame.display.flip()
