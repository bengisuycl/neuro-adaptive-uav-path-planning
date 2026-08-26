import copy
import math
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from wcci_conference_project.aircraft.jet_model import Aircraft
from wcci_conference_project.ai_modules.dqn_pilot import F16Agent
from wcci_conference_project.ai_modules.generate_real_dataset import resolve_dem_path
from wcci_conference_project.environment.terrain import Terrain
from wcci_conference_project.planners.rl_planner import RLPlanner
from wcci_conference_project.scenarios.definitions import get_all_scenarios


EPISODES = int(os.environ.get("WCCI_RL_EPISODES", "4000"))
MAX_STEPS = int(os.environ.get("WCCI_RL_MAX_STEPS", "360"))
LOG_INTERVAL = int(os.environ.get("WCCI_RL_LOG_INTERVAL", "250"))
SAVE_DRIVE_COPY = os.environ.get("WCCI_RL_DRIVE_BACKUP", "1") == "1"
SAFE_ALT = float(os.environ.get("WCCI_RL_SAFE_ALT", "1000.0"))
SCENARIO_PROBS = np.asarray(
    [
        float(os.environ.get("WCCI_RL_P_S1", "0.10")),
        float(os.environ.get("WCCI_RL_P_S2", "0.30")),
        float(os.environ.get("WCCI_RL_P_S3", "0.35")),
        float(os.environ.get("WCCI_RL_P_S4", "0.25")),
    ],
    dtype=float,
)
if np.sum(SCENARIO_PROBS) <= 0.0:
    SCENARIO_PROBS = np.asarray([0.10, 0.30, 0.35, 0.25], dtype=float)
SCENARIO_PROBS = SCENARIO_PROBS / np.sum(SCENARIO_PROBS)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DEM_PATH = resolve_dem_path(os.environ.get("WCCI_DEM_PATH"))
SAVE_PATH = os.environ.get(
    "WCCI_RL_SAVE_PATH",
    os.path.join(project_root, "wcci_conference_project", "f16_dqn_weights.pth"),
)


def calculate_reward(prev_dist, curr_dist, prev_nav_dist, curr_nav_dist, is_collision, is_success, next_state, revisit_count):
    reward = 0.0

    progress = prev_dist - curr_dist
    nav_progress = prev_nav_dist - curr_nav_dist
    reward += progress * 0.10
    reward += nav_progress * 0.14
    reward -= 0.12

    heading_err = abs(float(next_state[1]))
    risk_vec = np.asarray(next_state[-5:], dtype=float)
    center_risk = float(risk_vec[2])
    side_risk = float(np.mean(np.r_[risk_vec[:2], risk_vec[3:]]))
    reward -= 4.8 * center_risk
    reward -= 1.8 * side_risk
    reward -= 1.0 * heading_err
    reward -= 0.6 * min(revisit_count, 3)

    if is_collision:
        reward -= 350.0
    if is_success:
        reward += 700.0

    return float(reward)


def train():
    print("Training Environment Initializing...")
    print(f"DEM Path Check: {DEM_PATH}")
    try:
        terrain = Terrain(DEM_PATH)
    except Exception:
        print(f"HATA: DEM dosyasi bulunamadi: {DEM_PATH}")
        return

    aircraft = Aircraft(V_cruise=220.0)
    scenarios = get_all_scenarios(safe_alt=SAFE_ALT)
    planner = RLPlanner(terrain, scenarios[0]["threats"], aircraft, safe_alt=SAFE_ALT, verbose=False)
    guide_cache = {}

    agent = planner.agent if planner.agent is not None else F16Agent(state_dim=8, action_dim=5)
    planner.agent = agent
    planner.dqn = agent.policy_net

    print(f"Starting RL-Pilot training ({EPISODES} episodes)...")
    print("Strategy: guide-aligned mixed scenarios + threat jitter + long-range stabilization")

    target_sync = 20
    episode_rewards = []
    for ep in tqdm(range(EPISODES)):
        # Slightly bias training toward the harder scenarios where RL-Pilot previously underperformed.
        scen_idx = np.random.choice([0, 1, 2, 3], p=SCENARIO_PROBS)
        scen = scenarios[scen_idx]
        scen_id = scen["id"]

        goal_pos = np.array(scen["goal"][:2], dtype=float)
        start_pos = np.array(scen["start"][:2], dtype=float)

        active_threats = copy.deepcopy(scen["threats"])
        for th in active_threats:
            th.x += np.random.uniform(-800.0, 800.0)
            th.y += np.random.uniform(-800.0, 800.0)
            if hasattr(th, "reset"):
                th.reset()

        planner.threats = active_threats
        if planner.feasibility is not None:
            planner.feasibility.threats = active_threats
        if scen_id not in guide_cache:
            guide_cache[scen_id] = planner._compute_guide_path(scen["start"], scen["goal"])
        guide_path = guide_cache[scen_id]

        curr_pos = start_pos + np.random.uniform(-250.0, 250.0, 2)
        curr_h = math.degrees(math.atan2(goal_pos[1] - curr_pos[1], goal_pos[0] - curr_pos[0]))
        visited = {}
        nav_xy = planner._guide_target(guide_path, curr_pos[0], curr_pos[1], goal_pos[0], goal_pos[1])

        state = planner._get_state(curr_pos, curr_h, nav_xy).numpy()
        prev_dist = float(np.linalg.norm(curr_pos - goal_pos))
        prev_nav_dist = float(np.linalg.norm(curr_pos - np.asarray(nav_xy, dtype=float)))

        ep_reward = 0.0
        for _ in range(MAX_STEPS):
            action_idx = agent.select_action(state, is_training=True)
            dist_to_goal = float(np.linalg.norm(goal_pos - curr_pos))
            step_size = float(np.clip(900.0 + 0.045 * dist_to_goal, 1000.0, 2300.0))

            for th in active_threats:
                if getattr(th, "dynamic", False):
                    th.update(1.0)

            nx, ny, nh = planner._step_kinematics(curr_pos[0], curr_pos[1], curr_h, planner.actions[action_idx], step_size)
            ok, _, reason = planner.feasibility.check_segment((curr_pos[0], curr_pos[1], planner.safe_alt), (nx, ny, planner.safe_alt), planner.safe_alt)

            new_pos = np.array([nx, ny], dtype=float)
            curr_dist = float(np.linalg.norm(new_pos - goal_pos))
            nav_xy = planner._guide_target(guide_path, nx, ny, goal_pos[0], goal_pos[1])
            curr_nav_dist = float(np.linalg.norm(new_pos - np.asarray(nav_xy, dtype=float)))
            is_success = curr_dist < 1800.0
            is_collision = (not ok) or (reason in {"boundary", "terrain", "hard_threat"})

            vkey = planner._visit_key(nx, ny)
            visited[vkey] = visited.get(vkey, 0) + 1

            next_state = planner._get_state(new_pos, nh, nav_xy).numpy()
            reward = calculate_reward(
                prev_dist,
                curr_dist,
                prev_nav_dist,
                curr_nav_dist,
                is_collision,
                is_success,
                next_state,
                visited[vkey],
            )
            ep_reward += reward

            done = is_collision or is_success
            agent.memory.append((state, action_idx, reward, next_state, done))
            agent.train_step()

            state = next_state
            curr_pos = new_pos
            curr_h = nh
            prev_dist = curr_dist
            prev_nav_dist = curr_nav_dist

            if done:
                break

        episode_rewards.append(ep_reward)

        if ep % target_sync == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if ep % 15 == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.985)
        if (ep + 1) % LOG_INTERVAL == 0:
            avg_reward = float(np.mean(episode_rewards[-LOG_INTERVAL:]))
            print(
                f"  Episode {ep + 1:04d}: "
                f"avg_reward={avg_reward:.2f}, "
                f"epsilon={agent.epsilon:.3f}"
            )

    print(f"Training Complete. Saving model to {SAVE_PATH}...")
    torch.save(agent.policy_net.state_dict(), SAVE_PATH)
    print("Model Saved locally.")

    drive_mount = "/content/drive/My Drive"
    if SAVE_DRIVE_COPY and os.path.exists(drive_mount):
        msc_path = os.path.join(drive_mount, "msc")
        dest_dir = msc_path if os.path.exists(msc_path) else drive_mount
        dest_file = os.path.join(dest_dir, os.path.basename(SAVE_PATH))
        try:
            shutil.copy(SAVE_PATH, dest_file)
            print(f"Drive backup saved: {dest_file}")
        except Exception:
            pass


if __name__ == "__main__":
    train()
