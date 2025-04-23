import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO, TD3, DDPG
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

# ------------------------------------------------------------------------
# 1) Always create a 1200-step speed dataset for each speed profile.
#    Generate 3 additional lead vehicle profiles.
# ------------------------------------------------------------------------
DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"

# Define constants between environments
MIN_DISTANCE = 5.0
MAX_DISTANCE = 30.0
MAX_SAFE_ACCELERATION = 2.0
MAX_SAFE_DECELERATION = -2.0

# Define a global training step tracker
GLOBAL_STEP = 0

# Define the generation of lead vehicle profiles
# Create difficulty-based profiles using shared reference speed
def generate_difficulty_profiles(ref_speed, timesteps=1200):
    t = np.arange(timesteps)

    profiles = {
        "easy": ref_speed + 0.25 * np.sin(0.05 * t) + 0.2 * np.random.randn(timesteps),
        "medium": ref_speed + 1.0 * np.sin(0.05 * t) + 0.5 * np.random.randn(timesteps),
        "difficult": ref_speed + 4.0 * np.sin(0.05 * t) + 1.2 * np.random.randn(timesteps)
    }

    return {k: np.cumsum(v) for k, v in profiles.items()}, profiles

# Force-generate (or load) the reference 1200-step sinusoidal + noise speed profile
if not os.path.exists(CSV_FILE):
    speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
    df_fake = pd.DataFrame({"speed": speeds})
    df_fake.to_csv(CSV_FILE, index=False)
    print(f"Created {CSV_FILE} with {DATA_LEN} steps.")
    print(f"Created ALL difficulty profiles with {DATA_LEN} steps.")
else:
    print(f"{CSV_FILE} already exists. Skipping creation.")


df = pd.read_csv(CSV_FILE)
REF_SPEED_DATA = df["speed"].values
assert len(REF_SPEED_DATA) == DATA_LEN, "Dataset must be 1200 steps after generation."

# Create difficulty-based lead vehicle profiles
LEAD_POSITION_PROFILES, LEAD_SPEED_PROFILES = generate_difficulty_profiles(REF_SPEED_DATA, DATA_LEN)
print(f"Loaded and generated difficulty profiles with {DATA_LEN} steps each.")

# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
# ------------------------------------------------------------------------
def chunk_into_episodes(data, chunk_size):
    """
    Splits `data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        episodes.append(chunk)
        start = end
    return episodes

def chunk_positions_into_episodes(positions_data, chunk_size):
    """
    Splits `positions_data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    Used to segment lead vehicle positions into episodes for adaptive cruise control.
    """
    positions = []
    start = 0
    while(start < len(positions_data)):
        end = start + chunk_size
        position_chunk = positions_data[start:end]
        positions.append(position_chunk)
        start = end
    return positions


# Define an error/reward function
def calc_error(current_speed, ref_speed, distance_to_lead, jerk, accel):

    # Speed error (squared to penalize large deviations)
    speed_error = abs(current_speed - ref_speed)

    # Distance error (scale factor of displacement from max/min distance)
    # Penalize being too close much more than being too far away
    if distance_to_lead < MIN_DISTANCE:
        distance_error = 3.0 * abs(MIN_DISTANCE - distance_to_lead)
    elif distance_to_lead > MAX_DISTANCE:
        distance_error = abs(MAX_DISTANCE - distance_to_lead)
    else:
        distance_error = 0

    # Calculate a jerk error to influence smooth movement
    # 1. Jerk = difference in accelerations
    # 2. Also, minimize accel to promote limited changes in force
    jerk_error = abs(jerk) + 0.5 * abs(accel)

    # Combine all terms (weight distance error more significantly than speed error to overcorrect distance errors)
    total_error = speed_error + 3.0 * distance_error + 0.1 * jerk_error
    return speed_error, distance_error, total_error

# Define curriculum schedule functions
def curriculum_schedule(step):
    if step < 33333:
        return "easy"
    elif step < 66666:
        return "medium"
    else:
        return "difficult"

def anti_curriculum_schedule(step):
    if step < 33333:
        return "difficult"
    elif step < 66666:
        return "medium"
    else:
        return "easy"

def baseline_schedule(step):
    return "difficult"

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):

    def __init__(self, ref_data, lead_positions, schedule_fn, chunk_size, delta_t=1.0):
        super().__init__()

        # Store speed/position metrics
        self.ref_data = ref_data
        self.lead_positions_dict = lead_positions
        self.schedule_fn = schedule_fn
        self.chunk_size = chunk_size
        self.delta_t = delta_t

        # Create the action and observation spaces following RL2
        self.action_space = spaces.Box(low=MAX_SAFE_DECELERATION, high=MAX_SAFE_ACCELERATION, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([50.0, 50.0, 100.0]), dtype=np.float32)

        # Define episode specific metrics
        self.current_episode = None
        self.current_position = None
        self.step_idx = 0
        self.episode_len = 0
        self.current_speed = 0.0
        self.ref_speed = 0.0
        self.prev_accel = 0.0
        self.ego_position = 0.0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        global GLOBAL_STEP

        # Split into episodes based on curriculum
        difficulty = self.schedule_fn(GLOBAL_STEP)
        ref_chunks = chunk_into_episodes(self.ref_data, self.chunk_size)
        lead_chunks = chunk_positions_into_episodes(self.lead_positions_dict[difficulty], self.chunk_size)

        # Initialize current position metrics
        ep_idx = np.random.randint(0, len(ref_chunks))
        self.current_episode = ref_chunks[ep_idx]
        self.current_position = lead_chunks[ep_idx]

        # Reset episode specific metrics
        self.step_idx = 0
        self.current_speed = 0.0
        self.ref_speed = self.current_episode[self.step_idx]
        self.prev_accel = 0.0
        self.ego_position = self.current_position[self.step_idx] - 17.5

        obs = np.array([self.current_speed, self.ref_speed, self.current_position[self.step_idx] - self.ego_position], dtype=np.float32)
        info = {}
        return obs, info


    def step(self, action):

        # Force realistic acceleration limits
        accel = np.clip(action[0], MAX_SAFE_DECELERATION, MAX_SAFE_ACCELERATION)

        # Calculate jerk
        jerk = (accel - self.prev_accel) / self.delta_t
        self.prev_accel = accel

        # Calculate ego vehicle state
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        # Update position based on speed
        self.ego_position += self.current_speed * self.delta_t

        # Get the reference speed
        self.ref_speed = self.current_episode[self.step_idx]

        # Get lead vehicle state
        distance_to_lead = self.current_position[self.step_idx] - self.ego_position

        # Calculate the error and reward
        speed_error, distance_error, error = calc_error(self.current_speed, self.ref_speed, distance_to_lead, jerk, accel)
        reward = -error

        # Update iteration information
        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False

        # Return system information
        obs = np.array([self.current_speed, self.ref_speed, distance_to_lead], dtype=np.float32)
        info = {"speed_error": speed_error, "distance_error": distance_error, "distance": distance_to_lead, "jerk": jerk}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    """

    def __init__(self, full_data, lead_positions, delta_t=1.0):
        super().__init__()

        # Original metrics
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        # Reference all of the lead vehicle's positions
        self.lead_positions = lead_positions

        # Define the same action and observation space as the training environment
        self.action_space = spaces.Box(low=MAX_SAFE_DECELERATION, high=MAX_SAFE_ACCELERATION, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32), high=np.array([50.0, 50.0, 100.0], dtype=np.float32), dtype=np.float32)

        # Tracked per episode
        self.idx = 0
        self.current_speed = 0.0
        self.prev_accel = 0.0
        self.ego_position = 0.0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use ALL episodes
        self.idx = 0
        self.current_speed = 0.0
        self.prev_accel = 0.0
        ref_speed = self.full_data[self.idx]

        # Start our vehicle at a safe distance behind the lead vehicle ([30 + 5] / 2)
        self.ego_position = self.lead_positions[self.idx] - 17.5

        obs = np.array([self.current_speed, ref_speed, self.lead_positions[self.idx] - self.ego_position], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):

        # Force realistic acceleration limits
        accel = np.clip(action[0], MAX_SAFE_DECELERATION, MAX_SAFE_ACCELERATION)

        # Calculate jerk
        jerk = (accel - self.prev_accel) / self.delta_t
        self.prev_accel = accel

        # Update ego vehicle state
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0
        
        # Update position based on speed
        self.ego_position += self.current_speed * self.delta_t

        # Get reference speed
        self.ref_speed = self.full_data[self.idx]

        # Get lead vehicle state
        distance_to_lead = self.lead_positions[self.idx] - self.ego_position

        # Calculate the error and reward
        speed_error, distance_error, error = calc_error(self.current_speed, self.ref_speed, distance_to_lead, jerk, accel)
        reward = -1 * error

        # Calculate the speed of the lead vehicle using displacement
        if self.idx > 0:
            lead_speed = (self.lead_positions[self.idx] - self.lead_positions[self.idx - 1]) / self.delta_t
        else:
            lead_speed = 0

        # Update iteration information
        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        # Return same metrics as training, with additional lead speed
        obs = np.array([self.current_speed, self.ref_speed, distance_to_lead], dtype=np.float32)
        info = {"speed_error": speed_error, "distance_error": distance_error, "distance": distance_to_lead, "jerk": jerk,"lead_speed": lead_speed}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (optional)
# ------------------------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward'])

    def _on_step(self):
        global GLOBAL_STEP
        GLOBAL_STEP += 1
        difficulty = self.model.get_env().envs[0].schedule_fn(GLOBAL_STEP)
        if self.verbose > 0 and GLOBAL_STEP % 1000 == 0:
            print(f"[TRAINING] Step {GLOBAL_STEP}: Difficulty = {difficulty}")

        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        self.episode_rewards.append(reward)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward])
            self.logger.record("reward/average_reward", avg_reward)
            self.episode_rewards.clear()

        return True


# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_chunk_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="RL Model Index (0-3)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (Default = 3e-4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (Default = 256)"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=200000,
        help="Buffer size (Default = 200000)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Tau (Default = 0.005)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Gamma (Default = 0.99)"
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=-1.0,
        help="Gamma (Auto = -1.0, Rest = Float)"
    )
    parser.add_argument(
        "--net_arch",
        type=str,
        default="256,256",
        help="Net Arch defined as 2 integers (Default = 256,256)"
    )
    parser.add_argument(
        "--curriculum_type",
        type=str,
        choices=["curriculum", "anti", "baseline"],
        default="baseline",
        help="Which curriculum strategy to use"
    )
    args = parser.parse_args()
    if args.model == 1:
        ent_coef_param = 0.0 if args.ent_coef == -1.0 else args.ent_coef
    else:
        ent_coef_param = 'auto' if args.ent_coef == -1.0 else args.ent_coef
    net_arch_param = list(map(int, args.net_arch.split(",")))
    
    # Define the curriculum schedule function
    schedule_fn = {
        "curriculum": curriculum_schedule,
        "anti": anti_curriculum_schedule,
        "baseline": baseline_schedule
    }[args.curriculum_type]

    # Make a custom log directory for each hyperparameter combination
    log_dir = os.path.join(
        args.output_dir,
        f"model-{args.model}_schedule-{args.curriculum_type}_chunk-{args.chunk_size}_lr-{args.learning_rate}_batch-{args.batch_size}_buffer-{args.buffer_size}_tau-{args.tau}_gamma-{args.gamma}_ent-{ent_coef_param}_arch-{'-'.join(map(str, net_arch_param))}"
    )
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])
    chunk_size = args.chunk_size

    # 5A) Split the 1200-step dataset into chunk_size episodes (and positions episodes for the lead vehicle)
    episodes_list = chunk_into_episodes(REF_SPEED_DATA, chunk_size)
    positions_list = chunk_positions_into_episodes(LEAD_POSITION_PROFILES["difficult"], chunk_size)
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")

    # 5B) Create the TRAIN environment
    def make_train_env():
        return TrainEnv(
            ref_data=REF_SPEED_DATA,
            lead_positions=LEAD_POSITION_PROFILES,
            schedule_fn=schedule_fn,
            chunk_size=args.chunk_size
        )

    train_env = DummyVecEnv([make_train_env])

    # 5C) Build the model (SAC with MlpPolicy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    policy_kwargs = dict(net_arch=net_arch_param, activation_fn=nn.ReLU)

    """
    Pass in model type from command line
    0. SAC (Default)
    1. PPO
    2. TD3
    3. DDPG
    """

    # Model 1: SAC
    if args.model == 0:
        model = SAC(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            gamma=args.gamma,
            ent_coef=ent_coef_param,
            device=device                   # Do not change
        )
        model_str = "SAC"
    
    # Model 2: PPO
    elif args.model == 1:
        model = PPO(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=ent_coef_param,
            device=device                   # Do not change
        )
        model_str = "PPO"

    # Model 3: TD3
    elif args.model == 2:
        model = TD3(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            gamma=args.gamma,
            device=device                   # Do not change
        )
        model_str = "TD3"

    # Model 4: DDPG
    elif args.model == 3:
        model = DDPG(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            gamma=args.gamma,
            device=device                   # Do not change
        )
        model_str = "DDPG"
    
    # Invalid model number
    else:
        print(f"Invalid model code: {args.model}. Aborting code.")
        exit(1)

    # Print the current hyperparameter configuration
    print("\n\n----- Hyperparameter Configuration -----")
    print(f"Model = {model_str}")
    print(f"Chunk size = {chunk_size}")
    print(f"Learning rate = {args.learning_rate}")
    print(f"Batch size = {args.batch_size}")
    print(f"Buffer size = {args.buffer_size}")
    print(f"Tau = {args.tau}")
    print(f"Gamma = {args.gamma}")
    print(f"Entropy Coefficient = {ent_coef_param}")
    print(f"Net Arch = {args.net_arch}")
    print(f"Schedule (Curriculum) Function = {args.curriculum_type}")
    print("----------------------------------------\n")

    model.set_logger(logger)

    total_timesteps = 100_000
    callback = CustomLoggingCallback(log_dir)

    print(f"[INFO] Start training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        callback=callback
    )
    end_time = time.time()
    print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

    # 5D) Save the model
    save_path = os.path.join(log_dir, f"sac_speed_follow_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    
    # Combine speed profiles for testing
    test_ref = np.concatenate([
        REF_SPEED_DATA for _ in range(3)
    ])
    test_lead = np.concatenate([
        LEAD_SPEED_PROFILES["easy"],
        LEAD_SPEED_PROFILES["medium"],
        LEAD_SPEED_PROFILES["difficult"]
    ])
    test_lead_pos = np.cumsum(test_lead)

    # Run the combined difficulty testing
    test_env = TestEnv(
        full_data=test_ref,
        lead_positions=test_lead_pos,
        delta_t=1.0
    )

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    lead_speeds = []
    distances = []
    jerks = []
    rewards = []
    actions = []

    for _ in range(len(test_ref)):
        action, _states = model.predict(obs, deterministic=True)
        accel = action[0]
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        predicted_speeds.append(obs[0])
        reference_speeds.append(obs[1])
        lead_speeds.append(info["lead_speed"])
        distances.append(info["distance"])
        jerks.append(info["jerk"])
        rewards.append(reward)
        actions.append(accel)
        
        if terminated or truncated:
            break

    def compute_metrics(start, end):
        r = np.array(rewards[start:end])
        ref = np.array(reference_speeds[start:end])
        pred = np.array(predicted_speeds[start:end])
        lead = np.array(lead_speeds[start:end])
        d = np.array(distances[start:end])
        j = np.array(jerks[start:end])

        avg_reward = np.mean(r)
        mean_jerk = np.mean(j)
        var_jerk = np.var(j)
        mae = mean_absolute_error(ref, pred)
        rmse = np.sqrt(mean_squared_error(ref, pred))
        r2 = r2_score(ref, pred)
        safe_zone_pct = np.mean([(x >= MIN_DISTANCE) & (x <= MAX_DISTANCE) for x in d]) * 100
        avg_speed_diff = np.mean(np.abs(pred - lead))

        return {
            "avg_reward": avg_reward,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mean_distance": np.mean(d),
            "safe_zone_pct": safe_zone_pct,
            "min_distance": np.min(d),
            "max_distance": np.max(d),
            "mean_jerk": mean_jerk,
            "max_jerk": np.max(j),
            "var_jerk": var_jerk,
            "avg_speed_diff": avg_speed_diff
        }

    # Indices for sections
    section_len = len(reference_speeds) // 3
    sections = {
        "Easy": compute_metrics(0, section_len),
        "Medium": compute_metrics(section_len, 2 * section_len),
        "Difficult": compute_metrics(2 * section_len, 3 * section_len),
        "Overall": compute_metrics(0, 3 * section_len)
    }

    # Save all results
    with open(os.path.join(log_dir, "test_results.txt"), "w") as f:
        for label, metrics in sections.items():
            f.write(f"=== {label} Section Metrics ===\n")
            f.write(f"Average Reward: {metrics['avg_reward']:.3f}\n\n")
            
            f.write("=== Speed Tracking ===\n")
            f.write(f"MAE: {metrics['mae']:.3f} m/s\n")
            f.write(f"RMSE: {metrics['rmse']:.3f} m/s\n")
            f.write(f"R2 Score: {metrics['r2']:.3f}\n\n")
            
            f.write("=== Distance Maintenance ===\n")
            f.write(f"Mean Distance: {metrics['mean_distance']:.3f} m\n")
            f.write(f"Time in Safe Zone (5-30m): {metrics['safe_zone_pct']:.1f}%\n")
            f.write(f"Min Distance: {metrics['min_distance']:.3f} m\n")
            f.write(f"Max Distance: {metrics['max_distance']:.3f} m\n\n")
            
            f.write("=== Comfort Metrics ===\n")
            f.write(f"Mean Jerk: {metrics['mean_jerk']:.3f} m/s^2\n")
            f.write(f"Max Jerk: {metrics['max_jerk']:.3f} m/s^3\n")
            f.write(f"Jerk Variance: {metrics['var_jerk']:.3f} m^2/s^6\n\n")
            
            f.write("=== Speed Difference with Lead ===\n")
            f.write(f"Mean Absolute Difference: {metrics['avg_speed_diff']:.3f} m/s\n")
            f.write("\n\n")

    # Plot speed tracking performance
    plt.figure(figsize=(12, 6))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Ego Vehicle Speed", linestyle="-")
    plt.plot(lead_speeds, label="Lead Vehicle Speed", linestyle=":", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Tracking Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, "speed_comparison.png"))
    plt.close()

    # Plot distance from lead vehicle
    plt.figure(figsize=(12, 6))
    plt.plot(distances, label="Following Distance", color="blue")
    plt.axhline(y=MIN_DISTANCE, color='r', linestyle='--', label=f"Min Safe Distance ({MIN_DISTANCE}m)")
    plt.axhline(y=MAX_DISTANCE, color='orange', linestyle='--', label=f"Max Desired Distance ({MAX_DISTANCE}m)")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.title("Following Distance to Lead Vehicle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, "following_distance.png"))
    plt.close()

    # Plot jerk over time
    mean_jerk = sections["Overall"]["mean_jerk"]
    plt.figure(figsize=(12, 6))
    plt.plot(jerks, label="Jerk", color="purple")
    plt.axhline(y=mean_jerk, color='black', linestyle='--', label=f"Mean = {mean_jerk:.3f} m/s³")
    plt.xlabel("Timestep")
    plt.ylabel("Jerk (m/s³)")
    plt.title("Ride Comfort: Jerk Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, "jerk_analysis.png"))
    plt.close()

    # Plot the speed difference between lead and ego vehicles
    avg_speed_diff = sections["Overall"]["avg_speed_diff"]
    plt.figure(figsize=(12, 6))
    speed_diff = np.array(predicted_speeds) - np.array(lead_speeds)
    plt.plot(speed_diff, label="Speed Difference (Ego - Lead)", color="green")
    plt.axhline(y=avg_speed_diff, color='black', linestyle='--', label=f"Mean = {avg_speed_diff:.3f} m/s")
    plt.xlabel("Timestep")
    plt.ylabel("Speed Difference (m/s)")
    plt.title("Speed Difference: Ego vs Lead Vehicle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, "speed_difference.png"))
    plt.close()

    # Plot the reward over time
    reward_log_file = os.path.join(log_dir, "training_log.csv")
    if os.path.exists(reward_log_file):
        reward_data = pd.read_csv(reward_log_file)
        plt.figure(figsize=(12, 6))
        plt.plot(reward_data["timestep"], reward_data["average_reward"], label="Average Episode Reward", color="darkorange")
        plt.xlabel("Timestep")
        plt.ylabel("Average Reward")
        plt.title("Reward Penalty Over Time During Training")
        plt.grid(True, alpha=0.3)
        plt.ylim(-100, 10)
        plt.legend()
        plt.savefig(os.path.join(log_dir, "reward_penalty_over_time.png"))
        plt.close()
    else:
        print(f"[WARNING] Reward log file not found: {reward_log_file}")

    # Plot all difficulty profiles against the reference
    plt.figure(figsize=(12, 6))
    plt.plot(REF_SPEED_DATA, label="Reference Speed", linestyle="--", linewidth=2)
    for level, speeds in LEAD_SPEED_PROFILES.items():
        plt.plot(speeds, label=f"Lead - {level.capitalize()}", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title("Reference and Lead Vehicle Speed Profiles (All Difficulties)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "lead_profiles_comparison.png"))
    plt.close()


if __name__ == "__main__":
    main()
