"""
Simple Plotting Functions for Student Gym Environment

Basic plotting functions for visualizing observations and rewards.
Students can use these as inspiration to create their own visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_observations(
    observations: List[np.ndarray],
    actions: Optional[List[int]] = None,
    sensor_names: Optional[List[str]] = None,
    figsize: tuple = (12, 4),
    title: str = "Observation Dimensions Over Time"
) -> None:
    """
    Plot observation dimensions over time, handling batched observations.

    This function accepts a list where each element is an array of observations
    returned by a single `step()` call (e.g., shape (10, 9)). It concatenates
    all observations into a continuous sequence and, if actions are provided,
    marks the **first** observation of each batch with the corresponding action.
    Only Repair (1) and Sell (2) actions are shown; Do Nothing (0) is hidden.

    Args:
        observations: List of observation batches. Each batch is a 2D numpy array
                      of shape (n_steps_in_batch, 9). The final batch may be shorter.
        actions: Optional list of actions, one per batch. Must have the same length
                 as `observations`. The action is associated with the first step
                 of its batch.
        sensor_names: Optional list of 9 names for the observation dimensions.
        figsize: Figure size (width, height) for each individual dimension plot.
        title: Title for each plot (the dimension name is appended).

    Example:
        >>> from student_client import create_student_gym_env, plot_observations
        >>> env = create_student_gym_env()
        >>> obs, info = env.reset()
        >>> observations = [obs]          # obs shape (10,9)
        >>> actions = []
        >>> for step in range(50):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action, step_size=10)
        ...     observations.append(obs)   # another batch of observations
        ...     actions.append(action)
        ...     if terminated or truncated:
        ...         break
        >>> env.close()
        >>> plot_observations(observations, actions)
    """
    if not observations:
        print("⚠️ No observations provided.")
        return

    # Flatten all observation batches into one long array
    obs_arrays = []
    batch_starts = []      # starting index of each batch in the concatenated array
    current_idx = 0
    for obs in observations:
        # Ensure each batch is at least 2D
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs_arrays.append(obs)
        batch_starts.append(current_idx)
        current_idx += obs.shape[0]

    obs_full = np.concatenate(obs_arrays, axis=0)   # shape (total_steps, 9)
    total_steps = obs_full.shape[0]
    steps = np.arange(total_steps)

    # Handle actions: create an array aligned with obs_full
    action_full = None
    if actions is not None:
        # Ensure actions list matches number of batches
        if len(actions) != len(observations):
            print(f"⚠️ Warning: number of actions ({len(actions)}) != number of batches ({len(observations)}). "
                  f"Truncating to the shorter length.")
            min_len = min(len(actions), len(observations))
            actions = actions[:min_len]

            # Re‑compute batch_starts and obs_full using only the first min_len batches
            obs_arrays = []
            batch_starts = []
            current_idx = 0
            for i in range(min_len):
                obs = observations[i]
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                obs_arrays.append(obs)
                batch_starts.append(current_idx)
                current_idx += obs.shape[0]
            obs_full = np.concatenate(obs_arrays, axis=0)
            total_steps = obs_full.shape[0]
            steps = np.arange(total_steps)

        # Create action array: -1 indicates no action (Do Nothing is not plotted)
        action_full = np.full(total_steps, -1, dtype=int)
        for i, start_idx in enumerate(batch_starts):
            action_full[start_idx] = actions[i]

    # Default sensor names if not provided
    if sensor_names is None:
        sensor_names = [
            'HPC_Tout',      # High Pressure Compressor Temperature Outlet
            'HP_Nmech',      # High Pressure Shaft Mechanical Speed
            'HPC_Tin',       # High Pressure Compressor Temperature Inlet
            'LPT_Tin',       # Low Pressure Turbine Temperature Inlet
            'Fuel_flow',     # Fuel Flow Rate
            'HPC_Pout_st',   # High Pressure Compressor Pressure Outlet (static)
            'LP_Nmech',      # Low Pressure Shaft Mechanical Speed
            'phase_type',    # Flight Phase Type
            'DTAMB'
        ]

    num_dims = obs_full.shape[1]

    # Plot each dimension separately
    for i in range(num_dims):
        plt.figure(figsize=figsize)
        plt.plot(steps, obs_full[:, i], 'b-', linewidth=2, label='Observation')

        # Use sensor name if available, else fallback to index
        name = sensor_names[i] if i < len(sensor_names) else f"Dimension {i}"
        plt.title(f"{title} – {name}", fontsize=14, fontweight='bold')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add action markers only for Repair (1) and Sell (2)
        if action_full is not None:
            # Find indices where action is 1 or 2
            repair_idx = np.where(action_full == 1)[0]
            sell_idx   = np.where(action_full == 2)[0]

            if len(repair_idx) > 0:
                plt.scatter(repair_idx, obs_full[repair_idx, i],
                            color='red', marker='o', s=80,
                            label='Repair (1)', alpha=0.8, zorder=5)
            if len(sell_idx) > 0:
                plt.scatter(sell_idx, obs_full[sell_idx, i],
                            color='green', marker='s', s=80,
                            label='Sell (2)', alpha=0.8, zorder=5)

        # Build legend (unique labels only)
        if action_full is not None and (1 in action_full or 2 in action_full):
            handles = []
            if 1 in action_full:
                handles.append(plt.scatter([], [], color='red', marker='o', s=80, label='Repair (1)', alpha=0.8))
            if 2 in action_full:
                handles.append(plt.scatter([], [], color='green', marker='s', s=80, label='Sell (2)', alpha=0.8))
            plt.legend(handles=handles, loc='best', fontsize=10)

        plt.tight_layout()
        plt.show()


def plot_rewards(
    rewards: List[float],
    actions: Optional[List[int]] = None,
    figsize: tuple = (12, 6),
    title: str = "Step Rewards Over Time"
) -> None:
    """
    Simple function to plot step rewards over time.
    
    This function creates a clean visualization of individual step rewards,
    with cumulative reward shown in the legend.
    
    Args:
        rewards: List of reward values obtained at each step
        actions: Optional list of actions taken at each step
        figsize: Figure size (width, height)
        title: Plot title
        
    Example:
        >>> from student_client import create_student_gym_env, plot_rewards
        >>> 
        >>> # Collect rewards
        >>> env = create_student_gym_env()
        >>> obs, info = env.reset()
        >>> rewards = []
        >>> actions = []
        >>> 
        >>> for step in range(50):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     rewards.append(reward)
        ...     actions.append(action)
        ...     if terminated or truncated:
        ...         break
        >>> 
        >>> env.close()
        >>> 
        >>> # Plot step rewards
        >>> plot_rewards(rewards, actions)
    """
    if not rewards:
        print("⚠️ No rewards provided.")
        return
    
    steps = np.arange(len(rewards))
    cumulative_reward = np.sum(rewards)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot individual rewards as a line with markers
    plt.plot(steps, rewards, 'b-', linewidth=2, 
            marker='o', markersize=8, label=f'Step Reward')
    
    # Add action markers if provided (only Repair and Sell)
    if actions is not None and len(actions) == len(rewards):
        for step, action in enumerate(actions):
            if action == 1:  # Repair
                plt.scatter(step, rewards[step], color='red', 
                          marker='o', s=120, label='Repair',
                          edgecolor='black', linewidth=1, alpha=0.8, zorder=5)
            elif action == 2:  # Sell
                plt.scatter(step, rewards[step], color='green', 
                          marker='s', s=120, label='Sell',
                          edgecolor='black', linewidth=1, alpha=0.8, zorder=5)
    
    # Customize plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend with cumulative reward
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Remove duplicate labels and add cumulative reward info
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Add cumulative reward to legend
    cumulative_handle = plt.Line2D([], [], color='black', linestyle='none', 
                                   marker='', markersize=0, label=f'Cumulative: {cumulative_reward:.1f}')
    unique_handles.append(cumulative_handle)
    unique_labels.append(f'Cumulative: {cumulative_reward:.1f}')
    
    if unique_handles:
        plt.legend(handles=unique_handles, labels=unique_labels, 
                  loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"📊 Reward Statistics:")
    print(f"   Total Steps: {len(rewards)}")
    print(f"   Total Reward: {cumulative_reward:.2f}")
    print(f"   Average Reward: {np.mean(rewards):.2f}")
    print(f"   Max Reward: {np.max(rewards):.2f}")
    print(f"   Min Reward: {np.min(rewards):.2f}")