#!/usr/bin/env python3
"""
Stress Test for Student Gym Environment

Tests the system's ability to handle multiple parallel instances
efficiently using the same patterns as simple_example.py.
"""

import concurrent.futures
import time
import random
from student_client import create_student_gym_env
from typing import List, Dict, Any

def run_episode(episode_id: int, max_steps: int = 100) -> Dict[str, Any]:
    """
    Run a single episode using the exact same pattern as simple_example.py
    
    Args:
        episode_id: Identifier for this episode
        max_steps: Maximum steps to run
        
    Returns:
        Dictionary with episode statistics
    """
    try:
        # Create environment with step_size (same as simple_example.py)
        step_size = 10
        env = create_student_gym_env(step_size=step_size, user_token='baseline2')
        
        # Reset environment (same as simple_example.py)
        obs, info = env.reset()
        
        # Initialize data collection (same as simple_example.py)
        observations = []
        actions = []
        rewards = []
        total_timesteps = 0
        
        for step in range(max_steps // step_size):
            # Choose action using same policy as simple_example.py
            # Choose a random action (0=do nothing, 1=repair, 2=sell)
            action = env.action_space.sample()
            action = 0

            if step % 4 == 0 and step > 0:
                action = 1
            if step >= 35:
                action = 2

            # Take step in environment
            # obs, reward, terminated, truncated, info = env.step(action, step_size=10, return_all_states=True)

            obs_result, reward, terminated, truncated, info = env.step(
                action=action, step_size=step_size, return_all_states=True
            )

            # Handle the observation result
            if isinstance(obs_result, list):
                # Multiple observations returned
                observations.extend(obs_result)
                # Action is applied at the first timestep of this interval
                actions.append(action)
                if action != 2:
                    actions.extend([0] * (len(obs_result) - 1))  # fill with no actions
            else:
                actions.append(action)

            rewards.append(reward)
            total_timesteps += step_size

            if terminated or truncated:
                break
        
        # Clean up (same as simple_example.py)
        env.close()
        
        # Calculate statistics
        total_reward = sum(rewards)
        repair_count = len([a for a in actions if a == 1])
        sell_count = len([a for a in actions if a == 2])
        
        return {
            'episode_id': episode_id,
            'status': 'success',
            'steps_taken': len(actions),
            'total_timesteps': total_timesteps,
            'total_reward': total_reward,
            'average_reward': total_reward / len(actions) if len(actions) > 0 else 0,
            'repair_actions': repair_count,
            'sell_actions': sell_count,
            'error': None
        }
        
    except Exception as e:
        return {
            'episode_id': episode_id,
            'status': 'error',
            'steps_taken': 0,
            'total_timesteps': 0,
            'total_reward': 0,
            'average_reward': 0,
            'repair_actions': 0,
            'sell_actions': 0,
            'error': str(e)
        }

def run_stress_test(num_instances: int = 5, max_steps: int = 100) -> None:
    """
    Run stress test with multiple parallel instances.
    
    Args:
        num_instances: Number of parallel instances to run
        max_steps: Maximum steps per episode
    """
    print("🧪 Student Gym Environment - Stress Test")
    print("=" * 45)
    print(f"🔥 Launching {num_instances} parallel instances...")
    print(f"📊 Each instance follows simple_example.py pattern")
    print(f"💡 Testing system stability and performance...\n")
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
        # Submit all episodes
        futures = []
        for i in range(num_instances):
            future = executor.submit(run_episode, i, max_steps)
            futures.append(future)
            print(f"🚀 Launched instance {i + 1}/{num_instances}")
        
        # Wait for all episodes to complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ Instance {result['episode_id'] + 1} completed: "
                      f"{result['steps_taken']} steps, "
                      f"{result['total_timesteps']} timesteps, "
                      f"reward={result['total_reward']:.1f}, "
                      f"repairs={result['repair_actions']}, "
                      f"sells={result['sell_actions']}")
            else:
                print(f"❌ Instance {result['episode_id'] + 1} failed: "
                      f"{result['error']}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"\n📈 Stress Test Results:")
    print(f"   Total Instances: {num_instances}")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Average Time per Instance: {total_time / num_instances:.2f} seconds")
    
    if successful:
        avg_steps = sum(r['steps_taken'] for r in successful) / len(successful)
        avg_timesteps = sum(r['total_timesteps'] for r in successful) / len(successful)
        avg_reward = sum(r['total_reward'] for r in successful) / len(successful)
        total_repairs = sum(r['repair_actions'] for r in successful)
        total_sells = sum(r['sell_actions'] for r in successful)
        
        print(f"   Average Steps: {avg_steps:.1f}")
        print(f"   Average Timesteps: {avg_timesteps:.1f}")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Total Repairs: {total_repairs}")
        print(f"   Total Sells: {total_sells}")
    
    if failed:
        print(f"\n⚠️  Failed Instances:")
        for result in failed:
            print(f"   Instance {result['episode_id'] + 1}: {result['error']}")
    
    # Performance assessment
    if len(failed) == 0:
        print(f"\n🎉 Stress test PASSED!")
        print(f"   ✅ All {num_instances} instances completed successfully")
        print(f"   ✅ System handled parallel load well")
        print(f"   ✅ No errors or timeouts occurred")
        print(f"   📊 Average performance: {avg_timesteps:.0f} timesteps, {avg_reward:.1f} reward")
    else:
        print(f"\n⚠️  Stress test completed with {len(failed)} failures")
        print(f"   💡 Check server capacity and error messages above")

def main():
    """Main function for stress testing"""
    try:
        # Get user input for stress test parameters
        print("🎓 Student Gym Environment - Stress Test Setup")
        print("=" * 50)
        
        # Use default values if no input is provided (for non-interactive use)
        try:
            num_instances = int(input("Enter number of parallel instances (e.g., 5): ") or "5")
            max_steps = int(input("Enter max steps per episode (e.g., 100): ") or "100")
        except EOFError:
            # Handle case where input is not available (non-interactive mode)
            num_instances = 5
            max_steps = 100
            print("Running with default values (5 instances, 100 steps)")
        
        print(f"\n🔥 Starting stress test with {num_instances} instances...")
        
        # Run stress test
        run_stress_test(num_instances, max_steps)
        
    except KeyboardInterrupt:
        print("\n🛑 Stress test interrupted by user")
    except Exception as e:
        print(f"❌ Error during stress test: {e}")
        print("💡 Make sure the gym server is running and can handle multiple connections")

if __name__ == "__main__":
    main()