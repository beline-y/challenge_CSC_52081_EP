#!/usr/bin/env python3
"""
Simple Example Script for Student Gym Environment

This script demonstrates the simplest way to use the student gym environment
with proper action mapping for multi-step execution and plotting.
"""

from student_client import create_student_gym_env, plot_observations

def main():
    """Main function - demonstrating proper action mapping"""
    print("🎓 Student Gym Environment - Simple Example")
    print("=" * 50)
    print("Demonstrating proper action mapping with multi-step execution\n")
    
    try:
        step_size = 10 # step size between observations, recommended 10
        env = create_student_gym_env(step_size=step_size)

        # Reset environment to get initial observation
        obs, info = env.reset()
        print(f"📋 Starting episode {info.get('episode_id', 'unknown')}")


        observations = []
        actions = []
        rewards = []
        total_timesteps = 0

        for step in range(50):
            # time.sleep(2)

            # Choose a random action (0=do nothing, 1=repair, 2=sell)
            action = env.action_space.sample()
            action = 0

            if step % 4 == 0 and step > 0:
                action = 1
            if step >= 30:
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

            # Update total timesteps - server advances by step_size but returns only final observation
            total_timesteps += step_size

            # Print progress every step
            if step % 1 == 0:
                print(f" Step {total_timesteps}: Reward={reward:.2f}, Total={sum(rewards):.2f}")

            # Check if episode ended
            if terminated or truncated:
                print(f"🏁 Episode ended at step {total_timesteps} with reward={reward:.2f}")
                break


        # Print summary statistics
        total_reward = sum(rewards)
        print(f"\n Episode Summary:")
        print(f"   Total Steps: {total_timesteps}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Actions Taken: {len([a for a in actions if a == 1])} repairs, {len([a for a in actions if a == 2])} sell")

        # Finish episode
        env.close()
        
        # Generate plot
        print(f"\n📊 Generating plot...")
        plot_observations(
            observations=observations,
            actions=actions,
            title="Observations and Actions Over Time"
        )
        
        print(f"\n✅ Plot generated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the gym server is running and accessible")

if __name__ == "__main__":
    main()