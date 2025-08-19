import numpy as np
from ReinforcementLearningLoop import ReinforcementLearningLoop

if __name__ == "__main__":
    print("Starting PPO...")
    
    config = {
        "num_steps_per_epoch": 2048,
        "num_training_iterations": 2000,
        "k_epochs": 10
    }
    
    # Use rgb_array for training (faster, no visual window)
    rl_loop = ReinforcementLearningLoop(render_mode="rgb_array")
    
    # Train the model
    rl_loop.run(config)
    
    # Sample from trained policy with human rendering
    print("\n" + "="*50)
    print("EVALUATING TRAINED POLICY")
    print("="*50)
    
    # Sample 5 episodes with visual rendering
    episode_rewards = rl_loop.sample_trained_policy(num_episodes=5, render_mode="human")
    
    print(f"\nFinal evaluation complete! Model achieved average reward: {np.mean(episode_rewards):.3f}")
