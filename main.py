import torch
import torch.optim as optim
import torch.nn as nn
from student_client import create_student_gym_env # Import spécifique au challenge
from models import QNetwork
from train import EpsilonGreedy, train_one_episode

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 500
GAMMA = 0.99
LR = 0.001

def main():
    # 1. Créer l'environnement (assurez-vous d'avoir votre .env configuré)
    env = create_student_gym_env()
    n_obs = env.observation_space.shape[0] * env.observation_space.shape[1] # [10, 9] -> 90
    n_actions = env.action_space.n
    
    # 2. Initialiser les modèles
    q_net = QNetwork(n_obs, n_actions).to(DEVICE)
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    eg_policy = EpsilonGreedy(1.0, 0.05, 0.995, n_actions)
    
    # 3. Boucle d'entraînement
    print("Début de l'entraînement...")
    for ep in range(NUM_EPISODES):
        # Note: On doit aplatir l'observation car elle est de forme [10, 9]
        reward = train_one_episode(env, q_net, optimizer, loss_fn, eg_policy, GAMMA, DEVICE)
        
        if ep % 10 == 0:
            print(f"Episode {ep} | Reward: {reward:.2f} | Epsilon: {eg_policy.epsilon:.2f}")

    # 4. Sauvegarder le modèle
    torch.save(q_net.state_dict(), "turbofan_dqn.pth")
    print("Modèle sauvegardé !")

if __name__ == "__main__":
    main()