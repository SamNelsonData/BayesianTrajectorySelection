# take in a bunch of features, output probabillities for 4 actions
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    for Actor-Critic RL
        0= Up
        1= Left  
        2= Down
        3= Right
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_actions=4):
        """
        Initializes the policy network. (thanks ai for the comments lol)
        
        @param input_dim   : number of input features describing the state
        @param hidden_dim  : number of neurons in hidden layers
        @param num_actions : number of possible actions
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x, logits=False):
        # forward pass to get action probabilities
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype = torch.float32)
        
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        
        return logits if logits else probs
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy given a state
        """
        probs = self.forward(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_action(self, state, action):
        """
        Evaluate the log probability and entropy for a given state-action pair.
        Useful for PPO-style policy gradient updates.
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy
    
    def train(self, state_feature_list, reward_func, lr=0.1, beta = 1.0, deterministic=True):
        assert False # Figure out how to handle different feature info for policy and reward

        self.train()
        opt = torch.optim.Adam(self.parameters, lr=lr)

        logits = self.forward(state_feature_list, logits=True)

        rewards = torch.tensor([reward_func(x) for x in state_feature_list])

        with torch.no_grad():
            reward_dist = F.softmax(beta * rewards, dim=-1)
        
        policy_dist = F.log_softmax(logits, dim=-1)

        loss = -(reward_dist * policy_dist).sum(dim=-1).mean()

        loss.backwards()

        opt.step()




# ez test
if __name__ == "__main__":
    # 10 input features
    policy = PolicyNetwork(input_dim=10, hidden_dim=32, num_actions=4)
    
    # Test with a random state
    test_state = torch.randn(1, 10)
    probs = policy(test_state)
    
    print(f"Input state shape: {test_state.shape}")
    print(f"Output probabilities: {probs}")
    
    # Test action sampling
    action, log_prob = policy.get_action(test_state)
    print(f"Sampled action: {action.item()}")
    print(f"Log probability: {log_prob.item():.4f}")