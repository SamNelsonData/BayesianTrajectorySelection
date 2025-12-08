
import torch
import torch.nn as nn

class Policy(nn.Module):

	def __init__(self, features, device = 'cpu'):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(features, 4), nn.Softmax()
		)

		self.net.to(device)
	
	def __call__(self, *args, **kwargs):
		return self.net.forward(args[0])
	
class Reward(nn.Module):
	
    def __init__(self, features, actions, device = 'cpu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features + actions, 1)
        )

        self.net.to(device)

    def __call__(self, *args, **kwargs):
        assert len(args) == 2
        return self.net.forward(args[0], args[1])
    

    def preference_update(self, segments, preferences, lr = 1e-2):
        assert isinstance(segments, list)
        assert isinstance(segments[0], tuple)

        self.train()
        # each segment is a state-action pair
        BCELoss = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for s1, s2, mu in zip(segments, preferences):
            
            r1 = torch.sum([self(s, a) for s, a in s1])
            r2 = torch.sum([self(s, a) for s, a in s2])

            loss = BCELoss(r1 - r2, mu)
            loss.backward()
            opt.step()
        
        self.eval()




