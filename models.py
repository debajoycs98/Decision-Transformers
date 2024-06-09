import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class QNetworkFC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim+action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        #self.init_weights()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.xavier_normal_(m.weight, nn.init.calculate_gain("relu"))
                #nn.init.normal_(m.weight, 0, 1e-4)
                nn.init.normal_(m.weight, 0, 1)
                nn.init.zeros_(m.bias)

    def do_polyak(self, other, factor):
        with torch.no_grad():
            for param, target_param in zip(self.parameters(), other.parameters()):
                param.data.mul_(factor)
                param.data.add_((1-factor) * target_param.data)


class PolicyNetworkFC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3_act = nn.Linear(hidden, action_dim)
        self.fc3_sig = nn.Linear(hidden, action_dim)

        self.variance_eps = 1e-3

        #self.init_weights()

    def forward(self, state):
        # Get the action
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x_act = self.fc3_act(x)

        # Get the std of the output distribution
        x_sig = self.fc3_sig(x)

        # Ensure sigma > 0 and give some minimum value
        x_sig = torch.clamp(x_sig, -20, 2)
        x_sig = torch.exp(x_sig)
        #x_sig = F.softplus(x_sig) + self.variance_eps

        dist = Normal(x_act, x_sig)

        return dist

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1e-5)
                nn.init.zeros_(m.bias)

        # Provide some extra initial variance
        #self.fc3_sig.bias.data += .5


