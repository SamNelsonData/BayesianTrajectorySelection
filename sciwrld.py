
from numpy.random import uniform, choice, binomial
from numpy import where, zeros, array

import torch
import torch.nn as nn

class SciWrld:
    '''
    The SciWrld class manages map creation, cloud management, 
        and available actions to the agent. The assumed goal 
        of the agent is to collect seeds placed randomly 
        around the map. The agent should generally avoid 
        clouds as the agent is solar-powered and will lose 
        battery.

    TODO
    - I don't think it should handle available actions to the 
        agent. That functionality should probably be moved to 
        the Agent class
    '''
    
    # --- Graphical Representation ---
    item_to_value = {
        'Sand': 0,
        'Rock': 2,
        'Seed': 3,
        'Agent': 4,
        'Biofuel': 5,
        'Cloud': 6
    }
    
    symbols = ['.', '\u25b2', '6', 'X', '*', 'm']
    value_to_item = {value: key for key, value in item_to_value.items()}
    value_to_symbol = dict(zip(item_to_value.values(), symbols))
    
    # --- Constructor ---
    def __init__(
            self,
            size = (10, 10),
            starting_seeds = 3,
            rocks = 10):
        
        # --- basic properties ---
        self.size = size
        self.world = zeros(size).astype(int)
        self.__refresh_empty()
        self.world_time = 0

        # --- Build Map ---

        # -- Agent Spawn
        val = self.item_to_value['Agent']
        self.random_quota(1, val)
        self.agent = Agent(
            array(where(self.world == val)).reshape(-1),
            self.size
        )

        # -- Rocks
        self.random_quota(rocks, self.item_to_value['Rock'])

        # -- Seeds
        self.random_quota(starting_seeds, self.item_to_value['Seed'])

        # --- Cloud Initialization ---
        self.clouds = []
    
    '''
    For each empty tile, places the specified object with probability 'p'
    '''
    def random_placement(self, p, id):
        count = len(self.empty[0])
        values = binomial(1, p, size = count) * id
        self.world[self.empty] = values
        self.__refresh_empty()
    
    '''
    Places the specified object "quota" times on random empty tiles
    '''
    def random_quota(self, quota, id):
        assert quota <= len(self.empty[0])
        indices = choice(len(self.empty[0]), quota, replace=False)
        indices = (self.empty[0][indices], self.empty[1][indices])
        self.world[indices] = id
        self.__refresh_empty()

    '''
    Refreshes class tuple of empty tile indices (i.e. updates a 
    container of indices marking empty tiles)
    '''
    def __refresh_empty(self):
        self.empty = where(self.world == 0)

    '''
    Cycles the world state by some number of steps
    @steps: the number of steps to simulate
    @new_cloud_rate: the probability of a new cloud being 
        generated assuming there is room
    @cloud_limit: the number of clouds that can exist at 
        one time
    @cloud_bounds: the maximum area over which a cloud 
        can be spread
    @cloud_direction: a function for determining 
        direction. Unimplemented
    '''
    def step(
            self,
            steps = 1,
            new_cloud_rate = 0.2,
            cloud_limit = 3,
            cloud_size = 6,
            cloud_bounds = (4,4),
            cloud_direction = lambda x: x,
    ):
        for i in range(steps):
            # --- Clouds ---
            spawn_cloud = bool(binomial(1, new_cloud_rate))
            if (cloud_limit > len(self.clouds)) and spawn_cloud:
                row = binomial(self.size[0] - cloud_bounds[0], 0.5)
                col = binomial(self.size[1] - cloud_bounds[1], 0.5)
                edge = int(uniform(0, 4))
                match edge:
                    case 0:
                        location = (0 - cloud_bounds[0], col)
                        cloud_direction = 2
                    case 1:
                        location = (row, 0 - cloud_bounds[1])
                        cloud_direction = 3
                    case 2:
                        location = (self.size[0] - 1 + cloud_bounds[0], col)
                        cloud_direction = 0
                    case 3:
                        location = (row, self.size[1] - 1 + cloud_bounds[1])
                        cloud_direction = 1
                self.clouds += [(Cloud(
                    size=cloud_size,
                    bounds=cloud_bounds,
                    condense=location
                ), cloud_direction)]
            
            # -- Move the clouds
            for cloud, direction in self.clouds[:]:
                cloud.move(direction)

                # -- Track Offscreen Clouds
                onscreen = False
                for row, col in cloud:
                    if (row, col) in self:
                        onscreen = True
                if not onscreen:
                    self.clouds.remove((cloud, direction))
            
            # --- Agent Update ---
            arow, acol = tuple(self.agent.position)

            # -- Get Available Actions
            actions = []
            rock_val = self.item_to_value['Rock']
            # - Test Up
            if arow > 0 and self.world[arow - 1, acol] != rock_val:
                actions += ['Up']
            # - Test Left
            if acol > 0 and self.world[arow, acol - 1] != rock_val:
                actions += ['Left']
            # - Test Down
            if arow < self.size[0] - 1 and self.world[arow + 1, acol] != rock_val:
                actions += ['Down']
            # - Test Right
            if acol < self.size[1] - 1 and self.world[arow, acol + 1] != rock_val:
                actions += ['Right']

            self.agent.action(choice(actions, 1)[0])

            # --- Update Map ---
            self.world[arow, acol] = 0
            self.world[tuple(self.agent.position)] = self.item_to_value['Agent']

    # --- Overloaded Operators ---
    def __str__(self):
        answer = ""
        row_ind = 0
        for row in self.world:
            col_ind = 0
            for col in row:
                item = self.value_to_symbol[col]
                for cloud, _ in self.clouds:
                    if (row_ind, col_ind) in cloud:
                        item = "m" # TODO cloud symbol currently hardcoded
                answer += f'{item:<2}'
                col_ind += 1
            answer += '\n'
            row_ind += 1
        return answer
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return self.world[i]
    
    def __setitem__(self, i, v):
        self.world[i] = v
    
    def __contains__(self, v):
        row = v[0]
        col= v[1]
        if (row < 0) or (col < 0):
            return False
        if (row >= self.size[0]) or (col >= self.size[1]):
            return False
        return True

class Cloud:
    '''
    The Cloud class manages instances of clouds on the SciWrld map. 
    This includes interactions like determining if a coordinate is 
    currently shaded by a cloud or moving the cloud.
    '''
    def __init__(self, size=4, bounds = (4,4), condense=None):
        assert (bounds[0] * bounds[1]) > size
        cloud_mask = zeros(bounds).reshape(-1).astype(int)
        indices = choice(len(cloud_mask), size, replace=False)
        cloud_mask[indices] = 1
        self.crow, self.ccol = where(cloud_mask.reshape(bounds) == 1)
        self.size = bounds

        if condense is not None:
            self.condense(condense)
    
    '''
    This method places the cloud at a location
    '''
    def condense(self, location):
        self.row = location[0]
        self.col = location[1]

        assert isinstance(self.row, int) and isinstance(self.col, int)

        self.crow += self.row
        self.ccol += self.col

    '''
    This method moves the cloud in a particular direction
    '''
    def move(self, direction, speed=1):
        match direction:
            case 0:
                self.crow -= speed
            case 1:
                self.ccol -= speed
            case 2:
                self.crow += speed
            case 3:
                self.ccol += speed
            case default:
                raise Exception("Improper directional input in Class: Cloud")

    def __contains__(self, v):
        for i,j in self:
            if v[0] == i and v[1] == j:
                return True
        return False
    
    def __iter__(self):
        self._current = 0
        return self
    
    def __next__(self):
        if self._current < len(self.crow):
            curr = self._current
            self._current += 1
            return (self.crow[curr], self.ccol[curr])
        else:
            raise StopIteration
        
    def __str__(self):
        mask = zeros(self.size).astype(int)
        mask[(self.crow - self.row, self.ccol - self.col)] = 1
        answer = ''
        for row in mask:
            for col in row:
                answer += f'{col:<2}'
            answer += '\n'
        return answer


class Agent:
    '''
    The Agent class currently exists as a place holder. Eventually 
    it will include functionality like determining possible 
    actions and which actions may be optimal.
    '''
    
    def __init__(
            self,
            position,
            limits,
            battery=2
    ):
        self.position = position
        self.limits = limits
        self.battery = battery

    def action(self, action, speed=1):
        match action:
            case 'Up':
                if self.position[0] > 0:
                    self.position[0] -= speed
            case 'Left':
                if self.position[1] > 0:
                    self.position[1] -= speed
            case 'Down':
                if self.position[0] < self.limits[0] - 1:
                    self.position[0] += speed
            case 'Right':
                if self.position[1] < self.limits[1] - 1:
                    self.position[1] += speed
            case 'Sample':
                raise Exception('Sample is not yet implemented')

    def reduce_battery(self, reduction = 1):
        self.battery -= reduction
    
    def increase_battery(self, increase = 1):
        self.battery += increase

    def __call__(self, action):
        self.action(action)

    def __bool__(self):
        return self.battery > 0
    
    '''
    This method isn't really implemented. Important features 
    (other than the world tiles) include the agent's position 
    and other things.
    '''
    def get_features(self, world: SciWrld):
        return world.world

class RewardNet(nn.Module):
        
        '''
        This method is currently unimplemented as a reward function
        '''

        def __init__(self, model: nn.Sequential, device='mps'):
            super().__init__()

            self.net = model
            self.device = device

            self.to(self.device)
    
        def set_action_index(self, index):
            self.action_index = index

        def action_max(self, X, lr=5e-2, epochs=80):
            if not torch.is_tensor(X):
                X = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
            else:
                X = X.clone().detach()
                X.requires_grad_(True)
                X.to(self.device)

            for i in range(epochs):
                y = self(X)
                y.backward()
                gradient = X.grad[self.action_index]
                with torch.no_grad():
                    addition = torch.zeros(X.size()).requires_grad(False)
                    addition[self.action_index] = gradient
                    X += lr * addition
                self.net.zero_grad()
            return X[self.action_index]
                
        def __call__(self, X):
            if not torch.is_tensor(X):
                X = torch.tensor(X).to(self.device)
            return self.net(X).flatten()





