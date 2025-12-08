'''
General Agent class. Provides functionality for:
  - Determining possible actions via class field transition_prob, a matrix of possible actions transition probabilities (boolean in this case)
  - Making Random Actions
  - generating trajectories
  - traversing trajectories
'''
class Agent:
    
    def __init__(
            self,
            position,
            states
    ):
        self.trajectory = None
        self.position = position
        self.states = states
        self.battery = 2
        self.transition_prob = zeros(tuple(array(states.shape)**2)).astype(bool) # make matrix with (n*n)x(n*n) dim for nxn world

        for position in range(states.shape[0] * states.shape[1]):
            available_spaces = self.__adjacent_spaces(self.__index_swap(position))

            for available_space in available_spaces:
                if self.states[available_space[0], available_space[1]] != 2: # check there isn't a rock there
                    self.transition_prob[position, self.__index_swap(available_space)] = True

    def gen_trajectory(self, seeded = None, policy = None, steps = 5):
        
        original_pos = deepcopy(self.position)

        if policy is None:
            policy = lambda x: self.random_action()
        
        if seeded is not None: 
            seed(seeded)
            seeds = uniform(0, 100000, steps).astype(int)

        trajectory = []
        for step in range(steps):
            if seeded is not None:
                seed(seeds[step])
            trajectory += [policy(self)]

        self.position = original_pos
        
        self.trajectory = trajectory

        return trajectory
    
    '''
    Changes the agent position to the next location in the trajectory and returns the new location
    '''
    def traverse_trajectory(self):
        if self.trajectory is None:
            return None
        self.position = self.trajectory.pop(0)
        return self.position

    def del_trajectory(self):
        self.trajectory = None

    def random_action(self, seeded=None):
        available_actions = where(self.transition_prob[self.__index_swap(self.position)])[0]

        if seeded is not None:
            seed(seeded)
        action = self.__index_swap(choice(available_actions))

        self.position = action
        return action

    '''
    Switches from (row,column) indexing to single value indexing
    '''
    def __index_swap(self, index):
        dim = len(self.states)
        if hasattr(index, '__iter__'):
            return dim*index[0] + index[1]
        return (int(index / dim), int(index % dim))
    
    def __adjacent_spaces(self, position):
        spaces = []
        dim = len(self.states)

        # above
        if position[0] > 0:
            spaces += [[position[0] - 1, position[1]]]
        # left
        if position[1] > 0:
            spaces += [[position[0], position[1] - 1]]
        # down
        if position[0] < dim - 1:
            spaces += [[position[0] + 1, position[1]]]
        # right
        if position[1] < dim - 1:
            spaces += [[position[0], position[1] + 1]]
        
        return spaces

    def __direction(self, index1, index2):
        if index1[0] > index2[0]:
            return 0
        if index1[0] < index2[0]:
            return 2
        if index1[1] > index2[1]:
            return 1
        if index1[1] < index2[1]:
            return 3
        raise Exception("Indices occupies same position")
    
    def __str__(self):
        if self.trajectory is None:
            raise Exception("Trajectory not Generated")
        
        direction = "↑←↓→"

        traj = []

        old_pos = self.position
        for pos in self.trajectory:
            traj += [direction[self.__direction(old_pos, pos)]]

            old_pos = pos
        return "".join(traj)
            
# Incomplete Actor-Critic Agent, inherits Agent class
class AgentA2C(Agent):
    '''
    The Agent class currently exists as a place holder. Eventually 
    it will include functionality like determining possible 
    actions and which actions may be optimal.
    '''
    
    def __init__(
            self,
            position,
            states
    ):
        super().__init__(position, states)
    
    def act(self):
        self.random_action()
        return self.position
