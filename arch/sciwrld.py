"""
SciWrld Environment for IRD-RLHF experiments.

A solar-powered rover navigates a gridworld with:
- Seeds to collect (reward)
- Clouds that block solar power (penalty)
- Rocks as obstacles
"""

import numpy as np
from numpy.random import choice, binomial, seed as np_seed
from copy import deepcopy


class Cloud:
    """
    Manages cloud instances that move across the grid.
    Clouds block solar power, draining the rover's battery.
    """
    
    def __init__(self, size=4, bounds=(4, 4), location=None):

        assert bounds[0] * bounds[1] >= size, "Cloud size exceeds bounds"
        
        # Generate random cloud shape within bounds
        cloud_mask = np.zeros(bounds[0] * bounds[1], dtype=int)
        indices = choice(len(cloud_mask), size, replace=False)
        cloud_mask[indices] = 1
        cloud_mask = cloud_mask.reshape(bounds)
        
        self.local_rows, self.local_cols = np.where(cloud_mask == 1)
        self.bounds = bounds
        
        # Global position (top-left corner of bounding box)
        self.row = 0
        self.col = 0
        
        if location is not None:
            self.set_position(location)
    
    def set_position(self, location):
        """Place the cloud at a specific location."""
        self.row, self.col = int(location[0]), int(location[1])
    
    def move(self, direction, speed=1):
        """
        Move cloud in given direction.
        Directions: 0=Up, 1=Left, 2=Down, 3=Right
        """
        if direction == 0:
            self.row -= speed
        elif direction == 1:
            self.col -= speed
        elif direction == 2:
            self.row += speed
        elif direction == 3:
            self.col += speed
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def get_cells(self):
        """Return list of (row, col) tuples covered by this cloud."""
        return [
            (self.row + lr, self.col + lc)
            for lr, lc in zip(self.local_rows, self.local_cols)
        ]
    
    def __contains__(self, pos):
        """Check if position (row, col) is under this cloud."""
        r, c = pos
        for lr, lc in zip(self.local_rows, self.local_cols):
            if r == self.row + lr and c == self.col + lc:
                return True
        return False
    
    def __iter__(self):
        """Iterate over cells covered by cloud."""
        return iter(self.get_cells())
    
    def __repr__(self):
        return f"Cloud(pos=({self.row},{self.col}), cells={len(self.local_rows)})"


class SciWrld:
    """
    SciWrld gridworld environment.
    
    The rover collects seeds while avoiding clouds.
    Spending time under clouds drains battery.
    """
    
    # Cell types - using contiguous values for clean indexing
    EMPTY = 0
    ROCK = 1
    SEED = 2
    AGENT = 3
    BIOFUEL = 4
    
    # Display symbols
    SYMBOLS = {
        EMPTY: ' ',
        ROCK: '▲',
        SEED: '◆',
        AGENT: 'R',
        BIOFUEL: '*',
    }
    CLOUD_SYMBOL = '☁'
    
    def __init__(self, size=(12, 12), num_seeds=5, num_rocks=15, random_seed=None):
        """
        @param size: (rows, cols) grid dimensions
        @param num_seeds: number of seeds to place
        @param num_rocks: number of rock obstacles
        @param random_seed: optional seed for reproducibility
        """
        if random_seed is not None:
            np_seed(random_seed)
        
        self.size = size
        self.world = np.zeros(size, dtype=int)
        self.world_time = 0
        
        # Place rocks and seeds
        self._place_random(num_rocks, self.ROCK)
        self._place_random(num_seeds, self.SEED)
        
        # Cloud list: [(Cloud, direction), ...]
        self.clouds = []
        
        # Agent reference (set via add_agent)
        self.agent = None
        self._agent_start_pos = None
    
    def _get_empty_cells(self):
        """Return indices of empty cells."""
        return np.where(self.world == self.EMPTY)
    
    def _place_random(self, count, cell_type):
        """Place 'count' cells of given type randomly."""
        empty = self._get_empty_cells()
        num_empty = len(empty[0])
        
        if count > num_empty:
            count = num_empty
        
        if count == 0:
            return
        
        indices = choice(num_empty, count, replace=False)
        rows, cols = empty[0][indices], empty[1][indices]
        self.world[rows, cols] = cell_type
    
    def add_agent(self, agent_class, position=None, **agent_kwargs):
        """
        Add an agent to the world.
        
        @param agent_class: Agent class to instantiate
        @param position: (row, col) or None for random placement
        @param agent_kwargs: additional arguments for agent constructor
        """
        if position is None:
            empty = self._get_empty_cells()
            if len(empty[0]) == 0:
                raise ValueError("No empty cells for agent")
            idx = choice(len(empty[0]))
            position = (empty[0][idx], empty[1][idx])
        
        position = tuple(position)
        self._agent_start_pos = position
        
        # Create agent with reference to world state
        self.agent = agent_class(
            position=list(position),
            world=self,
            **agent_kwargs
        )
        
        self.world[position] = self.AGENT
    
    def reset_agent(self):
        """Reset agent to starting position with full battery."""
        if self.agent is None:
            return
        
        # Clear old position
        old_pos = tuple(self.agent.position)
        if self.world[old_pos] == self.AGENT:
            self.world[old_pos] = self.EMPTY
        
        # Reset to start
        self.agent.position = list(self._agent_start_pos)
        self.agent.battery = 2
        self.world[self._agent_start_pos] = self.AGENT
    
    def step(self, new_cloud_rate=0.2, cloud_limit=3, cloud_size=4, cloud_bounds=(4, 4)):
        """
        Advance world by one timestep.
        
        @param new_cloud_rate: probability of spawning a new cloud
        @param cloud_limit: maximum number of clouds
        @param cloud_size: cells per cloud
        @param cloud_bounds: bounding box for cloud shape
        """
        self.world_time += 1
        
        # Maybe spawn new cloud
        if len(self.clouds) < cloud_limit and binomial(1, new_cloud_rate):
            self._spawn_cloud(cloud_size, cloud_bounds)
        
        # Move existing clouds
        self._move_clouds()
    
    def _spawn_cloud(self, cloud_size, cloud_bounds):
        """Spawn a cloud at a random edge, moving inward."""
        edge = np.random.randint(0, 4)
        
        # Random position along the edge
        if edge in [0, 2]:  # Top or bottom
            col = np.random.randint(0, self.size[1] - cloud_bounds[1])
        else:
            col = -cloud_bounds[1] if edge == 1 else self.size[1]
        
        if edge in [1, 3]:  # Left or right
            row = np.random.randint(0, self.size[0] - cloud_bounds[0])
        else:
            row = -cloud_bounds[0] if edge == 0 else self.size[0]
        
        # Direction: clouds move toward center
        # 0=Up, 1=Left, 2=Down, 3=Right
        direction = (edge + 2) % 4  # Opposite of spawn edge
        
        cloud = Cloud(size=cloud_size, bounds=cloud_bounds, location=(row, col))
        self.clouds.append((cloud, direction))
    
    def _move_clouds(self):
        """Move all clouds and remove off-screen ones."""
        to_remove = []
        
        for cloud, direction in self.clouds:
            cloud.move(direction)
            
            # Check if completely off-screen
            on_screen = False
            for r, c in cloud:
                if 0 <= r < self.size[0] and 0 <= c < self.size[1]:
                    on_screen = True
                    break
            
            if not on_screen:
                to_remove.append((cloud, direction))
        
        for item in to_remove:
            self.clouds.remove(item)
    
    def is_under_cloud(self, position):
        """Check if a position is under any cloud."""
        for cloud, _ in self.clouds:
            if position in cloud:
                return True
        return False
    
    def get_cell(self, position):
        """Get cell type at position."""
        r, c = position
        if 0 <= r < self.size[0] and 0 <= c < self.size[1]:
            return self.world[r, c]
        return None
    
    def is_valid_position(self, position):
        """Check if position is within bounds and not a rock."""
        r, c = position
        if not (0 <= r < self.size[0] and 0 <= c < self.size[1]):
            return False
        return self.world[r, c] != self.ROCK
    
    def copy(self):
        """Create a deep copy of the world (for simulation without side effects)."""
        return deepcopy(self)
    
    def render(self, trajectory=None):
        """
        Render world as string.
        
        @param trajectory: optional list of positions to overlay
        @return: string representation
        """
        # Build trajectory markers
        traj_markers = {}
        if trajectory and len(trajectory) > 1:
            arrows = {
                (-1, 0): '↑', (1, 0): '↓',
                (0, -1): '←', (0, 1): '→',
                (0, 0): '•'
            }
            for i in range(len(trajectory) - 1):
                pos = tuple(trajectory[i])
                next_pos = tuple(trajectory[i + 1])
                delta = (next_pos[0] - pos[0], next_pos[1] - pos[1])
                traj_markers[pos] = arrows.get(delta, '•')
            traj_markers[tuple(trajectory[-1])] = '⊗'  # End marker
        
        lines = []
        for r in range(self.size[0]):
            row_str = ""
            for c in range(self.size[1]):
                pos = (r, c)
                
                # Priority: trajectory > cloud > cell
                if pos in traj_markers:
                    symbol = traj_markers[pos]
                elif self.is_under_cloud(pos):
                    symbol = self.CLOUD_SYMBOL
                else:
                    cell = self.world[r, c]
                    symbol = self.SYMBOLS.get(cell, '?')
                
                row_str += f'{symbol:<2}'
            lines.append(row_str)
        
        return '\n'.join(lines)
    
    def __str__(self):
        return self.render()
    
    def __contains__(self, pos):
        """Check if position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.size[0] and 0 <= c < self.size[1]


def encode_state(world: SciWrld, position=None):
    """
    Encode world state as feature vector for neural network.
    
    Features:
    - agent_row, agent_col (normalized)
    - nearest_seed_row, nearest_seed_col (normalized, or -1 if none)
    - under_cloud (binary)
    - battery (normalized)
    - time (normalized)
    
    @param world: SciWrld instance
    @param position: optional position override (default: agent position)
    @return: numpy array of shape (7,)
    """
    if position is None:
        if world.agent is None:
            raise ValueError("No agent in world and no position provided")
        position = world.agent.position
    
    ar, ac = position
    battery = world.agent.battery if world.agent else 2
    t = world.world_time
    
    # Normalize by grid size
    norm_r = ar / world.size[0]
    norm_c = ac / world.size[1]
    
    # Find nearest seed
    seed_positions = np.argwhere(world.world == world.SEED)
    if len(seed_positions) > 0:
        dists = np.abs(seed_positions[:, 0] - ar) + np.abs(seed_positions[:, 1] - ac)
        nearest = seed_positions[np.argmin(dists)]
        sr, sc = nearest[0] / world.size[0], nearest[1] / world.size[1]
    else:
        sr, sc = -1, -1
    
    # Check if under cloud
    in_cloud = 1.0 if world.is_under_cloud(position) else 0.0
    
    return np.array([
        norm_r, norm_c,
        sr, sc,
        in_cloud,
        battery / 2.0,
        min(t / 100.0, 1.0)  # Cap time normalization
    ], dtype=np.float32)