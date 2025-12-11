
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np


class TrajectoryVisualizer:

    # Color scheme
    COLORS = {
        'sand': '#F4E4C1',      # Light tan
        'rock': '#5D5D5D',      # Dark gray
        'seed': '#4CAF50',      # Green
        'agent': '#2196F3',     # Blue
        'cloud': '#90CAF9',     # Light blue (semi-transparent)
        'cloud_dark': '#64B5F6', # Darker blue for cloud overlay
        'path': '#FF9800',      # Orange
        'start': '#4CAF50',     # Green
        'end': '#F44336',       # Red
    }
    
    def __init__(self, figsize=(14, 6)):
        self.figsize = figsize
    
    def _get_grid_size(self, world):
        """Get grid size from world object (handles different attribute names)."""
        if hasattr(world, 'size'):
            if isinstance(world.size, tuple):
                return world.size[0]
            return world.size
        elif hasattr(world, 'grid_size'):
            return world.grid_size
        elif hasattr(world, 'world'):
            return world.world.shape[0]
        else:
            return 12  # Default fallback
    
    def _get_cell_value(self, world, r, c):
        """Get cell value from world grid."""
        if hasattr(world, 'world'):
            return world.world[r, c]
        elif hasattr(world, 'grid'):
            return world.grid[r, c]
        return 0
    
    def _get_item_value(self, world, item_name):
        """Get numeric value for an item type."""
        if hasattr(world, 'item_to_value'):
            return world.item_to_value.get(item_name, -1)
        # Common defaults
        defaults = {'Sand': 0, 'Rock': 1, 'Seed': 2, 'Agent': 3}
        return defaults.get(item_name, -1)
    
    def _get_cloud_positions(self, world):
        """Extract cloud positions from world."""
        cloud_positions = set()
        
        if hasattr(world, 'clouds'):
            for cloud_data in world.clouds:
                # Handle different cloud data structures
                if isinstance(cloud_data, tuple):
                    cloud = cloud_data[0]  # (cloud_positions, velocity) format
                elif isinstance(cloud_data, list):
                    cloud = cloud_data
                else:
                    cloud = cloud_data
                
                # Cloud is a list of positions
                if isinstance(cloud, (list, set)):
                    for pos in cloud:
                        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                            cloud_positions.add((pos[0], pos[1]))
        
        return cloud_positions
    
    def _get_seed_positions(self, world):
        """Extract seed positions from world."""
        seed_positions = set()
        grid_size = self._get_grid_size(world)
        seed_value = self._get_item_value(world, 'Seed')
        
        for r in range(grid_size):
            for c in range(grid_size):
                if self._get_cell_value(world, r, c) == seed_value:
                    seed_positions.add((r, c))
        
        return seed_positions
    
    def show_trajectory_pair(self, world1, traj1, world2, traj2, 
                            reward1=None, reward2=None,
                            title1="Trajectory 1", title2="Trajectory 2",
                            save_path=None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot trajectory 1
        stats1 = self._plot_trajectory(ax1, world1, traj1, title1, reward1)
        
        # Plot trajectory 2
        stats2 = self._plot_trajectory(ax2, world2, traj2, title2, reward2)
        
        # Add legend (shared)
        self._add_legend(fig)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for legend
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved to {save_path}")
        else:
            plt.show()
        
        return stats1, stats2
    
    def show_single_trajectory(self, world, trajectory, title="Trajectory", 
                               reward=None, save_path=None):
        """
        Show a single trajectory.
        
        @param world: SciWrld instance
        @param trajectory: list of positions
        @param title: plot title
        @param reward: computed reward
        @param save_path: if provided, save instead of showing
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        stats = self._plot_trajectory(ax, world, trajectory, title, reward)
        self._add_legend(fig)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved to {save_path}")
        else:
            plt.show()
        
        return stats
    
    def _plot_trajectory(self, ax, world, trajectory, title, reward=None):
        """
        Plot a single trajectory on given axes.
        
        @param ax: matplotlib axes
        @param world: SciWrld instance
        @param trajectory: list of positions
        @param title: plot title
        @param reward: optional reward value
        @return: trajectory statistics dict
        """
        grid_size = self._get_grid_size(world)
        
        # Get world elements
        cloud_positions = self._get_cloud_positions(world)
        seed_value = self._get_item_value(world, 'Seed')
        rock_value = self._get_item_value(world, 'Rock')
        
        # Create color grid (RGB)
        color_grid = np.zeros((grid_size, grid_size, 3))
        
        # Fill base terrain
        for r in range(grid_size):
            for c in range(grid_size):
                val = self._get_cell_value(world, r, c)
                
                if val == rock_value:
                    color = self._hex_to_rgb(self.COLORS['rock'])
                elif val == seed_value:
                    color = self._hex_to_rgb(self.COLORS['seed'])
                else:
                    color = self._hex_to_rgb(self.COLORS['sand'])
                
                color_grid[r, c] = color
        
        # Overlay clouds
        for (r, c) in cloud_positions:
            if 0 <= r < grid_size and 0 <= c < grid_size:
                color_grid[r, c] = self._hex_to_rgb(self.COLORS['cloud'])
        
        # Display grid
        ax.imshow(color_grid, interpolation='nearest')
        
        # Compute trajectory statistics
        stats = self._compute_trajectory_stats(world, trajectory)
        
        # Convert trajectory to list if needed
        traj_list = list(trajectory) if not isinstance(trajectory, list) else trajectory
        
        # Draw trajectory path
        if len(traj_list) > 1:
            for i in range(len(traj_list) - 1):
                pos1 = traj_list[i]
                pos2 = traj_list[i + 1]
                
                # Handle different position formats
                if isinstance(pos1, (tuple, list)):
                    r1, c1 = pos1[0], pos1[1]
                else:
                    continue
                    
                if isinstance(pos2, (tuple, list)):
                    r2, c2 = pos2[0], pos2[1]
                else:
                    continue
                
                # Draw arrow
                arrow = FancyArrowPatch(
                    (c1, r1), (c2, r2),
                    arrowstyle='->', 
                    mutation_scale=15,
                    linewidth=2.5,
                    color=self.COLORS['path'],
                    zorder=10
                )
                ax.add_patch(arrow)
        
        # Mark start position
        if traj_list:
            start_pos = traj_list[0]
            if isinstance(start_pos, (tuple, list)):
                start_r, start_c = start_pos[0], start_pos[1]
                circle = Circle(
                    (start_c, start_r), 0.35,
                    color=self.COLORS['start'],
                    zorder=11
                )
                ax.add_patch(circle)
                ax.text(start_c, start_r, 'S', 
                       ha='center', va='center', 
                       fontweight='bold', fontsize=10, color='white',
                       zorder=12)
        
        # Mark end position
        if traj_list:
            end_pos = traj_list[-1]
            if isinstance(end_pos, (tuple, list)):
                end_r, end_c = end_pos[0], end_pos[1]
                circle = Circle(
                    (end_c, end_r), 0.35,
                    color=self.COLORS['end'],
                    zorder=11
                )
                ax.add_patch(circle)
                ax.text(end_c, end_r, 'E',
                       ha='center', va='center',
                       fontweight='bold', fontsize=10, color='white',
                       zorder=12)
        
        # Configure axes
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal')
        
        # Title with stats
        title_str = f"{title}\n"
        title_str += f"Seeds: {stats['seeds_collected']} | "
        title_str += f"Cloud steps: {stats['time_under_clouds']} | "
        title_str += f"Battery warnings: {stats['battery_depletions']}"
        
        if reward is not None:
            title_str += f"\nReward: {reward:.2f}"
        
        ax.set_title(title_str, fontsize=11, pad=10)
        
        return stats
    
    def _compute_trajectory_stats(self, world, trajectory):
        """
        Compute statistics for a trajectory.
        
        @param world: SciWrld instance
        @param trajectory: list of positions
        @return: dict with statistics
        """
        grid_size = self._get_grid_size(world)
        cloud_positions = self._get_cloud_positions(world)
        seed_value = self._get_item_value(world, 'Seed')
        
        # Track which seeds we've collected (avoid double counting)
        collected_seeds = set()
        seeds_collected = 0
        time_under_clouds = 0
        battery_depletions = 0
        battery = 2  # Starting battery
        
        traj_list = list(trajectory) if not isinstance(trajectory, list) else trajectory
        
        for pos in traj_list:
            if not isinstance(pos, (tuple, list)):
                continue
            
            r, c = pos[0], pos[1]
            
            # Check if in bounds
            if not (0 <= r < grid_size and 0 <= c < grid_size):
                continue
            
            # Check seed collection
            if (r, c) not in collected_seeds:
                if self._get_cell_value(world, r, c) == seed_value:
                    seeds_collected += 1
                    collected_seeds.add((r, c))
            
            # Check cloud
            if (r, c) in cloud_positions:
                time_under_clouds += 1
                battery -= 1
            else:
                # Recharge in sunlight
                if battery < 2:
                    battery = min(2, battery + 1)
            
            # Check battery depletion
            if battery <= 0:
                battery_depletions += 1
                battery = 1  # Partial recovery
        
        # Compute average movement
        total_movement = 0
        for i in range(1, len(traj_list)):
            if isinstance(traj_list[i], (tuple, list)) and isinstance(traj_list[i-1], (tuple, list)):
                r1, c1 = traj_list[i-1][0], traj_list[i-1][1]
                r2, c2 = traj_list[i][0], traj_list[i][1]
                total_movement += abs(r2 - r1) + abs(c2 - c1)
        
        avg_movement = total_movement / max(1, len(traj_list) - 1)
        
        return {
            'seeds_collected': seeds_collected,
            'time_under_clouds': time_under_clouds,
            'battery_depletions': battery_depletions,
            'final_battery': max(0, battery),
            'avg_movement': avg_movement,
            'path_length': len(traj_list)
        }
    
    def _add_legend(self, fig):
        """Add a legend explaining the colors."""
        legend_elements = [
            mpatches.Patch(color=self.COLORS['sand'], label='Sand'),
            mpatches.Patch(color=self.COLORS['rock'], label='Rock'),
            mpatches.Patch(color=self.COLORS['seed'], label='Seed (+10)'),
            mpatches.Patch(color=self.COLORS['cloud'], label='Cloud (-1, drains battery)'),
            mpatches.Patch(color=self.COLORS['path'], label='Path'),
            mpatches.Patch(color=self.COLORS['start'], label='Start'),
            mpatches.Patch(color=self.COLORS['end'], label='End'),
        ]
        
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=7,
            frameon=False,
            fontsize=9
        )
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple (0-1 range)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def create_demo_world(grid_size=12, num_seeds=5, num_rocks=10, num_clouds=2):
    """
    Create a simple demo world without depending on SciWrld.
    Returns a mock world object for testing the visualizer.
    """
    class MockWorld:
        def __init__(self, size, num_seeds, num_rocks, num_clouds):
            self.size = (size, size)
            self.item_to_value = {'Sand': 0, 'Rock': 1, 'Seed': 2, 'Agent': 3}
            
            # Create grid
            self.world = np.zeros((size, size), dtype=int)
            
            # Add rocks randomly
            rock_positions = set()
            while len(rock_positions) < num_rocks:
                r, c = np.random.randint(0, size, 2)
                if (r, c) not in rock_positions:
                    rock_positions.add((r, c))
                    self.world[r, c] = self.item_to_value['Rock']
            
            # Add seeds randomly
            seed_positions = set()
            while len(seed_positions) < num_seeds:
                r, c = np.random.randint(0, size, 2)
                if self.world[r, c] == 0 and (r, c) not in seed_positions:
                    seed_positions.add((r, c))
                    self.world[r, c] = self.item_to_value['Seed']
            
            # Add clouds
            self.clouds = []
            for _ in range(num_clouds):
                # Random cloud center
                cr, cc = np.random.randint(2, size-2, 2)
                cloud_cells = []
                # Create 3x3 cloud
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            cloud_cells.append((nr, nc))
                self.clouds.append((cloud_cells, (0, 0)))  # (positions, velocity)
    
    return MockWorld(grid_size, num_seeds, num_rocks, num_clouds)


def generate_random_trajectory(world, steps=8):
    """Generate a random trajectory in the world."""
    grid_size = world.size[0] if isinstance(world.size, tuple) else world.size
    
    # Start at random position
    start_r = np.random.randint(1, grid_size - 1)
    start_c = np.random.randint(1, grid_size - 1)
    
    trajectory = [(start_r, start_c)]
    
    # Random walk
    for _ in range(steps - 1):
        r, c = trajectory[-1]
        
        # Random direction
        dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)][np.random.randint(5)]
        
        new_r = max(0, min(grid_size - 1, r + dr))
        new_c = max(0, min(grid_size - 1, c + dc))
        
        trajectory.append((new_r, new_c))
    
    return trajectory


def demo():
    """
    Demo of the visualizer with mock worlds.
    """
    print("Creating demo visualizations...")
    print("(Using mock worlds - replace with your SciWrld for real usage)\n")
    
    # Create two mock worlds
    np.random.seed(42)
    world1 = create_demo_world(grid_size=12, num_seeds=5, num_rocks=10, num_clouds=2)
    world2 = create_demo_world(grid_size=12, num_seeds=5, num_rocks=10, num_clouds=2)
    
    # Generate random trajectories
    traj1 = generate_random_trajectory(world1, steps=10)
    traj2 = generate_random_trajectory(world2, steps=10)
    
    # Create visualizer
    viz = TrajectoryVisualizer()
    
    # Show comparison
    print("Showing trajectory comparison...")
    stats1, stats2 = viz.show_trajectory_pair(
        world1, traj1, world2, traj2,
        reward1=9.10, reward2=-10.90,
        title1="Good Trajectory",
        title2="Poor Trajectory"
    )
    
    print("\nTrajectory 1 stats:", stats1)
    print("Trajectory 2 stats:", stats2)
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()