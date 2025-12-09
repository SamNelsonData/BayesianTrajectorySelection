"""
Beautiful visualization tool for SciWorld trajectories.

Creates color-coded gridworld displays with:
- Clear trajectory paths
- Color-coded elements (seeds, rocks, clouds)
- Trajectory statistics
- Side-by-side comparisons

Usage:
    visualizer = TrajectoryVisualizer()
    visualizer.show_trajectory_pair(world1, traj1, world2, traj2)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
from BayesianTrajectorySelection.arch.sciwrld import SciWrld


class TrajectoryVisualizer:
    """
    Beautiful visualizations for SciWorld trajectories.
    """
    
    # Color scheme
    COLORS = {
        'sand': '#F4E4C1',      # Light tan
        'rock': '#5D5D5D',      # Dark gray
        'seed': '#4CAF50',      # Green
        'agent': '#2196F3',     # Blue
        'cloud': '#90CAF9',     # Light blue
        'path': '#FF9800',      # Orange
        'start': '#4CAF50',     # Green
        'end': '#F44336',       # Red
    }
    
    def __init__(self, figsize=(14, 6)):
        self.figsize = figsize
    
    def show_trajectory_pair(self, world1, traj1, world2, traj2, 
                            reward1=None, reward2=None,
                            title1="Trajectory 1", title2="Trajectory 2"):
        """
        Show two trajectories side-by-side for comparison.
        
        @param world1: SciWrld instance for trajectory 1
        @param traj1: trajectory 1 positions
        @param world2: SciWrld instance for trajectory 2  
        @param traj2: trajectory 2 positions
        @param reward1: computed reward for trajectory 1
        @param reward2: computed reward for trajectory 2
        @param title1: title for left plot
        @param title2: title for right plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot trajectory 1
        stats1 = self._plot_trajectory(ax1, world1, traj1, title1, reward1)
        
        # Plot trajectory 2
        stats2 = self._plot_trajectory(ax2, world2, traj2, title2, reward2)
        
        # Add legend (shared)
        self._add_legend(fig)
        
        plt.tight_layout()
        plt.show()
        
        return stats1, stats2
    
    def show_single_trajectory(self, world, trajectory, title="Trajectory", reward=None):
        """
        Show a single trajectory.
        
        @param world: SciWrld instance
        @param trajectory: list of positions
        @param title: plot title
        @param reward: computed reward
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        stats = self._plot_trajectory(ax, world, trajectory, title, reward)
        self._add_legend(fig)
        
        plt.tight_layout()
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
        grid_size = world.size[0]
        
        # Create color grid
        color_grid = np.zeros((grid_size, grid_size, 3))
        
        # Fill in world elements
        for r in range(grid_size):
            for c in range(grid_size):
                val = world.world[r, c]
                
                if val == world.item_to_value['Sand']:
                    color = self._hex_to_rgb(self.COLORS['sand'])
                elif val == world.item_to_value['Rock']:
                    color = self._hex_to_rgb(self.COLORS['rock'])
                elif val == world.item_to_value['Seed']:
                    color = self._hex_to_rgb(self.COLORS['seed'])
                elif val == world.item_to_value['Agent']:
                    color = self._hex_to_rgb(self.COLORS['agent'])
                else:
                    color = self._hex_to_rgb(self.COLORS['sand'])
                
                color_grid[r, c] = color
        
        # Overlay clouds
        for cloud, _ in world.clouds:
            for r, c in cloud:
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    color_grid[r, c] = self._hex_to_rgb(self.COLORS['cloud'])
        
        # Display grid
        ax.imshow(color_grid, interpolation='nearest')
        
        # Compute trajectory statistics
        stats = self._compute_trajectory_stats(world, trajectory)
        
        # Draw trajectory path
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                r1, c1 = trajectory[i]
                r2, c2 = trajectory[i + 1]
                
                # Draw arrow
                arrow = FancyArrowPatch(
                    (c1, r1), (c2, r2),
                    arrowstyle='->', 
                    mutation_scale=20,
                    linewidth=2.5,
                    color=self.COLORS['path'],
                    zorder=10
                )
                ax.add_patch(arrow)
        
        # Mark start position
        if trajectory:
            start_r, start_c = trajectory[0]
            circle = Circle(
                (start_c, start_r), 0.3,
                color=self.COLORS['start'],
                zorder=11
            )
            ax.add_patch(circle)
            ax.text(start_c, start_r, 'S', 
                   ha='center', va='center', 
                   fontweight='bold', fontsize=10, color='white',
                   zorder=12)
        
        # Mark end position
        if trajectory:
            end_r, end_c = trajectory[-1]
            circle = Circle(
                (end_c, end_r), 0.3,
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
        from copy import deepcopy
        
        world_copy = deepcopy(world)
        
        seeds_collected = 0
        time_under_clouds = 0
        battery_depletions = 0
        battery = 2
        
        for pos in trajectory:
            # Check seed
            if world_copy.world[pos] == world_copy.item_to_value['Seed']:
                seeds_collected += 1
                world_copy.world[pos] = world_copy.item_to_value['Sand']
            
            # Check cloud
            under_cloud = False
            for cloud, _ in world_copy.clouds:
                if pos in cloud:
                    under_cloud = True
                    time_under_clouds += 1
                    battery -= 1
                    break
            
            if not under_cloud and battery < 2:
                battery = min(2, battery + 1)
            
            if battery <= 0:
                battery_depletions += 1
        
        return {
            'seeds_collected': seeds_collected,
            'time_under_clouds': time_under_clouds,
            'battery_depletions': battery_depletions,
            'final_battery': battery,
            'path_length': len(trajectory)
        }
    
    def _add_legend(self, fig):
        """Add a legend explaining the colors."""
        legend_elements = [
            mpatches.Patch(color=self.COLORS['sand'], label='Sand'),
            mpatches.Patch(color=self.COLORS['rock'], label='Rock'),
            mpatches.Patch(color=self.COLORS['seed'], label='Seed (+10)'),
            mpatches.Patch(color=self.COLORS['cloud'], label='Cloud (-5, -1 battery)'),
            mpatches.Patch(color=self.COLORS['path'], label='Trajectory path'),
            mpatches.Patch(color=self.COLORS['start'], label='Start (S)'),
            mpatches.Patch(color=self.COLORS['end'], label='End (E)'),
        ]
        
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=7,
            frameon=False,
            fontsize=9
        )
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple (0-1 range)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def save_comparison(self, world1, traj1, world2, traj2, 
                       reward1=None, reward2=None, filename='comparison.png'):
        """
        Save trajectory comparison to file.
        
        @param filename: output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        self._plot_trajectory(ax1, world1, traj1, "Trajectory 1", reward1)
        self._plot_trajectory(ax2, world2, traj2, "Trajectory 2", reward2)
        self._add_legend(fig)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved to {filename}")


def demo():
    """
    Demo of the visualizer with random trajectories.
    """
    from BayesianTrajectorySelection.arch.sciwrld import SciWrld
    from BayesianTrajectorySelection.arch.Agents import Agent
    
    print("Creating demo visualizations...")
    
    # Create two worlds
    world1 = SciWrld(size=(12, 12), starting_seeds=5, rocks=15)
    world1.add_agent(Agent)
    
    # Add clouds
    for _ in range(2):
        world1.step(steps=0, new_cloud_rate=1.0, cloud_limit=3, cloud_size=4)
    
    world2 = SciWrld(size=(12, 12), starting_seeds=5, rocks=15)
    world2.add_agent(Agent)
    
    for _ in range(2):
        world2.step(steps=0, new_cloud_rate=1.0, cloud_limit=3, cloud_size=4)
    
    # Generate trajectories
    traj1 = world1.agent.gen_trajectory(steps=10)
    traj2 = world2.agent.gen_trajectory(steps=10)
    
    # Visualize
    viz = TrajectoryVisualizer()
    
    print("\nShowing trajectory comparison...")
    viz.show_trajectory_pair(
        world1, traj1, world2, traj2,
        reward1=9.2, reward2=-0.8,
        title1="Good Trajectory", 
        title2="Poor Trajectory"
    )
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()