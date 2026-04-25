"""
Pygame-based renderer for LevelForge grid visualization.
"""

import os
import sys
from typing import List

import pygame
from PIL import Image

# Path setup for imports
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# Tile size and colors
TILE_SIZE = 32
TILE_COLORS = {
    ".": (230, 230, 230),  # light gray (empty)
    "#": (80, 80, 80),     # dark gray (wall)
    "^": (220, 50, 50),    # red (spike)
    "$": (255, 200, 0),    # yellow (coin)
    "E": (150, 50, 200),   # purple (enemy)
    "P": (50, 100, 220),   # blue (player start)
    "G": (50, 200, 80),    # green (goal)
}
BORDER_COLOR = (0, 0, 0)  # black
BORDER_WIDTH = 1
TEXT_COLOR = (0, 0, 0)    # black


def render_grid(grid: List[List[str]], title: str = "", output_path: str = "level.png") -> None:
    """
    Render a grid to a PNG file using pygame.

    Args:
        grid: 2D list of tile characters (8 rows x 16 cols)
        title: Optional title text to display above the grid
        output_path: Path to save the PNG file
    """
    # Initialize pygame in headless mode
    pygame.init()
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Calculate dimensions
    grid_width = cols * TILE_SIZE
    grid_height = rows * TILE_SIZE
    title_height = 40 if title else 0

    # Create surface
    surface = pygame.Surface((grid_width, grid_height + title_height))
    surface.fill((255, 255, 255))  # white background

    # Render title if provided
    if title:
        font = pygame.font.Font(None, 28)
        text_surface = font.render(title, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(grid_width // 2, title_height // 2))
        surface.blit(text_surface, text_rect)

    # Render grid tiles
    for row in range(rows):
        for col in range(cols):
            tile = grid[row][col]
            color = TILE_COLORS.get(tile, (200, 200, 200))  # default to light gray

            # Calculate tile position (offset by title height)
            x = col * TILE_SIZE
            y = row * TILE_SIZE + title_height

            # Draw filled rectangle
            pygame.draw.rect(surface, color, (x, y, TILE_SIZE, TILE_SIZE))

            # Draw border
            pygame.draw.rect(surface, BORDER_COLOR, (x, y, TILE_SIZE, TILE_SIZE), BORDER_WIDTH)

    # Save to file
    pygame.image.save(surface, output_path)
    pygame.quit()


def make_evolution_gif(
    grids: List[List[List[str]]],
    labels: List[str],
    output_path: str = "evolution.gif"
) -> None:
    """
    Create an animated GIF showing grid evolution over training.

    Args:
        grids: List of grids (one per checkpoint)
        labels: List of labels for each grid (e.g., "Step 0", "Step 50")
        output_path: Path to save the GIF file
    """
    if len(grids) != len(labels):
        raise ValueError("Number of grids must match number of labels")

    # Render each grid to a temporary PNG
    temp_files = []
    for i, (grid, label) in enumerate(zip(grids, labels)):
        temp_path = f"/tmp/frame_{i}.png"
        render_grid(grid, title=label, output_path=temp_path)
        temp_files.append(temp_path)

    # Load all frames as PIL Images
    frames = [Image.open(f) for f in temp_files]

    # Save as animated GIF
    # Each frame shows for 1500ms (1.5 seconds)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1500,
        loop=0
    )

    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tilemap import make_basic_level

    # Test render_grid
    grid = make_basic_level()
    render_grid(grid.to_list(), title="Test Render", output_path="test_render.png")
    print("Rendered to test_render.png")
