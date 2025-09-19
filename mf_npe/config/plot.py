# mf_npe/config/plot.py

"""
Defines a centralized style configuration for all plots in the project.
"""

from typing import Dict, Any

PLOTTING_STYLE: Dict[str, Any] = {
    # Figure dimensions in pixels
    "width_px": 700,
    "height_px": 500,

    # Font sizes for text elements
    "label_font_size": 25,
    "title_font_size": 27,

    # Line widths and colors
    "grid_linewidth": 2,
    "axis_color": '#6A798F',  # A muted blue-grey

    # Controls whether plots are displayed interactively by default. 
    # False for batch processing or saving figures without showing them.
    "show_plots_default": False}
