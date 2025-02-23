import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from taipy.gui import Gui, Html
import taipy.gui.builder as tgb
import ast
from utils import plot_stance, compute_rt_stats, compute_accuracy_stats, compute_defense_stats, compute_dodge_stats

df = pd.read_csv("output.csv")

rt_stats = compute_rt_stats(df)
accuracy_stats = compute_accuracy_stats(df)
defense_stats = compute_defense_stats(df)
dodge_stats = compute_dodge_stats(df)
stance_plots = plot_stance(df)
image_path = "/Users/shreyagarwal/Desktop/taipy/stance_0.png"

color_map_rt = {
    0.0: "green",
    (rt_stats[4]/3): "yellow",
    (rt_stats[4]*2/3): "red"
}

html_file_path = "index.html"

with open(html_file_path, "r", encoding="utf-8") as file:
    html_string = file.read()


if __name__ == "__main__":
    page = Html(html_string)
    Gui(page).run()
