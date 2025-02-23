import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from taipy import Gui
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


def generate_stance_images(punch_type):
    return "".join(
        f'<taipy:image>{img_path}</taipy:image>'
        for punch, img_path in stance_plots if punch == punch_type
    )


color_map_rt = {
    0.0: "green",
    (rt_stats[4]/3): "yellow",
    (rt_stats[4]*2/3): "red"
}

index_file_path = "index.html"
with open(index_file_path, "r", encoding="utf-8") as file:
    index_string = file.read()

sim_file_path = "sim.html"
with open(sim_file_path, "r", encoding="utf-8") as file:
    sim_string = file.read()

sim_md_file_path = "sim.md"
with open(sim_md_file_path, "r", encoding="utf-8") as file:
    sim_md_string = file.read()

if __name__ == "__main__":
    index_html = Html(index_string)
    Gui(index_html).run()
