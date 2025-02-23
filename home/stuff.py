# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from taipy.gui import Gui, Html
# import taipy.gui.builder as tgb
# import ast
# from home.utils import plot_stance, compute_rt_stats, compute_accuracy_stats, compute_defense_stats, compute_dodge_stats
#
# # df = pd.read_csv("output.csv")
# #
# # rt_stats = compute_rt_stats(df)
# # accuracy_stats = compute_accuracy_stats(df)
# # defense_stats = compute_defense_stats(df)
# # dodge_stats = compute_dodge_stats(df)
# # stance_plots = plot_stance(df)
# # image_path = "/Users/shreyagarwal/Desktop/taipy/stance_0.png"
# #
# # color_map_rt = {
# #     0.0: "green",
# #     (rt_stats[4]/3): "yellow",
# #     (rt_stats[4]*2/3): "red"
# # }
# #
# # html_file_path = "taipy.html"
#
# # with open(html_file_path, "r", encoding="utf-8") as file:
# #     html_string = file.read()
#
# def return_html_content(csv_file_path):
#     df = pd.read_csv(csv_file_path)
#
#     rt_stats = compute_rt_stats(df)
#     accuracy_stats = compute_accuracy_stats(df)
#     defense_stats = compute_defense_stats(df)
#     dodge_stats = compute_dodge_stats(df)
#     stance_plots = plot_stance(df)
#
#     color_map_rt = {
#         0.0: "green",
#         (rt_stats[4] / 3): "yellow",
#         (rt_stats[4] * 2 / 3): "red"
#     }
#
#     html_file_path = "templates/home/taipy.html"
#
#     with open(html_file_path, "r", encoding="utf-8") as file:
#         html_string = file.read()
#
#     return Html(html_string)
