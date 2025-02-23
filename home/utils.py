import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from matplotlib.lines import Line2D

vector_pairs = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (23, 24),
    (11, 23), (12, 24), (23, 25),
    (25, 27), (24, 26), (26, 28)
]

professional_stances = {
    "jab": [(242, 95), (248, 87), (250, 87), (252, 87), (241, 84), (237, 82), (233, 80), (248, 85), (221, 73), (239, 102), (231, 98), (243, 104), (173, 93), (222, 110), (123, 106), (217, 111), (150, 110), (217, 110), (160, 107), (220, 112), (162, 102), (219, 112), (161, 108), (203, 208), (156, 205), (232, 305), (131, 306), (209, 388), (120, 351), (196, 401), (121, 354), (233, 427), (111, 394)],
    "hook": [(203, 232), (209, 217), (214, 216), (219, 216), (193, 215), (184, 213), (176, 212), (222, 220), (158, 215), (208, 251), (188, 249), (253, 292), (104, 276), (289, 401), (58, 377), (247, 298), (145, 348), (239, 272), (171, 336), (234, 259), (172, 314), (233, 267), (168, 320), (203, 515), (121, 504), (214, 672), (84, 622), (239, 821), (119, 667), (251, 851), (137, 680), (193, 866), (72, 712)],
    "uppercut": [(375, 155), (375, 141), (377, 140), (380, 139), (363, 142), (356, 141), (350, 140), (375, 139), (334, 142), (378, 170), (361, 171), (410, 196), (282, 226), (477, 255), (334, 324), (417, 179), (357, 205), (406, 158), (357, 177), (399, 154), (347, 170), (398, 163), (347, 178), (414, 421), (331, 431), (432, 549), (297, 580), (429, 702), (281, 611), (428, 719), (284, 605), (409, 762), (261, 665)]
}

def plot_stance(data: pd.DataFrame) -> list:
    figures = []

    for index, row in data.iterrows():
        try:
            punch_type, coord_list = ast.literal_eval(row["Stance"])

            x_coords = []
            y_coords = []

            prof_coord_list = professional_stances.get(punch_type)
            offset_x = prof_coord_list[0][0] - (coord_list[0][0]*500)
            offset_y = prof_coord_list[0][1] - (coord_list[0][1]*700)

            fig, ax = plt.subplots()
            for start, end in vector_pairs:
                if start < len(coord_list) and end < len(coord_list):
                    x_coords.append(coord_list[start][0]*500+offset_x)
                    x_coords.append(coord_list[end][0]*500+offset_x)
                    y_coords.append(coord_list[start][1]*700+offset_y)
                    y_coords.append(coord_list[end][1]*700+offset_y)
                    x = [(coord_list[start][0])*500+offset_x,
                         (coord_list[end][0])*500+offset_x]
                    y = [(coord_list[start][1])*700+offset_y,
                         (coord_list[end][1])*700+offset_y]

                    ax.plot(x, y, color="red")

            for start, end in vector_pairs:
                if start < len(prof_coord_list) and end < len(prof_coord_list):
                    x_coords.append(prof_coord_list[start][0])
                    x_coords.append(prof_coord_list[end][0])
                    y_coords.append(prof_coord_list[start][1])
                    y_coords.append(prof_coord_list[end][1])
                    x = [prof_coord_list[start][0], prof_coord_list[end][0]]
                    y = [prof_coord_list[start][1], prof_coord_list[end][1]]

                    ax.plot(x, y, color="blue")

            print(x_coords)
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
            ax.set_ylim(min(y_coords) - 10, max(y_coords) + 10)
            ax.set_title(f"Stance for {punch_type}")
            ax.invert_yaxis()
            ax.grid(True)
            legend_elements = [Line2D([0], [0], color='red', lw=2, label='User'),
                               Line2D([0], [0], color='blue', lw=2, label='Professional')]
            ax.legend(handles=legend_elements)

            plt.savefig(
                f"home/static/home/images/stance_{index}.png", bbox_inches="tight")
            plt.close(fig)
            figures.append(
                (punch_type, f"home/static/home/images/stance_{index}.png"))
            print(figures)

        except Exception as e:
            print(e)

    return figures



def compute_rt_stats(data: pd.DataFrame) -> list:
    avg_reaction_time = data["Reaction Time"].apply(
        lambda x: ast.literal_eval(x)[0]).mean()
    avg_jab_rt = data[data["Reaction Time"].apply(lambda x: ast.literal_eval(x)[1] == (
        "jab"))]["Reaction Time"].apply(lambda x: ast.literal_eval(x)[0]).mean()
    avg_hook_rt = data[data["Reaction Time"].apply(lambda x: ast.literal_eval(x)[1] == (
        "hook"))]["Reaction Time"].apply(lambda x: ast.literal_eval(x)[0]).mean()
    avg_uppercut_rt = data[data["Reaction Time"].apply(lambda x: ast.literal_eval(x)[1] == (
        "uppercut"))]["Reaction Time"].apply(lambda x: ast.literal_eval(x)[0]).mean()
    max_reaction_time = data["Reaction Time"].apply(
        lambda x: ast.literal_eval(x)[0]).max()

    return [avg_reaction_time, avg_jab_rt, avg_hook_rt, avg_uppercut_rt, max_reaction_time]


def compute_accuracy_stats(data: pd.DataFrame) -> list:
    jab_hits = ast.literal_eval(data["Accuracy"].iloc[0])[0]
    jab_attempts = ast.literal_eval(data["Accuracy"].iloc[0])[1]
    hook_hits = ast.literal_eval(data["Accuracy"].iloc[1])[0]
    hook_attempts = ast.literal_eval(data["Accuracy"].iloc[1])[1]
    uppercut_hits = ast.literal_eval(data["Accuracy"].iloc[2])[0]
    uppercut_attempts = ast.literal_eval(data["Accuracy"].iloc[2])[1]

    if (jab_attempts == 0):
        jab_attempts = jab_hits
    if (hook_attempts == 0):
        hook_attempts = hook_hits
    if (uppercut_attempts == 0):
        uppercut_attempts = uppercut_hits

    jab_accuracy = jab_hits / jab_attempts
    hook_accuracy = hook_hits / hook_attempts
    uppercut_accuracy = uppercut_hits / uppercut_attempts

    return [jab_accuracy, hook_accuracy, uppercut_accuracy]


def compute_defense_stats(data: pd.DataFrame) -> list:
    head_covered_frames = data["Defense"].iloc[0]
    body_covered_frames = data["Defense"].iloc[1]
    total_frames = data["Defense"].iloc[2]

    if (total_frames == 0):
        total_frames = 1

    concussion_opps = round((1-(head_covered_frames / total_frames))*22)
    rib_fracture_opps = round((1-(body_covered_frames / total_frames))*22)

    return [concussion_opps, rib_fracture_opps]


def compute_dodge_stats(data: pd.DataFrame) -> list:
    left_dodge = ast.literal_eval(data["Dodges"].iloc[0])[0]
    left_dodge_attempts = ast.literal_eval(data["Dodges"].iloc[0])[1]
    right_dodge = ast.literal_eval(data["Dodges"].iloc[1])[0]
    right_dodge_attempts = ast.literal_eval(data["Dodges"].iloc[1])[1]

    if (left_dodge_attempts == 0):
        left_dodge_attempts = 1
    if (right_dodge_attempts == 0):
        right_dodge_attempts = 1

    left_dodge_accuracy = left_dodge / left_dodge_attempts
    right_dodge_accuracy = right_dodge / right_dodge_attempts

    return [left_dodge_accuracy, right_dodge_accuracy]
