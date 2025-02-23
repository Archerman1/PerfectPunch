import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import io
import base64

vector_pairs = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (23, 24),
    (11, 23), (12, 24), (23, 25),
    (25, 27), (24, 26), (26, 28)
]


def plot_stance(data: pd.DataFrame) -> list:

    figures = []

    for index, row in data.iterrows():
        try:
            punch_type, coord_list = ast.literal_eval(row["Stance"])

            fig, ax = plt.subplots()
            for start, end in vector_pairs:
                if start < len(coord_list) and end < len(coord_list):
                    x = [coord_list[start][0], coord_list[end][0]]
                    y = [coord_list[start][1], coord_list[end][1]]

                    ax.plot(x, y, color="black")

            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_title(f"Stance for {punch_type}")
            ax.invert_yaxis()
            ax.grid(True)

            # buf = io.BytesIO()
            # plt.savefig(buf, format="png", bbox_inches="tight")
            # plt.close(fig)
            # buf.seek(0)
            # img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            # buf.close()

            plt.savefig(
                f"/Users/shreyagarwal/Desktop/taipy/images/stance_{index}.png", bbox_inches="tight")
            plt.close(fig)
            figures.append(
                (punch_type, f"/Users/shreyagarwal/Desktop/taipy/images/stance_{index}.png"))

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
