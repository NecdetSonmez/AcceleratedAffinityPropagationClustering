import matplotlib.pyplot as plt
import random


def plot_data(filename, subplot, label_colors):
    x = []
    y = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
            label = int(parts[2])
            labels.append(label)

            if label not in label_colors:
                label_colors[label] = (random.random(), random.random(), random.random())

    colors = [label_colors[label] for label in labels]

    ax = plt.subplot(1, 3, subplot)
    ax.scatter(x, y, color=colors, edgecolors='k', alpha=0.75)
    ax.set_title(f'Data from {filename}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')


plt.figure(figsize=(18, 6))
global_label_colors = {}
files = ["CpuClusters.txt", "GpuV1Clusters.txt", "GpuV2Clusters.txt"]

for i, file in enumerate(files):
    plot_data(file, i + 1, global_label_colors)

plt.show()


