"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import visualize
import neat
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df_labeled = pd.read_csv("labeled_experiments.csv")  # Load the labeled DataFrame
# Filter for class 0 and class 2
# Path to folder with your CSV experiment files
# Filter for class 0 and class 2
# Filter for class 0 and class 2
df_filtered = df_labeled[df_labeled['label'].isin([0, 1])]

#print(df_filtered)
X_segments = []
y_labels = []

# Group by experiment and label
for (experiment, label), group in df_filtered.groupby(['experiment', 'label']):
    ch1_values = group["CH1"].values
    #print(f"Processing experiment: {experiment}, label: {label}, number of segments: {len(ch1_values)}")

    ## Skip segments that are too short or inconsistent
    # if len(ch1_values) != 1800:
    #    continue

    X_segments.append(ch1_values[:1799])
    y_labels.append(label)

# Convert to NumPy arrays
X = np.array(X_segments)  # shape: (n_segments, 1800)
X_padded = np.pad(X, ((0, 0), (0, 1)), mode='edge')  # Now (330, 1800)
X_downsampled = X_padded.reshape(330, 300, 6).mean(axis=2)  # (330, 300)
X_normalized = (X_downsampled - X_downsampled.mean(axis=1, keepdims=True)) / X_downsampled.std(axis=1, keepdims=True)
X = X_normalized  # Now (330, 300)
y = np.array(y_labels)  # shape: (n_segments,)

# 2-input XOR inputs and expected outputs.
#xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
#xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
# Load the linearly separable dataset
#df = pd.read_csv("nonlinear_separable_dataset.csv")
#X = df[["f1", "f2"]].values
#print(X)
#y = df["label"].values
# Preview to verify structure

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        correct = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for xi, yi in zip(X, y):
            output = net.activate(xi)[0]
            prediction = 1 if output > 0.5 else 0
            correct += int(prediction == yi)

        genome.fitness = correct/330  # max = 100


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(X, y):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

"""
# Create meshgrid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([winner_net.activate(xi)[0] for xi in grid])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=["blue", "red"])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors="k")
plt.title("Decision Boundary of Best NEAT Individual")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
"""