import os
import pandas as pd
import numpy as np

include_P3 = [

    "jun7_0", "jun7_1", "jun7_2", "jun7_3", "jun7_4",
    "jun8_0", "jun8_1", "jun8_2", "jun8_3", "jun8_4",
    "jun9_0", "jun9_1", "jun9_2", "jun9_3", "jun9_4",  # messy
    "jun10_0", "jun10_1", "jun10_3", "jun10_4",
    "jun13_2", "jun13_3", "jun13_4",
    "jun14_0", "jun14_2", "jun14_3", "jun14_4",
    "jun15_0", "jun15_1", "jun15_3", "jun15_4",
    "jun16_0", "jun16_2", "jun16_3", "jun16_4",
    "jun17_0", "jun17_1", "jun17_3", "jun17_4",
    "jun24_0", "jun24_1", "jun24_2", "jun24_3", "jun24_4",
    "jun25_0", "jun25_1", "jun25_2", "jun25_3", "jun25_4",
    "jun26_0", "jun26_3", "jun26_4",

    "jul12_1", "jul12_2", "jul12_3", "jul12_4",
    "jul13_0", "jul13_2", "jul13_3", "jul13_4",

    "jul16_0", "jul16_1", "jul16_2", "jul16_3", "jul16_4",
    "jul20_0", "jul20_1", "jul20_2", "jul20_3", "jul20_4",
    "jul21_0", "jul21_2", "jul21_3", "jul21_4",
    "jul22_0", "jul22_1", "jul22_2", "jul22_3", "jul22_4",
    "jul24_0", "jul24_1", "jul24_2", "jul24_3", "jul24_4",
    "jul25_0", "jul25_1", "jul25_2", "jul25_3", "jul25_4",
    "jul26_0", "jul26_1", "jul26_2", "jul26_3", "jul26_4",
]

include_P5 = [
    "jun7_0", "jun7_1", "jun7_2", "jun7_3", "jun7_4",
    "jun8_0", "jun8_1", "jun8_2", "jun8_3", "jun8_4",
    "jun9_0", "jun9_1", "jun9_3", "jun9_4",
    "jun10_0", "jun10_1", "jun10_2", "jun10_3", "jun10_4",

    "jun15_0", "jun15_1", "jun15_2", "jun15_3", "jun15_4",
    "jun16_0", "jun16_1", "jun16_2", "jun16_3", "jun16_4",
    "jun24_2", "jun24_3", "jun24_4",  # noisy
    "jun25_0", "jun25_1", "jun25_2", "jun25_3", "jun25_4",
    "jun26_0", "jun26_1", "jun26_2", "jun26_3", "jun26_4",

    "jul10_1", "jun26_2",
    "jul11_4",
    "jul12_0", "jul12_1", "jul12_2", "jul12_3", "jul12_4",
    "jul13_1", "jul13_2", "jul13_3", "jul13_4",
    "jul21_3", "jul21_4",
    "jul22_0", "jul22_1", "jul22_2", "jul22_3", "jul22_4",
    "jul25_0", "jul25_1", "jul25_2", "jul25_3", "jul25_4",
    "jul26_0", "jul26_1", "jul26_2", "jul26_3", "jul26_4",
    "jul27_3", "jul27_4",
]


def label_experiments(data_dir, output_csv=None):
    """
    Reads all CSV files from a directory, labels rows by 30-min intervals,
    and returns a merged labeled DataFrame.
    """
    all_data = []

    for filename in os.listdir(data_dir):

        if not filename.endswith(".csv"):
            continue

        basename = filename.replace(".csv", "").lower()

        #print(f"Checking {basename}...")
        # Check for P3
        if "p3" in basename:
            if basename[2:] not in include_P3:
                print(f"Skipping {filename} (not in include_P3)")
                continue

        # Check for P5
        elif "p5" in basename:
            if basename[2:] not in include_P5:
                print(f"Skipping {filename} (not in include_P5)")
                continue

        # Otherwise skip
        else:
            print(f"Skipping {filename} (not recognized as P3 or P5)")
            continue
        print(f"Processing {filename}...")
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)

            # Convert tstamp to datetime
            df['tstamp'] = pd.to_datetime(df['tstamp'])

            # Get start time of the experiment
            start_time = df['tstamp'].min()

            # Calculate elapsed time in minutes
            df['minutes'] = (df['tstamp'] - start_time).dt.total_seconds() / 60

            # Assign label based on 30-minute intervals
            df['label'] = pd.cut(
                df['minutes'],
                bins=[-1, 30, 60, 90],
                labels=[0, 1, 2]
            ).astype(int)

            df['experiment'] = filename  # track source file
            all_data.append(df)

    # Combine all labeled experiments
    merged = pd.concat(all_data, ignore_index=True)
    print(merged)
    # Save if output path is provided
    if output_csv:
        merged.to_csv(output_csv, index=False)

    return merged

def read():
    # Path to folder with your CSV experiment files
    #data_folder = "path/to/your/experiment_csvs"
    df_labeled = pd.read_csv("labeled_experiments.csv")  # Load the labeled DataFrame
    # Filter for class 0 and class 2
    # Path to folder with your CSV experiment files
    # Filter for class 0 and class 2
    # Filter for class 0 and class 2
    df_filtered = df_labeled[df_labeled['label'].isin([0, 1])]

    print(df_filtered)
    X_segments = []
    y_labels = []

    # Group by experiment and label
    for (experiment, label), group in df_filtered.groupby(['experiment', 'label']):
        ch1_values = group["CH1"].values
        print(f"Processing experiment: {experiment}, label: {label}, number of segments: {len(ch1_values)}")

        ## Skip segments that are too short or inconsistent
        #if len(ch1_values) != 1800:
        #    continue

        X_segments.append(ch1_values[:1799])
        y_labels.append(label)

    # Convert to NumPy arrays
    X = np.array(X_segments)  # shape: (n_segments, 1800)
    y = np.array(y_labels)  # shape: (n_segments,)

    print(X)
    print(y)
    print(np.shape(X))
    print(np.shape(y))


if __name__ == "__main__":
    data_directory = "trials/"  # Directory containing CSV files
    output_file = "labeled_experiments.csv"  # Output file path

    #labeled_data = label_experiments(data_directory, output_file)
    #print(labeled_data.head())  # Display first few rows of the labeled DataFrame

    read()