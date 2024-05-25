import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_train_cg_from_csv(base_directory):
    for root, _, files in os.walk(base_directory):
        for filename in files:
            if filename == "results.csv":
                filepath = os.path.join(root, filename)
                try:
                    df = pd.read_csv(filepath)
                    
                    # Check if necessary columns exist
                    if 'Epoch' not in df.columns or 'TestCG' not in df.columns:
                        print(f"Skipping {filepath}: Required columns are missing.")
                        continue
                    
                    # Extract the TestCG column, convert string representation of lists to actual lists
                    df['TestCG'] = df['TestCG'].apply(lambda x: list(map(float, x.strip('[]').split())))
                    # df['TestCG_Accuracy'] = df['TestCG'].apply(lambda x: sum(x) / len(x) * 100) # Average accuracy
                    df['TestCG_Accuracy'] = df['TestCG'].apply(lambda x: x[-1] * 100) # Accuracy on last predicted value
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(df["Epoch"], df["TestCG_Accuracy"], marker='o', linestyle='-')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy (%)')
                    plt.title(f'TestCG Accuracy Over Epochs for {os.path.basename(root)}')
                    plt.grid(True)
                    
                    # Save the plot in the same directory as the CSV file
                    plot_filename = "results_plot.png"
                    plot_filepath = os.path.join(root, plot_filename)
                    plt.savefig(plot_filepath)
                    plt.close()
                
                except pd.errors.EmptyDataError:
                    print(f"Skipping {filepath}: Empty data error.")
                except Exception as e:
                    print(f"Skipping {filepath} due to an unexpected error: {e}")

base_directory = os.path.join(os.getcwd(), "checkpoints")
plot_train_cg_from_csv(base_directory)
