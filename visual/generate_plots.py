import pandas as pd
import matplotlib.pyplot as plt
import os
import json

columns_to_plot = ['TestCG', 'TrainCG']

name_dict = {
    'TestCG': 'Test',
    'TrainCG': 'Train',
    'ValiCG': 'Validation'
}

def is_appl(model_id, freq):
    if model_id == "APPl":
        return "d"
    else:
        return "d"

def plot_column_accuracy(df, column_names, root, args):
    for column_name in column_names:
        if column_name not in df.columns:
            print(f"Skipping {column_name}: Required column is missing in {root}.")
            return

        # Extract the column, convert string representation of lists to actual lists
        df[column_name] = df[column_name].apply(lambda x: list(map(float, x.strip('[]').split())))
        # df['TestCG_Accuracy'] = df['TestCG'].apply(lambda x: sum(x) / len(x) * 100) # Average accuracy
        df[f'{column_name}_Accuracy'] = df[column_name].apply(lambda x: x[-1] * 100) # Accuracy on last predicted value

    plt.figure(figsize=(8, 5))
    for column_name in column_names:
        plt.plot(df["Epoch"], df[f'{column_name}_Accuracy'], marker='o', linestyle='-',  label=f'{name_dict[column_name]} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy on {args["model_id"]} dataset with {int(args["seq_step"]) * int(args["pred_len"])} {is_appl(args["model_id"], args["freq"])} predictions')
    plt.grid(True)
    plt.legend()

    # Save the plot in the same directory as the CSV file
    plot_filename = f"all_cols_results_plot.png"
    plot_filepath = os.path.join(root, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

def plot_cg_from_csv(base_directory):
    for root, _, files in os.walk(base_directory):
        args_filepath = os.path.join(root, "args")
        if os.path.exists(args_filepath):
            with open(args_filepath, 'r') as file:
                args = json.load(file)
        for filename in files:
            if filename == "results.csv":
                filepath = os.path.join(root, filename)
                try:
                    df = pd.read_csv(filepath)
                    
                    # Check if necessary columns exist
                    if 'Epoch' not in df.columns:
                        print(f"Skipping {filepath}: 'Epoch' column is missing.")
                        continue
                    
                    
                    # for column in columns_to_plot:
                    plot_column_accuracy(df, columns_to_plot, root, args)
                
                except pd.errors.EmptyDataError:
                    print(f"Skipping {filepath}: Empty data error.")
                except Exception as e:
                    print(f"Skipping {filepath} due to an unexpected error: {e}")

base_directory = os.path.join(os.getcwd(), "checkpoints")
plot_cg_from_csv(base_directory)
