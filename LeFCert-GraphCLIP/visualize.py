import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import pickle
import os.path
import os
from matplotlib.pyplot import MultipleLocator

rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 27
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rcParams['legend.title_fontsize'] = MEDIUM_SIZE
palette = ["#ffaaa5","#adcbe3", "#5779C1"]
font = {'family': 'serif',
        'color': 'darkblue',
        'weight': 'normal',
        'size': 5}

#-----string mapping----


def plot_certified_accuracy(poisoning_sizes, certified_accuracies, model_type, dataset_type, json_file_path):
    """
    Plot certified accuracy vs poisoning size and save the figure to the same directory as the JSON file

    Args:
        poisoning_sizes (list): List of poisoning sizes (T values)
        certified_accuracies (list): List of corresponding certified accuracies
        model_type (str): Name of the model used
        dataset_type (str): Name of the dataset used
        json_file_path (str): Path to the JSON file (plot will be saved in same directory)
    """
    plt.figure(figsize=(8, 5))
    plt.plot(poisoning_sizes, certified_accuracies, marker='o', linestyle='-', color='b')
    plt.title(f"Certified Accuracy vs Poisoning Size\nModel: {model_type}, Dataset: {dataset_type}")
    plt.xlabel("Poisoning Size (T)")
    plt.ylabel("Certified Accuracy")
    plt.xticks(poisoning_sizes)
    plt.ylim(0, 1.0)
    plt.grid(True)

    # Generate plot path in the same directory as JSON file
    dir_name = os.path.dirname(json_file_path) or '.'  # Use current dir if json_file_path has no dir
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    plot_filename = os.path.join(dir_name, f"{base_name}.png")

    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename}")


def merged_certified_curve(df_combine, output_dir, options):
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{options.dataset_type},$(C,K,M$={options.C},{options.K},{options.M})")

    # Define a color mapping for each model
    model_colors = {
        'LeFCert':'#92a8d1',  # Red
        'FCert':  '#ff6f69',   # Blue
        'DPA': '#ffcc5c',      # Green
        'KNN': '#88d8b0'       # Yellow
    }

    # Filter the palette to only include models present in the data
    unique_models = df_combine['certify_model'].unique()
    palette = [model_colors[model] for model in unique_models if model in model_colors]

    # Plot with specified colors and model order
    sns.lineplot(
        x='poisoning size', 
        y="certified accuracy",
        hue='certify_model',
        hue_order=['LeFCert', 'FCert', 'DPA', 'KNN'],  # Ensures correct order
        palette=palette,
        data=df_combine,
        linewidth=3
    )

    plt.xlabel('T', fontsize=28)
    plt.legend(loc='lower right', fancybox=True, framealpha=0.5)
    plt.savefig(output_dir + f'_merged_certaccuracy.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}_merged_certaccuracy.pdf')
    plt.show()

    
'''
def merged_certified_curve(df_combine,output_dir,options):
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{options.dataset_type},$(C,K,M$={options.C},{options.K},{options.M})")
    sns.lineplot(x='poisoning size', y="certified accuracy",hue='certify_model', palette=['#ff6f69','#8f8787', '#ffcc5c','#88d8b0','#92a8d1','#aa96da'], data=df_combine,linewidth = 3)
    plt.xlabel('T', fontsize=28)
    plt.legend(loc='lower right', fancybox=True,framealpha=0.5)  #
    plt.savefig(output_dir + f'/merged_certaccuracy.pdf', dpi=300, bbox_inches='tight')
    print(f'Save result to {output_dir}/merged_certaccuracy.pdf')
    plt.show()
'''
