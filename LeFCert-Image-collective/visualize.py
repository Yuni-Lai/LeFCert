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
name_map = {'cifarfs': 'CIFAR-FS','cubirds200': 'CUB200-2011','tiered_imagenet': 'Tiered-ImageNet'}

def merged_certified_curve(df_combine, output_file, options):
    plt.figure(constrained_layout=True, figsize=(8, 6))
    plt.title(fr"{options.dataset_type},$(C,K,M$={options.C},{options.K},{options.M})")

    model_colors = {
        'LeFCert-C': '#fc913a',  # orange
        'LeFCert-LD': '#b4a7d6', # purple
        'LeFCert-L': '#8b9dc3' , # blue
        'LeFCert': '#e35d6a',  # red
        'FCert': '#aaaaaa',    # gray
        'DPA': '#aecc81',   # green
        'KNN': '#ffcf40'  # yellow
    }
    
    fixed_order = ['LeFCert-C','LeFCert-LD','LeFCert-L','LeFCert', 'FCert', 'DPA', 'KNN']

    existing_models = [model for model in fixed_order if model in df_combine['certify_model'].unique()]
    palette = [model_colors[model] for model in existing_models]
    
    sns.lineplot(
        x='poisoning size',
        y="certified accuracy",
        hue='certify_model',
        hue_order=fixed_order,  # order
        palette=palette,
        data=df_combine,
        linewidth=3
    )
    if 'LeFCert-LD' in existing_models or 'LeFCert-L' in existing_models:
        plt.text(
            0.97, 0.12,  # Coordinates in axes fraction (x, y)
            fr"$\sigma$: {options.sigma}"+"\n"+fr"$r$: {options.r}",  # Text to display
            fontsize=20,  # Font size
            ha='right',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            transform=plt.gca().transAxes  # Use axes coordinates
        )
    plt.xlabel('T', fontsize=28)
    plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def compare_various_methods(options):
    dfs = []
    for model_name in ['LeFCert-LD', 'LeFCert-L', 'LeFCert', 'FCert', 'KNN', 'DPA']:
        try:
            if model_name in ['LeFCert-L', 'LeFCert-LD']:
                CERTIFIY_TYPE = f"{options.certify_type}_sigma{options.sigma}_r{r}_ld{options.Lambda}"
            elif model_name == 'LeFCert':
                CERTIFIY_TYPE = f"{options.certify_type}_ld{options.Lambda}"
            else:
                CERTIFIY_TYPE = options.certify_type
            out_dir = f'./output/{options.dataset_type}_{options.model_type}/{model_name}/C{options.C}_K{options.K}_M{options.M}'
            f = open(f'{out_dir}/certify_result_{CERTIFIY_TYPE}.pkl', 'rb')
            df = pickle.load(f)
            dfs.append(df)
            f.close()
        except FileNotFoundError:
            print(f"Warning: {model_name} result not found, skipping.")

    if dfs:
        options.file_path = f'./output/{options.dataset_type}_{options.model_type}/merged_certaccuracy_C{options.C}_K{options.K}_M{options.M}_sig{options.sigma}_r{r}.pdf'
        df_combine = pd.concat(dfs, ignore_index=True)
        merged_certified_curve(df_combine, options.file_path, options)
    part_of_df = df_combine[df_combine.loc[:, "poisoning size"].isin([0, 1, 3, 5, 7, 9])]
    print(f'Dataset:{options.dataset_type}, C{options.C}_K{options.K}_M{options.M}_sig{options.sigma}_r{r}')
    print(part_of_df.pivot_table(index='poisoning size', columns='certify_model', values='certified accuracy'))


def visualize_parameters(df_combine, parameter_type, parameter_values, options):
    """
    Visualize certified accuracy curves for various parameter values.

    Args:
        df_combine (pd.DataFrame): Combined DataFrame of results.
        parameter_type (str): The parameter being varied (e.g., 'Lambda').
        parameter_values (list): List of parameter values.
        options: Configuration object containing model and dataset settings.
    """
    plt.figure(constrained_layout=True, figsize=(10, 6))
    plt.title(fr"{options.dataset_type}, $(C,K,M$={options.C},{options.K},{options.M})")

    sns.lineplot(
        x='poisoning size',
        y="certified accuracy",
        hue=parameter_type,
        hue_order=parameter_values,
        palette="viridis",
        data=df_combine,
        linewidth=3
    )

    plt.xlabel('Poisoning Size')
    plt.ylabel('Certified Accuracy')
    plt.legend(title=parameter_type, loc='upper right', fancybox=True, framealpha=0.5)
    plt.savefig(f"{options.file_path}/certified_curves_{parameter_type}_C{options.C}_K{options.K}_M{options.M}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def compare_various_parameters(options, parameter_type):
    # Define the range of values for the parameter
    if parameter_type == 'Lambda':
        parameter_values = [0, 5, 10, 15, 20, 25, 30]
    else:
        raise ValueError(f"Unsupported parameter type: {parameter_type}")

    dfs = []
    for value in parameter_values:
        setattr(options, parameter_type, value)  # Update the parameter in options
        print(f"Loading results for {parameter_type}={value}")

        # Construct file path
        if options.certify_model in ['LeFCert-L', 'LeFCert-LD']:
            CERTIFIY_TYPE = f"{options.certify_type}_sigma{options.sigma}_r{options.r}_ld{value}"
        elif options.certify_model == 'LeFCert':
            CERTIFIY_TYPE = f"{options.certify_type}_ld{value}"
        else:
            CERTIFIY_TYPE = options.certify_type

        file_path = f'{options.file_path}/certify_result_{CERTIFIY_TYPE}.pkl'

        # Load results
        try:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: Results for {parameter_type}={value} not found, skipping.")

    # Combine results and save
    if dfs:
        df_combine = pd.concat(dfs, ignore_index=True)
        print(df_combine.to_string())
    else:
        print("No results found for the specified parameter values.")
    visualize_parameters(df_combine, parameter_type, parameter_values, options)

def compare_time(options):
    options.C=5
    options.K=20
    options.M = int((options.K - 1) / 2)  # default
    dfs = []
    for certify_model in ['LeFCert','LeFCert-C']:
        file_path=options.file_path + f'{options.dataset_type}_{options.model_type}/{certify_model}/C{options.C}_K{options.K}_M{options.M}'

        f = open(f'{file_path}/time_T.pkl', 'rb')
        df = pickle.load(f)
        f.close()
        df['Model'] = certify_model
        dfs.append(df)

    df_combine = pd.concat(dfs, ignore_index=True)

    plt.figure(constrained_layout=True, figsize=(10, 6))
    plt.title(fr"{options.dataset_type}, $(C,K,M$={options.C},{options.K},{options.M})")
    sns.lineplot(
        x='T',
        y="times",
        hue='Model',
        data=df_combine,
        linewidth=3,
        palette=["#8b9dc3", "#fc913a"]  # blue for LeFCert, orange for LeFCert-C
    )
    plt.xlabel('Poisoning Size (T)')
    plt.ylabel('Runtime (s)')
    plt.legend(title='Model', loc='upper left', fancybox=True, framealpha=0.5)
    plt.savefig(f"{options.file_path}/{options.dataset_type}_{options.model_type}/runtime_T_C{options.C}_K{options.K}_M{options.M}.pdf",
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import argparse
    # Model Settings=======================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',type=str,help='path to dataset',default='..' + os.sep + 'dataset')
    parser.add_argument('--iterations',type=int,default=10,
                        help='number of batches, the batch size is C')
    parser.add_argument( '--C',type=int,default=5,
                        help='number of random classes per episode for validation')
    parser.add_argument( '--K',type=int,default=10,
                        help='number of samples per class to use as support for validation')
    parser.add_argument('--num_query',type=int,default=1,
                        help='number of samples per class to use as query, default=1')
    parser.add_argument('--manual_seed',type=int,default=7,
                        help='input for the manual seeds initializations')
    parser.add_argument('--cuda',default=True,help='enables cuda')
    parser.add_argument('--certify_type',type=str, help='group or individual',default="group")
    parser.add_argument('--model_type',type=str,help='',default="CLIP")
    parser.add_argument('--certify_model', type=str, choices=['FCert','LeFCert','KNN','DPA','LeFCert-L','LeFCert-LD'], default='LeFCert')
    parser.add_argument('--metric_type',type=str,default="cosine",choices=['cosine','l2'])
    parser.add_argument( '--M',type=int,default=3,help='number of trimmed samples (K prime),default M = (K-1)/2')
    parser.add_argument( '--Lambda',type=float,default=10,help='Lambda is a hyperparameter that controls the weight of the text feature in LeFCert')
    parser.add_argument('--box_constraint', action='store_true', default=False,
                        help="employ the box constraint of the cosine similarity [0,2]")
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma for LeFCertL, the noise level for smoothing')
    parser.add_argument('--Lr', type=float, help='Lr=Lipschitz constant for the model x attack radius')
    parser.add_argument('--dataset_type',type=str,help='',default="cifarfs",choices=['cifarfs','cubirds200','tiered_imagenet'])
    parser.add_argument( '--file_path',type=str,help='',default='./output/')
    options = parser.parse_args()
    if options.certify_model in ['LeFCert-L', 'LeFCert-LD']:
        options.metric_type = "l2"  # LeFCertL only supports l2 metric
        options.r = 0.1  # default radius for the attacker constraint
        options.Lr = np.sqrt(2 / (np.pi * options.sigma ** 2)) * options.r
    if options.metric_type == "l2":
        options.Lambda = 0.4 #"l2"
    else:
        options.Lambda = 25 #"cosine"


    options.M = int((options.K-1)/2)# default
    # options.file_path = options.file_path + f'{options.dataset_type}_{options.model_type}/{options.certify_model}/C{options.C}_K{options.K}_M{options.M}'

    # compare_various_methods(options)

    # compare_various_parameters(options, 'Lambda')

    compare_time(options)




