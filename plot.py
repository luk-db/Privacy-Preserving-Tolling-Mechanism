import json
import matplotlib.pyplot as plt
import os
import glob
import sys

def plot_macro_results(filepath, filename, remove_first=0, plot_latency=False, plot_mean_dist=False):
    """
    Reads a JSON-formatted macroscopic results file and plots the Total Latency and IPoA
    against the scaling parameter (mu). Saves the plot as a PNG.
    """
    path = os.path.join(filepath, filename)

    print(f"Plotting results for: {filename}")
    with open(path, 'r') as f:
        data = json.load(f)
    
    for network_name, results in data.items():
        mu = results.get('mu', [])
        ipoa = results.get('IPoA_mean', [])
        latency = results.get('total_latency_mean', [])
        print(f"{len(mu)} - {len(ipoa)} - {len(latency)}")

        mu = mu[remove_first:]
        ipoa = ipoa[remove_first:]
        latency = latency[remove_first:]

        no_toll_ipoa = ipoa[-1] if ipoa else None
        mu = mu[:-1]
        ipoa = ipoa[:-1]
        latency = latency[:-1]

        print([f"{mu:.2f}" for mu in mu])
        
        # Some fields might be missing depending on the exact format, safely extract
        so_latency = results.get('total_latency_SO', [None])[0]
        no_toll_latency = results.get('total_latency_no_tolling', [None])[0]
        theta_values = results.get('theta_values', [])
        avg_theta = sum(theta_values) / len(theta_values) if theta_values else None
        
        if not mu or not latency:
            print(f"Missing essential data for {network_name} in {filepath}")
            continue

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_ylabel('IPoA (Price of Anarchy)', color=color if plot_latency else 'black')  
        ax1.axhline(y=1, color='tab:green', linestyle='--', label='IPoA=1 (UE=SO)') 
        ax1.set_xlabel(r'Scaling Parameter $\mu$')
        ax1.plot(mu, ipoa, marker='x', linestyle=':', color=color, label='IPoA')
        ax1.tick_params(axis='y', labelcolor=color if plot_latency else 'black')        

        if no_toll_ipoa is not None:
            ax1.plot(1.0, no_toll_ipoa, marker='o', color=color, 
                     transform=ax1.get_yaxis_transform(), clip_on=False, 
                     label='No Tolling (\u221e)')
            
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Latency Plot
        if plot_latency:
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Total Latency', color=color)
            ax2.plot(mu, latency, marker='o', color=color, label=r'Total Latency ($\mu$-MCT)')
        
        # Baselines
            if so_latency is not None:
                ax2.axhline(y=so_latency, color='tab:green', linestyle='--', label='System Optimum (SO)')
            if no_toll_latency is not None:
                ax2.axhline(y=no_toll_latency, color='tab:blue', linestyle='--', label='No Tolling (UE)')
        
            
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper left')

        if plot_mean_dist and avg_theta is not None:
            ax2.axvline(x=avg_theta, color='tab:red', linestyle='-.', label=rf'Average $\theta$ ({avg_theta:.2f})')

        plt.title(f'Results for {network_name.replace(".json", "").replace("_1","").replace("-Example", "")}')

        fig.tight_layout()  
        
        # Save the plot
        out_filename = filename.split('_')[1]+'.png'
        out_path = os.path.join(filepath, out_filename)
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close()

def plot1(filenumber, remove_first=0, plot_latency=False, plot_mean_dist=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = [file for file in os.listdir(script_dir) if file.endswith('.txt')]
    file = [file for file in filenames if file.startswith(str(filenumber)+"_")][0]
    if os.path.exists(os.path.join(script_dir, file)):
        print(f"found file: {file}")
        plot_macro_results(script_dir, file, remove_first=remove_first, plot_latency=plot_latency, plot_mean_dist=plot_mean_dist)
    else:
        print(f"File not found: {file}")

def _plot_on_ax(ax1, filepath, filename, remove_first=0, plot_latency=False, plot_mean_dist=False):
    """Helper to plot data on a specific matplotlib ax."""
    path = os.path.join(filepath, filename)
    with open(path, 'r') as f:
        data = json.load(f)
    
    for network_name, results in data.items():
        mu = results.get('mu', [])
        ipoa = results.get('IPoA_mean', [])
        latency = results.get('total_latency_mean', [])

        mu = mu[remove_first:]
        ipoa = ipoa[remove_first:]
        latency = latency[remove_first:]

        no_toll_ipoa = ipoa[-1] if ipoa else None
        mu = mu[:-1]
        ipoa = ipoa[:-1]
        latency = latency[:-1]

        so_latency = results.get('total_latency_SO', [None])[0]
        no_toll_latency = results.get('total_latency_no_tolling', [None])[0]
        theta_values = results.get('theta_values', [])
        avg_theta = sum(theta_values) / len(theta_values) if theta_values else None
        
        if not mu or not latency:
            continue

        color = 'tab:blue'
        ax1.set_ylabel('IPoA (Price of Anarchy)', color=color if plot_latency else 'black')  
        ax1.axhline(y=1, color='tab:green', linestyle='--', label='IPoA=1 (UE=SO)') 
        ax1.plot(mu, ipoa, marker='x', linestyle=':', color=color, label='IPoA')
        ax1.tick_params(axis='y', labelcolor=color if plot_latency else 'black')        
        ax1.set_xlabel(r'Scaling Parameter $\mu$')

        if no_toll_ipoa is not None:
            ax1.plot(1.0, no_toll_ipoa, marker='o', color=color, 
                     transform=ax1.get_yaxis_transform(), clip_on=False, 
                     label='No Tolling (\u221e)')
            
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_title(f'Results for {network_name.replace(".json", "").replace("_1","").replace("-Example", "")}')

        if plot_latency:
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Total Latency', color=color)
            ax2.plot(mu, latency, marker='o', color=color, label=r'Total Latency ($\mu$-MCT)')
        
            if so_latency is not None:
                ax2.axhline(y=so_latency, color='tab:green', linestyle='--', label='System Optimum (SO)')
            if no_toll_latency is not None:
                ax2.axhline(y=no_toll_latency, color='tab:blue', linestyle='--', label='No Tolling (UE)')
        
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper left')

        if plot_mean_dist and avg_theta is not None:
            target_ax = ax2 if plot_latency else ax1
            target_ax.axvline(x=avg_theta, color='tab:red', linestyle='-.', label=rf'Average $\theta$ ({avg_theta:.2f})')

        break # Only plot the first network of the file to this ax

def plot2(numbers, remove_first=0, plot_latency=False, plot_mean_dist=False):
    if len(numbers) != 2:
        print("plot2 expects exactly 2 numbers")
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = [file for file in os.listdir(script_dir) if file.endswith('.txt')]
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, num in enumerate(numbers):
        file = [f for f in filenames if f.startswith(str(num)+"_")]
        if file:
            _plot_on_ax(axs[i], script_dir, file[0], remove_first, plot_latency, plot_mean_dist)
        else:
            print(f"File not found for number: {num}")

    fig.tight_layout()
    out_path = os.path.join(script_dir, f'Z_combined_{numbers[0]}_{numbers[1]}.png')
    plt.savefig(out_path)
    print(f"Saved combined plot to {out_path}")
    plt.close()

def plot4(numbers, remove_first=0, plot_latency=False, plot_mean_dist=False):
    if len(numbers) != 4:
        print("plot4 expects exactly 4 numbers")
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = [file for file in os.listdir(script_dir) if file.endswith('.txt')]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    for i, num in enumerate(numbers):
        file = [f for f in filenames if f.startswith(str(num)+"_")]
        if file:
            _plot_on_ax(axs[i], script_dir, file[0], remove_first, plot_latency, plot_mean_dist)
        else:
            print(f"File not found for number: {num}")

    fig.tight_layout()
    out_path = os.path.join(script_dir, f'Z_combined_{"_".join(map(str, numbers))}.png')
    plt.savefig(out_path)
    print(f"Saved combined plot to {out_path}")
    plt.close()

def _plot_effectiveness_on_ax(ax, filepath, filename, remove_first=0):
    """Helper to plot effectiveness data on a specific matplotlib ax."""
    path = os.path.join(filepath, filename)
    with open(path, 'r') as f:
        data = json.load(f)
    
    for network_name, results in data.items():
        mu = results.get('mu', [])
        latency = results.get('total_latency_mean', [])

        mu = mu[remove_first:]
        latency = latency[remove_first:]

        mu = mu[:-1]
        latency = latency[:-1]

        so_latency_list = results.get('total_latency_SO', [])
        no_toll_latency_list = results.get('total_latency_no_tolling', [])
        
        so_latency = so_latency_list[0] if so_latency_list else None
        no_toll_latency = no_toll_latency_list[0] if no_toll_latency_list else None
        
        if not mu or not latency or so_latency is None or no_toll_latency is None:
            continue
            
        denom = no_toll_latency - so_latency
        if denom == 0:
            continue

        effectiveness = [(no_toll_latency - lat) / denom for lat in latency]

        color = 'tab:orange'
        ax.set_ylabel('Effectiveness\n(No Tolling - Mean) / (No Tolling - SO)', color='black')  
        ax.axhline(y=1, color='tab:green', linestyle='--', label='Max Effectiveness (SO)') 
        ax.axhline(y=0, color='gray', linestyle='--', label='No Effectiveness (No Tolling)')
        ax.plot(mu, effectiveness, marker='o', markersize=4, linestyle='-', color=color, label='Effectiveness')
        ax.tick_params(axis='y', labelcolor='black')        
        ax.set_xlabel(r'Scaling Parameter $\mu$')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Effectiveness for {network_name.replace(".json", "").replace("_1","").replace("-Example", "")}')

        break # Only plot the first network of the file to this ax

def plot1_effectiveness(filenumber, remove_first=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = [file for file in os.listdir(script_dir) if file.endswith('.txt')]
    file = [file for file in filenames if file.startswith(str(filenumber)+"_")]
    if file and os.path.exists(os.path.join(script_dir, file[0])):
        file = file[0]
        print(f"found file: {file}")
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_effectiveness_on_ax(ax, script_dir, file, remove_first=remove_first)
        fig.tight_layout()
        out_filename = file.split('_')[1] + '_effectiveness.png'
        out_path = os.path.join(script_dir, out_filename)
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close()
    else:
        print(f"File not found: number {filenumber}")

def plot2_effectiveness(numbers, remove_first=0):
    if len(numbers) != 2:
        print("plot2_effectiveness expects exactly 2 numbers")
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = [file for file in os.listdir(script_dir) if file.endswith('.txt')]
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, num in enumerate(numbers):
        file = [f for f in filenames if f.startswith(str(num)+"_")]
        if file:
            _plot_effectiveness_on_ax(axs[i], script_dir, file[0], remove_first)
        else:
            print(f"File not found for number: {num}")

    fig.tight_layout()
    out_path = os.path.join(script_dir, f'Z_combined_effectiveness_{numbers[0]}_{numbers[1]}.png')
    plt.savefig(out_path)
    print(f"Saved combined plot to {out_path}")
    plt.close()

def plot4_effectiveness(numbers, remove_first=0):
    if len(numbers) != 4:
        print("plot4_effectiveness expects exactly 4 numbers")
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = [file for file in os.listdir(script_dir) if file.endswith('.txt')]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    for i, num in enumerate(numbers):
        file = [f for f in filenames if f.startswith(str(num)+"_")]
        if file:
            _plot_effectiveness_on_ax(axs[i], script_dir, file[0], remove_first)
        else:
            print(f"File not found for number: {num}")

    fig.tight_layout()
    out_path = os.path.join(script_dir, f'Z_combined_effectiveness_{"_".join(map(str, numbers))}.png')
    plt.savefig(out_path)
    print(f"Saved combined plot to {out_path}")
    plt.close()

def plot_eff(numbers, remove_first=0):
    if type(numbers) == int:
        plot1_effectiveness(numbers, remove_first)
    elif type(numbers) == list:
        if len(numbers) == 2:
            plot2_effectiveness(numbers, remove_first)
        elif len(numbers) == 4:
            plot4_effectiveness(numbers, remove_first)
        else:
            print("Unsupported number of plots. Use 2 or 4.")
    else:
        print("Unsupported input type for numbers. Use int or list of ints.")

def plot(numbers, remove_first=0, plot_latency=False, plot_mean_dist=False):
    if isinstance(numbers, int):
        plot1(numbers, remove_first, plot_latency, plot_mean_dist)
    elif isinstance(numbers, list):
        if len(numbers) == 2:
            plot2(numbers, remove_first, plot_latency, plot_mean_dist)
        elif len(numbers) == 4:
            plot4(numbers, remove_first, plot_latency, plot_mean_dist)
        else:
            print("Unsupported number of plots. Use 2 or 4.")
    else:
        print("Unsupported input type for numbers. Use int or list of ints.")

if __name__ == "__main__":
    
    plot_all = True

    if plot_all:
        for i in [1,2,3,4,5,6,7,9,12]:
            plot1(i, remove_first=4)

        for i in [8,10,11]:
            plot1(i, remove_first=0)

