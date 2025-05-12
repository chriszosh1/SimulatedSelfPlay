import pandas as pd
import matplotlib.pyplot as plt


def plot_choices_over_time(filename, save_fig=False, output_path='choices_plot.png', yrange=(0,1)):
    """
    Creates a line plot showing choices over time with per-run averages and overall average.
    
    Args:
        filename (str): Path to CSV file containing the data
        save_fig (bool): Whether to save the figure
        output_path (str): Path where to save the figure if save_fig is True
        yrange (tuple): Range for y-axis (min, max)
    """
    # Read data
    df = pd.read_csv(filename)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Calculate and plot average for each run
    for run in df['run'].unique():
        run_data = df[df['run'] == run]
        # Calculate mean across all agents for this run
        run_mean = run_data.groupby('period')['choice'].mean()
        plt.plot(run_mean.index, run_mean.values,
                color='gray', alpha=0.3, linewidth=1)
    
    # Calculate and plot overall average across all runs and agents
    overall_mean = df.groupby('period')['choice'].mean()
    plt.plot(overall_mean.index, overall_mean.values,
            color='red', alpha=1, linewidth=2, label='Overall Average')
    
    # Customize plot
    plt.xlabel('Period')
    plt.ylabel('Choice')
    plt.ylim(yrange[0] - 0.05, yrange[1] + 0.05)
    plt.title('Agent Choices Over Time')
    plt.grid(True)
    plt.legend()
    
    if save_fig:
        plt.savefig(output_path)
    plt.show()