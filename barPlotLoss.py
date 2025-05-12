import matplotlib.pyplot as plt
import numpy as np

def plot_relative_mse(mse_dict, filename):
    """
    Creates a bar graph showing relative MSE values normalized to the first entry.
    
    Parameters:
    mse_dict (dict): Dictionary with tags as keys and MSE values as values
    """
    # Get the first MSE value for normalization
    base_mse = list(mse_dict.values())[0]
    
    # Calculate relative MSE values
    tags = list(mse_dict.keys())
    relative_mse = [mse / base_mse * 100 for mse in mse_dict.values()]
    
    # Create equally spaced positions for bars
    positions = np.arange(len(tags))
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(positions, relative_mse)
    
    # Customize the plot
    plt.ylabel(f'Percentage Loss relative to {tags[0]}')
    plt.title('Relative Loss Comparison')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Set x-axis ticks and labels
    plt.xticks(positions, tags, rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.gcf()
    plt.savefig(filename)

# Example usage:
if __name__ == "__main__":
    # Sample data
    mse_values = {
        'Self-Play': 17.871417042663246,
        'Burned In': 18.813998560576163,
        'Uniform': 42.75663159978545,
        'Fitted Bins': 33.33942199568185
    }
    
    # Create and show the plot
    plot_relative_mse(mse_values, 'test')
