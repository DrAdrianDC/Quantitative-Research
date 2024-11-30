# This script handles the visualization of the daily return, transition matrix, Markov chain simulated states trends.

# Import libraries

import matplotlib.pyplot as plt
import seaborn as sns

# Define functions

# Stock Daily Returns
def plot_daily_returns(data):
    """
    Plot the daily returns of the stock with colors based on states.
    
    Parameters:
        data (pd.DataFrame): Stock data containing 'Daily Return' and 'State'.
    """
    state_colors = {"Up": "green", "Stable": "blue", "Down": "red"}
    plt.figure(1, figsize=(10, 6))
    
    for state, color in state_colors.items():
        state_data = data[data['State'] == state]
        plt.scatter(state_data.index, state_data['Daily Return'], label=state, color=color, s=10)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("Stock Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    # Save the figure
    plt.savefig("stock_daily_returns.png", dpi=300, bbox_inches='tight')

# Markov Chain Transition Matrix 
def plot_transition_matrix(transition_matrix):
    """
    Plot the Markov Chain transition matrix as a heatmap.
    
    Parameters:
        transition_matrix (pd.DataFrame): Transition matrix.
    """
    plt.figure(2, figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Transition Matrix Heatmap")
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    # Save the figure
    plt.savefig("markov_chain_transition_matrix_heatmap.png", dpi=300, bbox_inches='tight')

# Simulated Markov Chain states
def plot_simulated_states(simulated_states):
    """
    Plot the simulated Markov Chain states.
    
    Parameters:
        simulated_states (list): List of simulated states.
    """
    state_order = ["Down", "Stable", "Up"]
    state_numeric = {"Up": 2, "Stable": 1, "Down": 0}
    numeric_states = [state_numeric[state] for state in simulated_states]

    plt.figure(3, figsize=(10, 6))
    plt.plot(numeric_states, marker='o', linestyle='-', color='b', alpha=0.7)
    plt.title("Simulated Markov Chain for Stock Movement Over 30 Days")
    plt.xlabel("Day")
    plt.ylabel("State")
    plt.yticks([0, 1, 2], state_order)
    plt.ylim(-0.5, 2.5)
    plt.grid(True)
    # Save the figure
    plt.savefig("simulated_markov_chain_stock_movement.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    from fetch_data import fetch_stock_data
    from markov_model import classify_state, build_transition_matrix, simulate_markov_chain

    # Fetch stock data
    data = fetch_stock_data("AAPL", "2024-01-01", "2024-11-28")
    data['State'] = data['Daily Return'].apply(classify_state)

    # Plot daily returns
    plot_daily_returns(data)
    
    # Build transition matrix and plot heatmap
    transition_matrix = build_transition_matrix(data['State'])
    plot_transition_matrix(transition_matrix)

    # Simulate the Markov chain for 30 days
    current_state = data['State'].iloc[-1]
    simulated_states = simulate_markov_chain(transition_matrix, current_state, days=30)

    # Plot simulated states
    plot_simulated_states(simulated_states)
    
    # Show all figures
    plt.show()
