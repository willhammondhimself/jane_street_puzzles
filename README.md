# Robot Javelin Puzzle Simulation

This repository contains a Python-based simulation and analysis of the Jane Street "Robot Javelin" puzzle. It uses Monte Carlo methods to determine the optimal strategies for a game with asymmetric information.

## The Puzzle

The core of the puzzle involves a two-player game where each player can either keep their first random score (from `[0,1]`) or reroll once. The twist is that one player, "Spears," gets a single bit of information about their opponent's first throw, creating an information asymmetry. We are tasked with finding the best response for our robot, "Java-lin," to maximize its win probability.

## The Solution

The full analysis, code, and step-by-step methodology are detailed in the Jupyter Notebook:

**[Robot_Javelin_Puzzle.ipynb](./Robot_Javelin_Puzzle.ipynb)**

The notebook covers:
1.  The analytical solution for the base symmetric game (Nash Equilibrium).
2.  A step-by-step simulation of the asymmetric game:
    -   Finding Spears' optimal strategy based on their information leak.
    -   Finding Java-lin's best-response strategy.
3.  A final, high-precision simulation to calculate the final win probability.

## How to Run

You can explore the notebook directly on GitHub or run the simulation yourself.

### Prerequisites
- Python 3
- NumPy
- Jupyter Notebook (for viewing the `.ipynb` file)

### Running the Script

The raw Python script is also included. To run it from your terminal:

1.  **Install dependencies**:
    ```bash
    pip install numpy
    ```
2.  **Run the simulation**:
    ```bash
    python equilibrium_simulation.py
    ```
    *Note: The script performs several intensive simulations and may take a few minutes to complete.*
