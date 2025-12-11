
import numpy as np

def simulate_game(t, n_sims=1_000_000, eps=0.001):
    """
    Simulates n_sims independent games with both players using cutoff t.

    This function estimates the overall expected payoff and checks the indifference
    condition at the cutoff t.

    Args:
        t (float): The cutoff value for the keep/discard strategy, in [0, 1].
        n_sims (int): The number of games to simulate.
        eps (float): A small value to define a tiny band around t for checking
                     the indifference condition.

    Returns:
        tuple: A tuple containing:
            - expected_payoff (float): Player 1's average payoff.
            - win_prob_keep (float): Empirical win probability for Player 1 if they
                                     keep a first draw X1 in the band [t-eps, t+eps].
            - win_prob_reroll (float): Empirical win probability for Player 1 if they
                                       discard and reroll a first draw X1 in that same band.
    """
    # Generate all random numbers needed for the simulation upfront
    # X1, X2 are the initial draws for Player 1 and Player 2
    X1 = np.random.rand(n_sims)
    X2 = np.random.rand(n_sims)

    # Y1, Y2 are the potential second draws if a player decides to discard
    Y1 = np.random.rand(n_sims)
    Y2 = np.random.rand(n_sims)

    # Determine final values based on the cutoff strategy
    # If the first draw is less than t, take the second draw; otherwise, keep the first.
    V1 = np.where(X1 < t, Y1, X1)
    V2 = np.where(X2 < t, Y2, X2)

    # Calculate Player 1's payoffs for each game (1 for win, 0.5 for tie, 0 for loss)
    payoffs = (V1 > V2) + 0.5 * (V1 == V2)
    expected_payoff = np.mean(payoffs)

    # --- Indifference Condition Check ---
    # We focus on games where Player 1's first draw X1 is very close to t.
    indifference_indices = np.abs(X1 - t) < eps
    n_indiff = np.sum(indifference_indices)

    if n_indiff == 0:
        # If no draws fall in the band, we can't estimate the conditional probabilities.
        # This might happen if eps is too small or n_sims is low.
        return expected_payoff, np.nan, np.nan

    # Get the opponent's final values for this subset of games
    V2_at_indifference = V2[indifference_indices]

    # (a) Estimate win rate if Player 1 chooses to KEEP their first draw X1
    # In this scenario, Player 1's value is X1 from the initial draw.
    X1_at_indifference = X1[indifference_indices]
    payoffs_keep = (X1_at_indifference > V2_at_indifference) + 0.5 * (X1_at_indifference == V2_at_indifference)
    win_prob_keep = np.mean(payoffs_keep)

    # (b) Estimate win rate if Player 1 chooses to DISCARD and REROLL
    # In this scenario, Player 1 gets a new random value. We simulate these new draws.
    Y1_reroll_at_indifference = np.random.rand(n_indiff)
    payoffs_reroll = (Y1_reroll_at_indifference > V2_at_indifference) + 0.5 * (Y1_reroll_at_indifference == V2_at_indifference)
    win_prob_reroll = np.mean(payoffs_reroll)

    return expected_payoff, win_prob_keep, win_prob_reroll


def search_equilibrium_ts(T_grid, n_sims_per_t):
    """
    Searches for the symmetric Nash Equilibrium cutoff t over a grid of candidates.

    For each t, it calculates the difference between the win probability from keeping
    a draw near t versus rerolling. The equilibrium t is where this difference is zero.

    Args:
        T_grid (np.array): An array of candidate t values to test.
        n_sims_per_t (int): The number of simulations to run for each t value.

    Returns:
        tuple: A tuple containing:
            - best_t (float): The t value from the grid where the difference is closest to 0.
            - best_diff (float): The difference value at best_t.
            - final_payoff (float): The estimated expected payoff at best_t.
    """
    diffs = []
    print(f"Simulating for {len(T_grid)} values of t...")

    for i, t in enumerate(T_grid):
        _, win_prob_keep, win_prob_reroll = simulate_game(t, n_sims=n_sims_per_t)

        if np.isnan(win_prob_keep):
            diff = np.nan
        else:
            diff = win_prob_keep - win_prob_reroll
        diffs.append(diff)
        print(f"  t={t:.3f}, diff={diff:+.4f}")

    diffs = np.array(diffs)

    # Find the t where the absolute difference is minimized, ignoring any NaNs
    valid_indices = ~np.isnan(diffs)
    if not np.any(valid_indices):
        print("Warning: Could not find a valid equilibrium point. All simulations failed.")
        return np.nan, np.nan, np.nan

    best_idx_in_valid = np.argmin(np.abs(diffs[valid_indices]))
    # Map back to the original index in T_grid
    original_idx = np.where(valid_indices)[0][best_idx_in_valid]

    best_t = T_grid[original_idx]
    best_diff = diffs[original_idx]

    # Run a final, larger simulation at the best t for a more accurate payoff estimate
    print(f"\nFound best t = {best_t:.4f}. Running a more precise simulation...")
    final_payoff, _, _ = simulate_game(best_t, n_sims=n_sims_per_t * 5)

    return best_t, best_diff, final_payoff


def spears_vs_java(t_java, d, t_low, t_high, n_sims=100_000):
    """
    Simulates a game between Java-lin and Spears with an information leak.

    Args:
        t_java (float): Java-lin's single cutoff.
        d (float): Spears' threshold for the information leak.
        t_low (float): Spears' cutoff if Java-lin's first throw is < d.
        t_high (float): Spears' cutoff if Java-lin's first throw is >= d.
        n_sims (int): Number of games to simulate.

    Returns:
        float: Java-lin's estimated win probability.
    """
    # Initial random draws for both players
    X_J = np.random.rand(n_sims)  # Java-lin's first throw
    X_S = np.random.rand(n_sims)  # Spears' first throw

    # Potential second throws
    Y_J = np.random.rand(n_sims)
    Y_S = np.random.rand(n_sims)

    # Java-lin's final value (V_J) is determined by their cutoff t_java
    V_J = np.where(X_J >= t_java, X_J, Y_J)

    # Spears' strategy depends on the bit b = I(X_J >= d)
    b = (X_J >= d)
    
    # Determine which cutoff Spears uses based on the bit b
    spears_cutoff = np.where(b, t_high, t_low)
    
    # Spears' final value (V_S) is determined by their cutoff
    V_S = np.where(X_S >= spears_cutoff, X_S, Y_S)

    # Calculate Java-lin's win probability
    java_lin_payoffs = (V_J > V_S) + 0.5 * (V_J == V_S)
    return np.mean(java_lin_payoffs)


def search_spears_strategy(t_java, n_sims_per_point=50_000):
    """
    Searches for Spears' best strategy (d, t_low, t_high) to minimize Java-lin's win rate.
    Uses a refined grid for higher precision.
    """
    print("\n--- Searching for Spears' Best Strategy (Fine Grid) ---")
    
    # Fine-grained grids centered around previously found optima
    d_grid = np.linspace(0.58, 0.62, 21)  # d around 0.60
    # From analysis, t_low should be exactly 0.5 if d < t_java
    t_low_grid = np.linspace(0.48, 0.52, 21) # t_low around 0.50
    t_high_grid = np.linspace(0.63, 0.67, 21) # t_high around 0.65
    
    min_win_rate = 1.0
    best_params = {}
    
    total_iterations = len(d_grid) * len(t_low_grid) * len(t_high_grid)
    current_iteration = 0

    print(f"Searching over {total_iterations} parameter combinations...")

    for d in d_grid:
        # Optimization: if d is less than t_java, t_low must be 0.5
        # We can skip the loop and just use the optimal value.
        # This holds true for our search since max(d_grid) < t_java.
        active_t_low_grid = [0.5] if d < t_java else t_low_grid

        for t_low in active_t_low_grid:
            for t_high in t_high_grid:
                current_iteration += 1
                if current_iteration % 1000 == 0:
                    print(f"  ...completed {current_iteration}/{total_iterations} iterations.")

                win_rate = spears_vs_java(t_java, d, t_low, t_high, n_sims=n_sims_per_point)
                
                if win_rate < min_win_rate:
                    min_win_rate = win_rate
                    best_params = {'d': d, 't_low': t_low, 't_high': t_high}
    
    print("Search complete.")
    print(f"Found best parameters for Spears: d={best_params['d']:.4f}, t_low={best_params['t_low']:.4f}, t_high={best_params['t_high']:.4f}")
    print(f"Java-lin's win rate minimized to: {min_win_rate:.8f}")
    
    return best_params, min_win_rate

def search_javalin_response(spears_params, n_sims_per_point=200_000):
    """
    With Spears' strategy fixed, finds the best cutoff for Java-lin to maximize win rate.
    """
    print("\n--- Searching for Java-lin's Best Response (Fine Grid) ---")
    
    # Fine-grained grid around the previously found optimum of 0.60
    t_java_grid = np.linspace(0.58, 0.62, 81)
    max_win_rate = 0.0
    best_t_java = -1

    d = spears_params['d']
    t_low = spears_params['t_low']
    t_high = spears_params['t_high']

    print(f"Searching over {len(t_java_grid)} Java-lin cutoffs...")

    for t_java in t_java_grid:
        win_rate = spears_vs_java(t_java, d, t_low, t_high, n_sims=n_sims_per_point)
        if win_rate > max_win_rate:
            max_win_rate = win_rate
            best_t_java = t_java
    
    print("Search complete.")
    print(f"Found best adjusted t_java: {best_t_java:.4f} (Win rate: {max_win_rate:.8f})")

    return best_t_java, max_win_rate


if __name__ == '__main__':
    # --- Part 1: Symmetric Game (from previous query) ---
    # We skip running this and use the known theoretical value for t_star.
    # --- Part 2: Robot Javelin Puzzle ---

    print("\n\n--- Starting Robot Javelin Puzzle Simulation ---")

    # Use the theoretical t_star for Java-lin's initial strategy
    t_star = (np.sqrt(5) - 1) / 2
    print(f"Using initial t_java = t_star = {t_star:.6f}")

    # 1. Find Spears' best strategy to minimize Java-lin's win rate
    best_spears_params, min_win_rate = search_spears_strategy(t_star)

    # 2. Find Java-lin's best response to that fixed Spears strategy
    best_t_java, final_win_rate = search_javalin_response(best_spears_params)
    
    # 3. Run a final high-precision simulation with the optimal parameters
    print("\n--- Final High-Precision Simulation ---")
    print(f"Java-lin's adjusted cutoff: {best_t_java:.4f}")
    print(f"Spears' parameters: d={best_spears_params['d']:.4f}, t_low={best_spears_params['t_low']:.4f}, t_high={best_spears_params['t_high']:.4f}")
    
    final_win_rate_precise = spears_vs_java(
        best_t_java,
        best_spears_params['d'],
        best_spears_params['t_low'],
        best_spears_params['t_high'],
        n_sims=100_000_000  # Very high number of simulations for precision
    )

    # 4. Print the final puzzle answer
    print("\n--- Robot Javelin Puzzle Results ---")
    print(f"Best parameters found for Spears: d={best_spears_params['d']:.4f}, t_low={best_spears_params['t_low']:.4f}, t_high={best_spears_params['t_high']:.4f}")
    print(f"Java-lin's optimal adjusted cutoff: {best_t_java:.4f}")
    print(f"Final win probability for Java-lin: {final_win_rate_precise:.10f}")
