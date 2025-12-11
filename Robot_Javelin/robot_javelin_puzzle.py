import numpy as np

def simulate_game(t, n_sims=100000, eps=1e-4):
    """Simulates the base game to find the equilibrium cutoff t."""
    x1 = np.random.rand(n_sims)
    x2 = np.random.rand(n_sims)

    reroll1 = x1 < t
    final1 = np.where(reroll1, np.random.rand(n_sims), x1)

    reroll2 = x2 < t
    final2 = np.where(reroll2, np.random.rand(n_sims), x2)

    p1_wins = (final1 > final2) + 0.5 * (final1 == final2)

    indifference_band = (x1 >= t - eps) & (x1 < t + eps)
    
    if np.sum(indifference_band) == 0:
        return np.mean(p1_wins), np.nan

    p2_reroll_if_kept = x2 < t
    p2_final_if_kept = np.where(p2_reroll_if_kept, np.random.rand(np.sum(indifference_band)), x2[indifference_band])
    win_prob_if_keep = np.mean((x1[indifference_band] > p2_final_if_kept) + 0.5 * (x1[indifference_band] == p2_final_if_kept))

    p1_reroll_val = np.random.rand(np.sum(indifference_band))
    p2_reroll_if_reroll = x2 < t
    p2_final_if_reroll = np.where(p2_reroll_if_reroll, np.random.rand(np.sum(indifference_band)), x2[indifference_band])
    win_prob_if_reroll = np.mean((p1_reroll_val > p2_final_if_reroll) + 0.5 * (p1_reroll_val == p2_final_if_reroll))

    diff = win_prob_if_keep - win_prob_if_reroll
    return np.mean(p1_wins), diff

def search_equilibrium_t(T_grid, n_sims_per_t):
    """Searches for the equilibrium t over a grid."""
    diffs = []
    for t in T_grid:
        _, diff = simulate_game(t, n_sims_per_t)
        diffs.append(diff)
    
    diffs = np.array(diffs)
    best_t_idx = np.nanargmin(np.abs(diffs))
    best_t = T_grid[best_t_idx]
    diff_at_best_t = diffs[best_t_idx]
    return best_t, diff_at_best_t

def spears_vs_java(t_java, d, t_low, t_high, n_sims=100000):
    """Simulates the game with Spears' information advantage."""
    x_j = np.random.rand(n_sims)
    x_s = np.random.rand(n_sims)

    reroll_j = x_j < t_java
    final_j = np.where(reroll_j, np.random.rand(n_sims), x_j)

    b = x_j >= d

    spears_cutoff = np.where(b, t_high, t_low)
    reroll_s = x_s < spears_cutoff
    final_s = np.where(reroll_s, np.random.rand(n_sims), x_s)

    java_lin_wins = (final_j > final_s) + 0.5 * (final_j == final_s)
    return np.mean(java_lin_wins)

def main():
    """Main function to run the full simulation and print results."""
    print("--- Part 1: Finding the Nash Equilibrium ---")
    t_star_analytical = (1 + np.sqrt(3)) / 4
    print(f"Analytical equilibrium t*: {t_star_analytical:.4f}")

    T_grid = np.linspace(0.6, 0.7, 51)
    t_star_sim, diff_t_star = search_equilibrium_t(T_grid, n_sims_per_t=100000)
    print(f"Simulated equilibrium t*: {t_star_sim:.4f}")
    print(f"Win prob difference at simulated t*: {diff_t_star:.6f}\n")

    print("--- Part 2: Finding Spears' Optimal Strategy ---")
    d_grid = np.linspace(0.1, 0.9, 9)
    t_low_grid = np.linspace(0.1, 0.9, 9)
    t_high_grid = np.linspace(0.1, 0.9, 9)

    best_spears_params = {}
    min_java_win_rate = 1.0
    t_java_nash = t_star_analytical

    for d in d_grid:
        for t_low in t_low_grid:
            for t_high in t_high_grid:
                win_rate = spears_vs_java(t_java_nash, d, t_low, t_high, n_sims=10000)
                if win_rate < min_java_win_rate:
                    min_java_win_rate = win_rate
                    best_spears_params = {'d': d, 't_low': t_low, 't_high': t_high}
    
    print(f"Spears' best parameters (d, t_low, t_high): {best_spears_params}")
    print(f"Java-lin's win rate against this (minimized): {min_java_win_rate:.6f}\n")

    print("--- Part 3: Finding Java-lin's Counter-Strategy ---")
    t_java_grid = np.linspace(0.1, 0.9, 81)
    best_java_win_rate = 0.0
    t_java_adjusted = 0.0

    d_opt = best_spears_params['d']
    t_low_opt = best_spears_params['t_low']
    t_high_opt = best_spears_params['t_high']

    for t_java in t_java_grid:
        win_rate = spears_vs_java(t_java, d_opt, t_low_opt, t_high_opt, n_sims=100000)
        if win_rate > best_java_win_rate:
            best_java_win_rate = win_rate
            t_java_adjusted = t_java
            
    print(f"Java-lin's adjusted cutoff: {t_java_adjusted:.4f}")
    print(f"Java-lin's final win probability: {best_java_win_rate:.10f}")

if __name__ == '__main__':
    main()
