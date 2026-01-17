import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Function to parse and extract relevant data
def parse_analysis_output(filename):
    with open(filename, "r") as f:
        text = f.read()

    # Structure: results[n_value][network][mode] = {...}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Updated regex to capture n value: "### ANALYSIS network1_modified for n = 4 ###"
    blocks = re.split(r"### ANALYSIS (network\d+)_(full|modified) for n = (\d+) ###", text)

    for i in range(1, len(blocks), 4):
        network = blocks[i]
        mode = blocks[i + 1]
        n_value = blocks[i + 2]
        block = blocks[i + 3]

        # Extract interactions (first number of each line)
        interactions = list(map(int, re.findall(r"^\d+$", block, re.MULTILINE)))
        if not interactions:
            continue

        first_interaction = interactions[0]

        # Extract times as list - look for "Time for each ... analysis below:" followed by a list
        time_line_match = re.search(r"Time for each .+? analysis below:", block)
        if not time_line_match:
            continue
        
        # Find the position after this line
        start_pos = time_line_match.end()
        remaining = block[start_pos:]
        
        # Find the first '[' and match until the corresponding ']'
        bracket_start = remaining.find('[')
        if bracket_start == -1:
            continue
        
        # Count brackets to find the matching closing bracket
        bracket_count = 0
        bracket_end = bracket_start
        for i, char in enumerate(remaining[bracket_start:], start=bracket_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    bracket_end = i + 1
                    break
        
        if bracket_count != 0:
            continue  # Unmatched brackets
        
        times_str = remaining[bracket_start:bracket_end].strip()
        
        try:
            times = ast.literal_eval(times_str)
            if not isinstance(times, list):
                raise ValueError(f"Parsed value is not a list: {type(times)}")
            times_s = np.array(times)  # times are already in seconds
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse times for {network}_{mode} (n={n_value}): {e}")
            print(f"  Captured string: {times_str[:200]}...")
            continue

        # Save extracted data for each n_value/network/mode
        results[n_value][network][mode] = {
            "interaction": first_interaction,
            "times": times_s
        }

    return results

# Function to calculate statistics (average, SD, median, total)
def compute_statistics(results):
    # Structure: stats[n_value][network][mode] = {...}
    stats = defaultdict(lambda: defaultdict(dict))

    for n_value in results:
        for network in results[n_value]:
            for mode in results[n_value][network]:
                times = results[n_value][network][mode]["times"]

                stats[n_value][network][mode] = {
                    "interaction": results[n_value][network][mode]["interaction"],
                    "avg": np.mean(times),    # Average time in ms
                    "sd": np.std(times, ddof=1),   # Standard Deviation in ms
                    "median": np.median(times),    # Median time in ms
                    "total": np.sum(times),      # Total time in ms
                }

    return stats

# Function to print a summary of the extracted statistics
def print_statistics(stats):
    for n_value in sorted(stats.keys(), key=int):
        print(f"\n{'='*85}")
        print(f"Statistics for n = {n_value}")
        print(f"{'='*85}")
        print(f"{'Network':>10} | {'Mode':>9} | {'Inter':>6} | {'Avg (s)':>9} | {'SD (s)':>8} | {'Median (s)':>12} | {'Total (s)':>12}")
        print("-" * 85)

        for network in sorted(stats[n_value].keys()):
            for mode in stats[n_value][network]:
                s = stats[n_value][network][mode]
                print(f"{network:>10} | {mode:>9} | {s['interaction']:>6} | {s['avg']:>9.2f} | {s['sd']:>8.2f} | {s['median']:>12.2f} | {s['total']:>12.2f}")


# Plot: Total Time (Y-axis: total time in ms, X-axis: first interaction value)
def plot_total_time_old(stats):
    networks = list(stats.keys())

    # Extract full and modified times for each network
    full_interactions = [stats[n]["full"]["interaction"] for n in networks]
    modified_interactions = [stats[n]["modified"]["interaction"] for n in networks]

    full_total_times = [stats[n]["full"]["total"] for n in networks]
    modified_total_times = [stats[n]["modified"]["total"] for n in networks]

    plt.figure(figsize=(10, 5))

    # Plot Full and Modified Analysis for Total Time
    plt.scatter(full_interactions, full_total_times, color="blue", label="Full Analysis")
    plt.scatter(modified_interactions, modified_total_times, color="red", label="Modified Analysis")

    plt.xlabel("Interactions")
    plt.ylabel("Total Time (s)")
    plt.title("Total Time vs Interactions (Full vs Modified Analysis)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_total_time(stats, n_value=None):
    if n_value is None:
        # Plot all n values in subplots
        n_values = sorted(stats.keys(), key=int)
        fig, axes = plt.subplots(1, len(n_values), figsize=(10 * len(n_values), 5))
        if len(n_values) == 1:
            axes = [axes]
        
        for idx, n_val in enumerate(n_values):
            ax = axes[idx]
            networks = list(stats[n_val].keys())
            
            full = sorted([(stats[n_val][n]["full"]["interaction"], stats[n_val][n]["full"]["total"]) for n in networks])
            modified = sorted([(stats[n_val][n]["modified"]["interaction"], stats[n_val][n]["modified"]["total"]) for n in networks])
            
            if full:
                full_x, full_y = zip(*full)
                ax.scatter(full_x, full_y, label="Full Analysis")
                ax.plot(full_x, full_y)
            
            if modified:
                mod_x, mod_y = zip(*modified)
                ax.scatter(mod_x, mod_y, label="Modified Analysis")
                ax.plot(mod_x, mod_y)
            
            ax.set_xlabel("Interactions")
            ax.set_ylabel("Total Time (ms)")
            ax.set_title(f"Total Time vs Interactions (n = {n_val})")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        # Plot single n value
        networks = list(stats[n_value].keys())
        
        full = sorted([(stats[n_value][n]["full"]["interaction"], stats[n_value][n]["full"]["total"]) for n in networks])
        modified = sorted([(stats[n_value][n]["modified"]["interaction"], stats[n_value][n]["modified"]["total"]) for n in networks])
        
        full_x, full_y = zip(*full) if full else ([], [])
        mod_x, mod_y = zip(*modified) if modified else ([], [])
        
        plt.figure(figsize=(10, 5))
        
        if full:
            plt.scatter(full_x, full_y, label="Full Analysis")
            plt.plot(full_x, full_y)
        
        if modified:
            plt.scatter(mod_x, mod_y, label="Modified Analysis")
            plt.plot(mod_x, mod_y)
        
        plt.xlabel("Interactions")
        plt.ylabel("Total Time (s)")
        plt.title(f"Total Time vs Interactions (n = {n_value})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Plot: Median Time (Y-axis: median time in ms, X-axis: first interaction value)
def plot_median_time_old(stats):
    networks = list(stats.keys())

    # Extract full and modified median times for each network
    full_interactions = [stats[n]["full"]["interaction"] for n in networks]
    modified_interactions = [stats[n]["modified"]["interaction"] for n in networks]

    full_median_times = [stats[n]["full"]["median"] for n in networks]
    modified_median_times = [stats[n]["modified"]["median"] for n in networks]

    plt.figure(figsize=(10, 5))

    # Plot Full and Modified Analysis for Median Time
    plt.scatter(full_interactions, full_median_times, color="blue", label="Full Analysis")
    plt.scatter(modified_interactions, modified_median_times, color="red", label="Modified Analysis")

    plt.xlabel("Interactions")
    plt.ylabel("Median Time (s)")
    plt.title("Median Time vs Interactions (Full vs Modified Analysis)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_median_time(stats, n_value=None):
    if n_value is None:
        # Plot all n values in subplots
        n_values = sorted(stats.keys(), key=int)
        fig, axes = plt.subplots(1, len(n_values), figsize=(10 * len(n_values), 5))
        if len(n_values) == 1:
            axes = [axes]
        
        for idx, n_val in enumerate(n_values):
            ax = axes[idx]
            networks = list(stats[n_val].keys())
            
            full = sorted([(stats[n_val][n]["full"]["interaction"], stats[n_val][n]["full"]["median"]) for n in networks])
            modified = sorted([(stats[n_val][n]["modified"]["interaction"], stats[n_val][n]["modified"]["median"]) for n in networks])
            
            if full:
                full_x, full_y = zip(*full)
                ax.scatter(full_x, full_y, label="Full Analysis")
                ax.plot(full_x, full_y)
            
            if modified:
                mod_x, mod_y = zip(*modified)
                ax.scatter(mod_x, mod_y, label="Modified Analysis")
                ax.plot(mod_x, mod_y)
            
            ax.set_xlabel("Interactions")
            ax.set_ylabel("Median Time (ms)")
            ax.set_title(f"Median Time vs Interactions (n = {n_val})")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        # Plot single n value
        networks = list(stats[n_value].keys())
        
        full = sorted([(stats[n_value][n]["full"]["interaction"], stats[n_value][n]["full"]["median"]) for n in networks])
        modified = sorted([(stats[n_value][n]["modified"]["interaction"], stats[n_value][n]["modified"]["median"]) for n in networks])
        
        full_x, full_y = zip(*full) if full else ([], [])
        mod_x, mod_y = zip(*modified) if modified else ([], [])
        
        plt.figure(figsize=(10, 5))
        
        if full:
            plt.scatter(full_x, full_y, label="Full Analysis")
            plt.plot(full_x, full_y)
        
        if modified:
            plt.scatter(mod_x, mod_y, label="Modified Analysis")
            plt.plot(mod_x, mod_y)
        
        plt.xlabel("Interactions")
        plt.ylabel("Median Time (s)")
        plt.title(f"Median Time vs Interactions (n = {n_value})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# Plot: Average Time with SD (Y-axis: average time, X-axis: first interaction value)
def plot_avg_with_sd_old(stats):
    networks = list(stats.keys())

    # Extract full and modified data for average time and SD
    full_interactions = [stats[n]["full"]["interaction"] for n in networks]
    modified_interactions = [stats[n]["modified"]["interaction"] for n in networks]

    full_avg_times = [stats[n]["full"]["avg"] for n in networks]
    modified_avg_times = [stats[n]["modified"]["avg"] for n in networks]

    full_sd_times = [stats[n]["full"]["sd"] for n in networks]
    modified_sd_times = [stats[n]["modified"]["sd"] for n in networks]

    plt.figure(figsize=(10, 5))

    # Plot Full and Modified Analysis for Average Time with SD
    plt.errorbar(full_interactions, full_avg_times, yerr=full_sd_times, fmt="o", label="Full Analysis", capsize=5)
    plt.errorbar(modified_interactions, modified_avg_times, yerr=modified_sd_times, fmt="s", label="Modified Analysis",
                 capsize=5)

    plt.xlabel("Interactions")
    plt.ylabel("Average Time (s)")
    plt.title("Average Time with SD vs Interactions (Full vs Modified Analysis)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_avg_with_sd(stats, n_value=None):
    if n_value is None:
        # Plot all n values in subplots
        n_values = sorted(stats.keys(), key=int)
        fig, axes = plt.subplots(1, len(n_values), figsize=(10 * len(n_values), 5))
        if len(n_values) == 1:
            axes = [axes]
        
        for idx, n_val in enumerate(n_values):
            ax = axes[idx]
            networks = list(stats[n_val].keys())
            
            full = sorted([(stats[n_val][n]["full"]["interaction"], stats[n_val][n]["full"]["avg"], stats[n_val][n]["full"]["sd"]) for n in networks])
            modified = sorted([(stats[n_val][n]["modified"]["interaction"], stats[n_val][n]["modified"]["avg"], stats[n_val][n]["modified"]["sd"]) for n in networks])
            
            if full:
                full_x, full_avg, full_sd = zip(*full)
                ax.errorbar(full_x, full_avg, yerr=full_sd, fmt="o", capsize=5, label="Full Analysis")
                ax.plot(full_x, full_avg)
            
            if modified:
                mod_x, mod_avg, mod_sd = zip(*modified)
                ax.errorbar(mod_x, mod_avg, yerr=mod_sd, fmt="s", capsize=5, label="Modified Analysis")
                ax.plot(mod_x, mod_avg)
            
            ax.set_xlabel("Interactions")
            ax.set_ylabel("Average Time (ms)")
            ax.set_title(f"Average Time with SD vs Interactions (n = {n_val})")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        # Plot single n value
        networks = list(stats[n_value].keys())
        
        full = sorted([(stats[n_value][n]["full"]["interaction"], stats[n_value][n]["full"]["avg"], stats[n_value][n]["full"]["sd"]) for n in networks])
        modified = sorted([(stats[n_value][n]["modified"]["interaction"], stats[n_value][n]["modified"]["avg"], stats[n_value][n]["modified"]["sd"]) for n in networks])
        
        full_x, full_avg, full_sd = zip(*full) if full else ([], [], [])
        mod_x, mod_avg, mod_sd = zip(*modified) if modified else ([], [], [])
        
        plt.figure(figsize=(10, 5))
        
        if full:
            plt.errorbar(full_x, full_avg, yerr=full_sd, fmt="o", capsize=5, label="Full Analysis")
            plt.plot(full_x, full_avg)
        
        if modified:
            plt.errorbar(mod_x, mod_avg, yerr=mod_sd, fmt="s", capsize=5, label="Modified Analysis")
            plt.plot(mod_x, mod_avg)
        
        plt.xlabel("Interactions")
        plt.ylabel("Average Time (s)")
        plt.title(f"Average Time with SD vs Interactions (n = {n_value})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_full_vs_modified_distribution(full, modified):
    plt.figure(figsize=(8, 5))
    plt.hist(full, bins=15, alpha=0.7, label="Full")
    plt.hist(modified, bins=15, alpha=0.7, label="Modified")
    plt.xlabel("Total Runtime (s)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Runtime: Full vs Modified")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_runtime_distribution(diff):
    plt.figure(figsize=(8, 5))
    plt.hist(diff, bins=15)
    plt.xlabel("Runtime Difference (Full - Modified) [s]")
    plt.ylabel("Frequency")
    plt.title("Distribution of Runtime Differences")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from scipy.stats import ttest_rel, wilcoxon, shapiro

def statistical_analysis(stats, n_value=None):
    if n_value is None:
        # Analyze all n values
        results = {}
        for n_val in sorted(stats.keys(), key=int):
            results[n_val] = statistical_analysis(stats, n_val)
        return results
    else:
        # Analyze single n value
        networks = list(stats[n_value].keys())

        full_totals = np.array([stats[n_value][n]["full"]["total"] for n in networks])
        modified_totals = np.array([stats[n_value][n]["modified"]["total"] for n in networks])

        diff = full_totals - modified_totals

        print(f"\nStatistical Analysis for n = {n_value}: Full vs Modified Total Runtime")
        print("--------------------------------------------------")
        print(f"Mean Full Runtime (s): {np.mean(full_totals):.2f}")
        print(f"Mean Modified Runtime (s): {np.mean(modified_totals):.2f}")
        print(f"Mean Difference (Full - Modified): {np.mean(diff):.2f} s")

        # Normality test on differences
        shapiro_stat, shapiro_p = shapiro(diff)
        print(f"\nShapiro-Wilk test for normality of differences: p = {shapiro_p:.4g}")

        if shapiro_p > 0.05:
            print("Differences look normal → using paired t-test")
            t_stat, p_val = ttest_rel(full_totals, modified_totals)
            test_name = "Paired t-test"
        else:
            print("Differences not normal → using Wilcoxon signed-rank test")
            t_stat, p_val = wilcoxon(full_totals, modified_totals)
            test_name = "Wilcoxon signed-rank test"

        print(f"{test_name} statistic: {t_stat:.4f}")
        print(f"{test_name} p-value: {p_val:.4g}")

        if p_val < 0.05:
            print("Result: Statistically significant difference (p < 0.05)")
        else:
            print("Result: No statistically significant difference (p ≥ 0.05)")

        return full_totals, modified_totals, diff



if __name__ == "__main__":
    # Load and parse the output file
    results = parse_analysis_output("output.txt")

    # Compute statistics for each network
    stats = compute_statistics(results)

    # Print out the summary of all statistics
    print_statistics(stats)

    # Create the plots for all networks (all n values)
    plot_total_time(stats)
    plot_median_time(stats)
    plot_avg_with_sd(stats)

    # Statistical analysis for all n values
    stat_results = statistical_analysis(stats)
    
    # Distribution plots for each n value
    for n_value in sorted(stats.keys(), key=int):
        if isinstance(stat_results, dict) and n_value in stat_results:
            full, modified, diff = stat_results[n_value]
            print(f"\n--- Distribution plots for n = {n_value} ---")
            plot_runtime_distribution(diff)
            plot_full_vs_modified_distribution(full, modified)
        elif not isinstance(stat_results, dict):
            # Backward compatibility if only one n value
            full, modified, diff = stat_results
            plot_runtime_distribution(diff)
            plot_full_vs_modified_distribution(full, modified)
            break