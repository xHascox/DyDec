import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import traceback
import csv
from matplotlib.collections import PolyCollection
import scipy.stats

K = 80 # Number of timesteps to consider for average speed and brake activation

OUTPUT_DIR = 'plots/'

def make_plots(scenarios_file = "2025-06-07 13:47:43.csv"):

    SCENARIO_NAME_MAPPING = {
        "Val0 : disguise y=2": "Route A: apple y=2",
        "Val0 : disguise y=3": "Route A: apple y=3",
        
        "Val0 : disguise y=5": "Route A: apple y=5",
        "Val2 : disguise y=2": "Val2 : apple y=2",
        "Val2 : disguise y=3": "Val2 : apple y=3",
        "Val2 : disguise y=4": "Val2 : apple y=4",
        "Val2 : disguise y=5": "Val2 : apple y=5",
        "Val3 : disguise y=2": "Route B: apple y=2",
        "Val3 : disguise y=3": "Route B: apple y=3",
        
        "Val3 : disguise y=5": "Route B: apple y=5",
        "Val3 : lefft y=5": "Route B: left y=5",

        "Val0 : adv y=4": "Attack",
        "Val0 : benign y=4": "Benign",
        "Val0 : left y=4": "Left patch",
        "Val0 : right y=4": "Right patch",
        "Val0 : no pedestrians": "No pedestrians",
        "Val0 : disguise y=4": "Apple",

        "Val3 : adv y=4": "Attack",
        "Val3 : benign y=4": "Benign",
        "Val3 : left y=4": "Left patch",
        "Val3 : right y=4": "Right patch",
        "Val3 : no pedestrians": "No pedestrians",
        "Val3 : disguise y=4": "Apple",

        "adv y=4": "Attack",
        "benign y=4": "Benign",
        "left y=4": "Left patch",
        "right y=4": "Right patch",
        "no pedestrians": "No pedestrians",
        "disguise y=4": "Apple",
    }

    GROUP_NAME_MAPPING = {
        "Val0 y=4": "Route A",
        "Val3 y=4": "Route B",
    }

    scenarios = pd.read_csv(scenarios_file, header=None)
    scenarios_filename = Path(scenarios_file).name


    # Get the image paths and labels, but keep only first occurrence of each unique label
    df_unique = scenarios.drop_duplicates(subset=[2], keep='first')  # Keep first occurrence of each label
    image_paths = df_unique[6].tolist()
    labels = df_unique[2].tolist()

    # Calculate grid dimensions
    n_images = len(image_paths)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(25, 13))


    # Handle axes flattening properly
    if grid_size > 1:
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    # Plot each image with its label
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(img_path)
        axes_flat[idx].imshow(img)
        label = SCENARIO_NAME_MAPPING.get(label, label)
        axes_flat[idx].set_title(label, pad=2, wrap=True) # NOTE changed fontsize
        axes_flat[idx].axis('off')  # Hide axes

    # Remove empty subplots
    for idx in range(n_images, grid_size * grid_size):
        fig.delaxes(axes_flat[idx])

    #plt.tight_layout()
    # Save the plot
    output_path = Path(f'{OUTPUT_DIR}plots_{scenarios_filename}/all_scenarios.png')

    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    # Print how many unique labels were found
    print(f"Plotting {n_images} images with unique labels (from original {len(scenarios)} images)")

    

    # Specify groups of scenarios to combine

    SCENARIO_GROUPS = [
        #{
        #    "name": "Val0 y=3",  # Name for the first group
        #    "scenarios": ["Val0 : actual stop sign", 
        #                "Val0 : adv y=3", 
        #                "Val0 : benign y=3",
        #                "Val0 : disguise y=3",
        #                "Val0 : left y=3",
        #                "Val0 : right y=3",
        #                "Val0 : no pedestrians"]
        #},
        {
            "name": "Val0 y=4",  # Name for the first group
            "scenarios": [#"Val0 : actual stop sign", 
                        "Val0 : adv y=4", 
                        "Val0 : benign y=4",
                        "Val0 : disguise y=4",
                        "Val0 : left y=4",
                        "Val0 : right y=4",
                        "Val0 : no pedestrians"]
        },
        #{
        #    "name": "Val2 y=3",    # Name for the second group
        #    "scenarios": ["Val2 : actual stop sign", 
        #                "Val2 : adv y=3", 
        #                "Val2 : benign y=3",
        #                "Val2 : disguise y=3",
        #                "Val2 : no pedestrians"]
        #},
        #{
        #    "name": "Val2 y=4",    # Name for the second group
        #    "scenarios": ["Val2 : actual stop sign", 
        #                "Val2 : adv y=4", 
        #                "Val2 : benign y=4",
        #                "Val2 : disguise y=4",
        #                "Val2 : no pedestrians"]
        #},
        #{
        #    "name": "Val3 y=3",
        #    "scenarios": ["Val3 : actual stop sign", 
        #                "Val3 : adv y=3", 
        #                "Val3 : benign y=3",
        #                "Val3 : disguise y=3",
        #                "Val3 : left y=3",
        #                "Val3 : right y=3",
        #                "Val3 : no pedestrians"]
        #},
        {
            "name": "Val3 y=4",
            "scenarios": [#"Val3 : actual stop sign", 
                        "Val3 : adv y=4", 
                        "Val3 : benign y=4",
                        "Val3 : disguise y=4",
                        "Val3 : left y=4",
                        "Val3 : right y=4",
                        "Val3 : no pedestrians"]
        },
        #{
        #    "name": "Val3 y=2",
        #    "scenarios": ["Val3 : actual stop sign", 
        #                "Val3 : adv y=2", 
        #                "Val3 : benign y=2",
        #                "Val3 : disguise y=2",
        #                "Val3 : left y=2",
        #                "Val3 : right y=2",
        #                "Val3 : no pedestrians"]
        #},
    ]

    # Use different colors for each scenario in the group
    #colors = plt.cm.tab20(np.linspace(0, 1, len(scenario_group)))
    colors = ("red", "green", "blue", "magenta", "cyan", "black", "brown", "purple" )
    
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size between two groups"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:  # Require at least 2 samples in each group
            return float('nan')
        
        # Pooled standard deviation
        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_s if pooled_s != 0 else float('nan')
        return d

    def create_boxplots(scenarios_df, scenario_groups, avg_brake_by_agent_scenario=[], avg_speed_by_agent_scenario=[]):        
        
        # First find global y-axis limits across all agents
        all_speeds = []
        for group in scenario_groups:
            for scenario in group["scenarios"]:
                # Use avg_speed_by_agent_scenario for global limits
                for agent in scenarios_df[1].unique():
                    speeds = avg_speed_by_agent_scenario.get(agent, {}).get(scenario, np.array([]))
                    if len(speeds) > 0:
                        all_speeds.extend(speeds)
        
        y_min = min(all_speeds) - 0.5 if all_speeds else 0
        y_max = max(all_speeds) + 0.5 if all_speeds else 1

        # Get unique agents
        unique_agents = scenarios_df[1].unique()

        cohens_d_results = []  # Collect results here

        # Create separate plots for each agent
        for agent in unique_agents:
            agent_scenarios = scenarios_df[scenarios_df[1] == agent]
            
            plt.figure(figsize=(15, 10))
            
            for group_idx, group in enumerate(scenario_groups):
                group_data = []
                labels = []
                group_colors = []
                
                for scenario_idx, scenario in enumerate(group["scenarios"]):
                    # Use avg_speed_by_agent_scenario for speed data
                    speeds = avg_speed_by_agent_scenario.get(agent, {}).get(scenario, np.array([]))
                    brakes = avg_brake_by_agent_scenario.get(agent, {}).get(scenario, np.array([]))
                    if len(speeds) > 0:
                        group_data.append(speeds)
                        display_name = SCENARIO_NAME_MAPPING.get(scenario, scenario.split(" : ")[1])

                        # Calculate statistics
                        mean = np.mean(speeds)
                        std = np.std(speeds)
                        
                        print(f"Scenario: {display_name}")
                        print(f"  Mean: {mean:.4f}")
                        print(f"  Std:  {std:.4f}")
                        print(f"  N:    {len(speeds)}")

                        # Calculate Cohen's d against other scenarios in the group
                        for other_scenario in group["scenarios"]:
                            if other_scenario != scenario:
                                other_speeds = avg_speed_by_agent_scenario.get(agent, {}).get(other_scenario, np.array([]))
                                other_brakes = avg_brake_by_agent_scenario.get(agent, {}).get(other_scenario, np.array([]))

                                # Only proceed if there is data for the other scenario
                                if len(other_speeds) > 0:
                                    # Speed stats
                                    d_speed = cohens_d(speeds, other_speeds)
                                    try:
                                        if len(speeds) == len(other_speeds) and len(speeds) > 0:
                                            wilcoxon_speed_stat, wilcoxon_speed_p = scipy.stats.ranksums(speeds, other_speeds)
                                        else:
                                            wilcoxon_speed_stat, wilcoxon_speed_p = float('nan'), float('nan')
                                    except Exception:
                                        wilcoxon_speed_stat, wilcoxon_speed_p = float('nan'), float('nan')

                                    # Brake stats
                                    d_brake = cohens_d(brakes, other_brakes)
                                    try:
                                        if len(brakes) == len(other_brakes) and len(brakes) > 0:
                                            wilcoxon_brake_stat, wilcoxon_brake_p = scipy.stats.ranksums(brakes, other_brakes)
                                        else:
                                            wilcoxon_brake_stat, wilcoxon_brake_p = float('nan'), float('nan')
                                    except Exception:
                                        wilcoxon_brake_stat, wilcoxon_brake_p = float('nan'), float('nan')

                                    other_display = SCENARIO_NAME_MAPPING.get(other_scenario, other_scenario.split(" : ")[1])
                                    print(f"  Cohen's d vs {other_display}: {d_speed:.4f}")
                                    # Store both speed and brake results
                                    cohens_d_results.append({
                                        "Agent": agent,
                                        "Group": group["name"],
                                        "Scenario": display_name,
                                        "Other Scenario": other_display,
                                        "Cohens_d_speed": d_speed,
                                        "Wilcoxon_p_speed": wilcoxon_speed_p,
                                        "Cohens_d_brake": d_brake,
                                        "Wilcoxon_p_brake": wilcoxon_brake_p,
                                        "N_Scenario": len(speeds),
                                        "N_Other": len(other_speeds),
                                        "Avg_Speed": np.mean(speeds) if len(speeds) > 0 else float('nan'),
                                        "Avg_Brake": np.mean(brakes) if len(brakes) > 0 else float('nan'),
                                        "Avg_Speed_Other": np.mean(other_speeds) if len(other_speeds) > 0 else float('nan'),
                                        "Avg_Brake_Other": np.mean(other_brakes) if len(other_brakes) > 0 else float('nan')
                                    })

                        labels.append(display_name)
                        group_colors.append(colors[scenario_idx])
                
                if group_data:
                    ax = plt.subplot(2, 4, group_idx + 1)
                    bp = plt.boxplot(group_data, labels=labels, vert=True, patch_artist=True)
                    
                    # Color each box
                    for box, color in zip(bp['boxes'], group_colors):
                        box.set(facecolor=color, alpha=0.6)
                        box.set(color=color)
                    
                    # Color the whiskers and caps
                    for whisker, median, cap, color in zip(bp['whiskers'][::2], bp['medians'], bp['caps'][::2], group_colors):
                        whisker.set(color=color)
                        bp['whiskers'][bp['whiskers'].index(whisker)+1].set(color=color)
                        median.set(color=color)
                        cap.set(color=color)
                        bp['caps'][bp['caps'].index(cap)+1].set(color=color)
                    
                    # Color the fliers (outlier points)
                    for flier, color in zip(bp['fliers'], group_colors):
                        flier.set(markerfacecolor=color, marker='o', markeredgecolor=color, alpha=0.6)
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.title(f"{GROUP_NAME_MAPPING.get(group['name'], group['name'])}\nAverage Speed Distribution")
                    plt.ylabel("Speed (m/s)")
                    plt.grid(True, alpha=0.3)
                    plt.ylim(y_min, y_max)
            
            plt.suptitle(f'Agent: {agent}', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save plot for this agent
            safe_agent_name = "".join(x for x in agent if x.isalnum() or x in (' ','-','_')).replace(' ', '_')
            output_path = Path(f'{OUTPUT_DIR}plots_{scenarios_filename}/speed_boxplots_{safe_agent_name}.png')
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
            plt.close()

        # Write Cohen's d results to CSV
        output_csv = Path(f'{OUTPUT_DIR}plots_{scenarios_filename}/cohens_d_results_for_K_{K}.csv')
        output_csv.parent.mkdir(exist_ok=True)
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = [
                "Agent", "Group", "Scenario", "Other Scenario",
                "Cohens_d_speed", "Wilcoxon_p_speed",
                "Cohens_d_brake", "Wilcoxon_p_brake",
                "N_Scenario", "N_Other",
                "Avg_Speed", "Avg_Brake", "Avg_Speed_Other", "Avg_Brake_Other"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in cohens_d_results:
                writer.writerow(row)

    

    def create_speed_histograms(scenarios_df, scenario_groups, bins=40):    
        exclude_scenarios = ["Val3 : actual stop sign", 
                        "Val3 : disguise y=4",
                        "Val3 : left y=4",
                        "Val3 : right y=4",
                        "Val3 : no pedestrians",
                        "Val0 : actual stop sign", 
                        "Val0 : disguise y=4",
                        "Val0 : left y=4",
                        "Val0 : right y=4",
                        "Val0 : no pedestrians",
                        "Val0 : actual stop sign", 
                        "Val0 : disguise y=3",
                        "Val0 : left y=3",
                        "Val0 : right y=3",
                        "Val0 : no pedestrians"]
        include_groups = ["Val3 y=4", "Val0 y=4", "Val0 y=3", "Val3 y=3", "Val3 y=2"]

        # Get unique agents
        unique_agents = scenarios_df[1].unique()
        
        # First find global x-axis limits
        all_speeds = []
        for group in scenario_groups:
            if group["name"] not in include_groups:
                continue
            for scenario in group["scenarios"]:
                scenario_rows = scenarios_df[scenarios_df[2] == scenario]
                if not scenario_rows.empty and not scenario in exclude_scenarios:
                    all_speeds.extend(scenario_rows[3].values)
        
        if len(all_speeds) == 0:
            x_min = -0.1
            x_max = 10
        else:
            x_min = min(all_speeds) - 0.1
            x_max = max(all_speeds) + 0.1
        
        # Create separate plots for each agent
        for agent in unique_agents:
            agent_scenarios = scenarios_df[scenarios_df[1] == agent]
            
            plt.figure(figsize=(20, 10))
            
            for group_idx, group in enumerate(scenario_groups):
                if group["name"] not in include_groups:
                    continue
                ax = plt.subplot(2, 4, group_idx + 1)
                
                # Plot histogram for each scenario in the group
                max_count = 0  # Track maximum count for y-axis scaling
                
                for scenario_idx, scenario in enumerate(reversed(group["scenarios"])):
                    if scenario in exclude_scenarios:  # Skip excluded scenarios
                        continue
                    scenario_rows = agent_scenarios[agent_scenarios[2] == scenario]
                    if not scenario_rows.empty:
                        speeds = scenario_rows[3].values
                        display_name = SCENARIO_NAME_MAPPING.get(scenario, scenario.split(" : ")[1])
                        
                        # Plot histogram with transparency
                        colors = ("black", "purple", "cyan", "blue", "green", "red", "brown", "magenta" ) # NOTE Overrule colors

                        counts, bins, _ = plt.hist(speeds, bins=bins, 
                                                alpha=0.5,
                                                color=colors[scenario_idx],
                                                label=f'{display_name} ', #(n={len(speeds)})',
                                                range=(x_min, x_max))
                        max_count = max(max_count, max(counts))
                
                plt.xlim(x_min, x_max)
                plt.ylim(0, max_count * 1.1)  # Add 10% margin on top
                
                plt.title(f"{group['name']}\nSpeed Distribution")
                plt.xlabel("Speed (m/s)")
                plt.ylabel("Count")
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'Agent: {agent}', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save plot for this agent
            safe_agent_name = "".join(x for x in agent if x.isalnum() or x in (' ','-','_')).replace(' ', '_')
            output_path = Path(f'{OUTPUT_DIR}plots_{scenarios_filename}/speed_histograms_{safe_agent_name}.png')
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

    # Add this line in the main make_plots function after creating the boxplots
    create_speed_histograms(scenarios, SCENARIO_GROUPS)

    # Create a dictionary to group data by agent, scenario, and column
    grouped_experiments = defaultdict(lambda: defaultdict(list))

    

    # Extract relevant columns and group the data
    for _, row in scenarios.iterrows():
        agent_name = row.iloc[1]
        scenario_name = row.iloc[2]
        filepath = row.iloc[5]
        
        if not any(scenario_name in group["scenarios"] for group in SCENARIO_GROUPS):
            continue
            
        if pd.notna(filepath):
            try:
                data = pd.read_csv(filepath)
                grouped_experiments[agent_name][scenario_name].append(data)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")



    # Aggregate average brake activation for each agent/scenario
    avg_brake_by_agent_scenario = defaultdict(dict)
    for agent_name, scenario_data in grouped_experiments.items():
        for scenario_name, data_list in scenario_data.items():
            # Concatenate all runs for this scenario
            all_brake = []
            for df in data_list:
                if "brake" in df.columns:
                    avg_brake = np.mean(df["brake"].values[:K])
                    all_brake.append(avg_brake)
            if all_brake:
                avg_brake_by_agent_scenario[agent_name][scenario_name] = np.array(all_brake)
    print(avg_brake_by_agent_scenario)

    # Aggregate average speed for each agent/scenario for K timesteps
    avg_speed_by_agent_scenario = defaultdict(dict)
    for agent_name, scenario_data in grouped_experiments.items():
        for scenario_name, data_list in scenario_data.items():
            # Concatenate all runs for this scenario
            all_speed= []
            for df in data_list:
                if "brake" in df.columns:
                    avg_speed = np.mean(df["speed"].values[:K])
                    all_speed.append(avg_speed)
            if all_speed:
                avg_speed_by_agent_scenario[agent_name][scenario_name] = np.array(all_speed)
    print(avg_speed_by_agent_scenario)

    create_boxplots(scenarios, SCENARIO_GROUPS, avg_brake_by_agent_scenario, avg_speed_by_agent_scenario)


    # Process each agent and create combined plots for each scenario group
    for agent_name, scenario_data in grouped_experiments.items():
        for group_idx, group in enumerate(SCENARIO_GROUPS):
            scenario_group = group["scenarios"]
            group_name = group["name"]
            
            if not any(scenario in scenario_data for scenario in scenario_group):
                continue
                
            try:
                # Get unique columns from the first scenario's data
                sample_data = next(iter(scenario_data.values()))[0]
                unique_columns = sample_data.columns
                unique_columns = [col for col in unique_columns if col in ["speed", "brake"]]
                
                # Set global font sizes
                plt.rcParams.update({'font.size': 17})  # Base font size

                # Create a subplot for each column
                num_cols = len(unique_columns)
                fig = plt.figure(figsize=(20, 6 * ((num_cols + 1) // 2)))  # Adjusted for 2 plots per row
                
                # Add title for the whole figure
                fig.suptitle(f'Agent: {agent_name}\nScenario Group: {group_name}', 
                           fontsize=20, y=1.1)
                
                #fig.suptitle(f'Agent: {agent_name} on {group_name}\nScenarios: {", ".join(scenario_group)}\n',fontsize=12, y=1.05)
                plt.subplots_adjust(top=0.90)  
                
                # Create subplot for each column
                for col_idx, column in enumerate(unique_columns, 1):
                    ax = fig.add_subplot((num_cols + 1) // 2, 2, col_idx)  # Changed to 2 columns
                    
                    if column == "brake" or column == "throttle":
                        # Process each scenario in this group
                        scenarios_with_data = [s for s in reversed(scenario_group) if s in scenario_data]

                        for scenario_idx, scenario_name in enumerate(scenarios_with_data):  # Added reverse here
                              
                            data_list = scenario_data[scenario_name]
                            temp_colors = colors[:len(scenario_group)]
                            base_color = temp_colors[scenario_group.index(scenario_name)]
                            
                            # Stack all experiments for this column
                            all_runs = np.stack([df[column].values for df in data_list])
                            mean_values = np.mean(all_runs, axis=0)
                            
                            # Adjust offset calculation to plot from bottom to top
                            offset = len(scenarios_with_data) - scenario_idx  # Changed this line
                            offset = scenario_idx + 1
                            # Create horizontal lines with fill
                            x = np.arange(len(mean_values))
                            
                            # Plot horizontal line for this scenario
                            ax.axhline(y=offset, color=base_color, linestyle='-', alpha=0.3)
                            
                            display_name = SCENARIO_NAME_MAPPING.get(scenario_name, scenario_name)

                            # Create fill_between with alpha proportional to mean_values
                            print("\n\n")
                            for i in range(len(x)-1):
                                ax.fill_between([x[i], x[i+1]], 
                                            offset-0.4, offset+0.4,
                                            color=base_color,
                                            alpha=mean_values[i],  # Alpha varies with mean value
                                            label=f'{display_name}' if i == 0 else "") # (n={len(data_list)})
                            
                        # Adjust y-axis
                        ax.set_ylim(0.5, len(scenarios_with_data) + 1.5)
                        ax.set_yticks(range(1, len(scenarios_with_data) + 1))
                        ax.set_yticklabels([SCENARIO_NAME_MAPPING.get(s, s) for s in scenarios_with_data])
                    else:
                        # Process each scenario in this group
                        for scenario_idx, scenario_name in enumerate(scenario_group):
                            if scenario_name not in scenario_data:
                                continue
                                
                            data_list = scenario_data[scenario_name]
                            base_color = colors[scenario_idx]
                            
                            # Stack all experiments for this column
                            all_runs = np.stack([df[column].values for df in data_list])
                            mean_values = np.mean(all_runs, axis=0)
                            std_values = np.std(all_runs, axis=0)
                            x = np.arange(len(mean_values))
                            
                            display_name = SCENARIO_NAME_MAPPING.get(scenario_name, scenario_name)

                            # Plot mean line
                            label = f'{display_name}'# (n={len(data_list)})'
                            linestyle = ['-', '--', ':', '-.'][scenario_idx % 4]

                            ax.plot(x, mean_values, label=label,
                                color=base_color,
                                linestyle=linestyle,
                                alpha=0.8)
                            
                            # Add shaded area for standard deviation
                            ax.fill_between(x,
                                        mean_values - std_values,
                                        mean_values + std_values,
                                        color=base_color,
                                        alpha=0.3)
                    

                    
                    # Set subplot titles and grid
                    plot_labels = {"speed": "Vehicle speed", "brake": "Brake activation", "throttle": "Throttle Value (alpha = avg)", "steer": "Steering Value"}
                    ax.set_title(plot_labels.get(column, column))
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('Time Step')
                    y_labels = {"speed": "Speed (m/s)", "brake": "", "throttle": "", }
                    ax.set_ylabel(y_labels.get(column, 'Value'))
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                    # Modify legend handling - only add legend for even numbered subplots
                    ax.set_title(plot_labels.get(column, column))
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('Time Step')
                    y_labels = {"speed": "Speed (m/s)", "brake": "", "throttle": "", }
                    ax.set_ylabel(y_labels.get(column, 'Value'))
                    
                    # Move legend outside to the right of both plots
                    if col_idx % 2 == 0 or col_idx == len(unique_columns):
                        handles, labels = ax.get_legend_handles_labels()
                        fig.legend(handles, labels, bbox_to_anchor=(1, 0.5), loc='center left')
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()


                # Adjust layout to prevent overlap
                plt.tight_layout()
                
                # Create safe filename from group name
                safe_group_name = "".join(x for x in group_name if x.isalnum() or x in (' ','-','_')).replace(' ', '_')
                
                # Save two versions - with and without std deviation
                base_output_path = Path(f'{OUTPUT_DIR}plots_{scenarios_filename}/{agent_name}_comparison_{safe_group_name}')
                
                # Save version with std deviation
                output_path = Path(f'{base_output_path}_with_std.png')
                output_path.parent.mkdir(exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
                print(f"Saving plot with std to {output_path}")

                # Remove std deviation fills and save second version
                for ax in fig.get_axes():
                    title = ax.get_title()
                    if "Vehicle speed" in title:  # Only remove std deviation from speed plot
                        for collection in ax.collections[:]:
                            if isinstance(collection, PolyCollection):
                                collection.remove()
                
                # Save version without std deviation
                output_path = Path(f'{base_output_path}_no_std.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
                print(f"Saving plot without std to {output_path}")

                plt.close()
                
            except Exception as e:
                print(f"Error processing agent {agent_name} for group '{group_name}': {e}")
                print(f"Error occurred on line {e.__traceback__.tb_lineno}")
                traceback.print_exc()

    print("Plotting complete! v2")
    return True
