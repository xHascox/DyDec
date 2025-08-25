import subprocess
import time
from datetime import datetime
from stat_analysis import make_plots

SEED = 0
N_EXPERIMENTS = 1 # TODO 10
AGENTS = [ 'if_if',  f'tfpp_aim_{SEED}', f'tfpp_l6_{SEED}', f'tfpp_lav_{SEED}', f'tfpp_wp_{SEED}', "neat_neat",]

SCENARIOS_FILE = "scenario definitions/scenarios_val_collusion_adjusted_only4.json"
SCENARIOS_FILE = "scenario definitions/scenarios_val_single_adjusted_only4.json"

SCENARIOS = list(range(14)) # only y=4 collusion
SCENARIOS = list(range(10)) # only y=4 single

csvfile = './logs/'+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ '.csv'
def main():
    # Calling child process
    for agent in AGENTS:
        for i in range(N_EXPERIMENTS):
            for scenario in SCENARIOS:
                attempts = 0
                while 1:
                    attempts += 1
                    print(f"{i}/{N_EXPERIMENTS} Running scenario {scenario} with agent {agent} attempt {attempts} saving to {csvfile}")
                    result = subprocess.run(['python', 'spawnactors_validate_cli.py', '-s', str(scenario), '-a', agent, "-l", csvfile, "-d", SCENARIOS_FILE], capture_output=True, text=True)

                    
                    # Handling different return codes
                    if result.returncode == 0:
                        print("Success: ", result.stdout)
                        break
                    elif result.returncode == 2:
                        print("Script failed with an error: ", result.stderr)
                        time.sleep(1)
                    else:
                        print(f"Script exited with unexpected return code: {result.returncode}")
                        print(result.stderr)

if __name__ == "__main__":
    main()
    make_plots(csvfile)
