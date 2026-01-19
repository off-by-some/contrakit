#!/usr/bin/env python3
"""
Run all hallucination experiments and capture terminal output.

This script runs all experiments in numerical order (1-11) and saves
all terminal output to terminal_output.txt for analysis.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_experiment(experiment_num):
    """Run a single experiment and return its output."""
    experiment_dir = Path(f"experiment_{experiment_num}")
    run_script = experiment_dir / "run.py"

    if not run_script.exists():
        return f"ERROR: {run_script} not found\n"

    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT {experiment_num}")
    print(f"{'='*80}\n")

    try:
        # Run the experiment and capture output
        result = subprocess.run(
            [sys.executable, str(run_script)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )

        output = f"EXPERIMENT {experiment_num} OUTPUT:\n"
        output += f"Return code: {result.returncode}\n\n"

        if result.stdout:
            output += "STDOUT:\n" + result.stdout + "\n"

        if result.stderr:
            output += "STDERR:\n" + result.stderr + "\n"

        output += f"{'='*80}\n"

        return output

    except subprocess.TimeoutExpired:
        return f"EXPERIMENT {experiment_num}: TIMEOUT (1 hour limit exceeded)\n{'='*80}\n"

    except Exception as e:
        return f"EXPERIMENT {experiment_num}: ERROR - {str(e)}\n{'='*80}\n"

def main():
    """Run all experiments and save output."""
    output_file = Path("terminal_output.txt")

    print("Starting hallucination experiments run...")
    print(f"Output will be saved to: {output_file.absolute()}")

    # Header for output file
    header = f"""HALLUCINATION EXPERIMENTS - COMPLETE RUN
{'='*80}
Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

"""

    with open(output_file, 'w') as f:
        f.write(header)

    # Run experiments 1 through 11
    for exp_num in range(1, 12):  # 1 to 11 inclusive
        print(f"Running experiment {exp_num}...")

        output = run_experiment(exp_num)

        # Append to file
        with open(output_file, 'a') as f:
            f.write(output)

        print(f"Experiment {exp_num} completed.")

    # Footer
    footer = f"""
{'='*80}
All experiments completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

    with open(output_file, 'a') as f:
        f.write(footer)

    print(f"\nAll experiments completed! Output saved to {output_file}")
    print(f"Total experiments run: 11")

if __name__ == "__main__":
    main()