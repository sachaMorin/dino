#!/usr/bin/env python
"""Run the nth experiment provided in a csv config file.

If you use a job number higher than the current number of experiments in the config, the jobs marked with job mod max_job
will be executed with a different random seed. E.g., if you have 3 experiments in the config and you call the jobs
 0, 1, ..., 8, all experiments will be executed 3 times with different seeds."""
import comet_ml
import os
import argparse

import pandas as pd
from run_experiment import run_experiment

# Parser
parser = argparse.ArgumentParser(description='Run full experiments with specific hyper parameters '
                                             'as configured in a .csv file...')
parser.add_argument('--comet_tag', '-t',
                    help='Comet tag for experiment', type=str, default=None)
parser.add_argument('--job', '-j',
                    help='Run all experiment in schedule marked with this number. Experiments with the same number are run sequentially. Intended to use with the Slurm Id.',
                    type=int,
                    default=0)
parser.add_argument('--config',
                    '-c',
                    help='Schedule path. A file listing all experiments to run with associated ids.',
                    type=str,
                    default=os.path.join(os.getcwd(), 'exp_schedule', 'main.csv'))
parser.add_argument('--data_path',
                    '-d',
                    help='Data path. Otherwise assumes a \'data\' folder in current working directory.',
                    type=str,
                    default=os.path.join(os.getcwd(), '../data'))
parser.add_argument('--write_path',
                    '-w',
                    help='Where to write temp files. Otherwise assumes current working directory.',
                    type=str,
                    default=os.getcwd())

args = parser.parse_args()

# Get Schedule
# Read schedule and only keep experiment tagged with current job
schedule = pd.read_csv(args.config)
n_jobs = int(schedule['job'].max() + 1)  # Number of different jobs

# Get fold and job_id from slurm id
# Cycle through values. args.job can take a value between 0 and (number of jobs * number of folds) - 1.
# Think of it as cycling through the schedule to run all jobs with different seeds.
seed, job_no = divmod(args.job, n_jobs)

# Filter schedule
schedule = schedule.loc[schedule['job'] == job_no].drop(['job'], axis=1)
if schedule.shape[0] == 0:
    raise Exception(f'No job marked with the following id : {args.job}.')

# Launch experiments
for _, exp_params in schedule.iterrows():
    params = exp_params.dropna().to_dict()

    # Use same random state for all experiments for reproducibility
    # Note : this seed is used for splitting the data. Ancillary seeds are used to initialize models
    params['random_state'] = (seed + 1) * 1234
    params['data_path'] = args.data_path
    params['write_path'] = args.write_path
    params['comet_tag'] = args.comet_tag

    try:
        run_experiment(**params)
    except Exception as e:
        print(e)