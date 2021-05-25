import json
import logging
import os
import argparse

from pathlib import Path
from typing import Union, List, Dict

from HPOBenchExperimentUtils.core.trajectories import create_trajectory
from HPOBenchExperimentUtils.utils.validation_utils import load_json_files

from HPOBenchExperimentUtils.utils import TRAJECTORY_V1_FILENAME, TRAJECTORY_V2_FILENAME, RUNHISTORY_FILENAME

from HPOBenchExperimentUtils import _log as _root_log

_root_log.setLevel(level=logging.INFO)
_log = logging.getLogger(__name__)


def extract_trajectory(output_dir: Union[Path, str], debug: Union[bool, None] = False):
    """
    We want to create a trajectory from the results of the previous optimization run.
    This method collects all runhistories which are in a certain output directory. Note that we search recursively
    through the specified directory.

    We offer two different trajectories. We refer to them as 'v1' and 'v2'.

    v1: The trajectory has incumbents ordered in time of appearance. A challenger configuration becomes incumbent if it
    either on a higher budget than the current incumbent OR on the same budget but with a lower (better) function value.

    v2: Similar to v1. But we ignore the budget in this case. Therefore, if a configuration has a better function value
    than the current incumbent, it becomes the new incumbent.

    After calling this script, the two trajectories are created and stored in the same folder where the runhistory
    was found.

    NOTE: This script is automatically called, after the optimization process has finished.

    STEPS:
    ------
    1) Load all runhistories from the optimization step.
        -> Search for the runhistory files recursively. Perform all other steps for all runhistories.
    2) For each history RH.
        3) Read in all runs of RH
        4) Extract Trajectory 1: Lower is better and larger budget is better.
        5) Extract Trajectory 2: Ignore Budget, only lower is better.
        6) Save both trajectories to file to the same folder as from RH.
    7) THE END.


    Parameters
    ----------
    output_dir : str, Path
        Directory where the optimizer stored its results.

    debug: bool, None
        Enables the debug message logging.
    """

    _log.info('Start extracting the trajectories')

    if debug:
        _root_log.setLevel(level=logging.DEBUG)

    output_dir = Path(output_dir)
    assert output_dir.is_dir(), f'Result folder doesn\'t exist: {output_dir}'

    # Search all runhistories in the output directory
    runhistory_paths = list(output_dir.rglob(RUNHISTORY_FILENAME))

    # Load the runhistories
    runhistories = load_json_files(runhistory_paths)

    def print_traj(trajectory):
        for i, r in enumerate(trajectory):
            if i != 0:
                print(f'{i:2d}, {r["function_call"]:3d}, {r["function_value"]:.15f},\t {list(r["fidelity"].values())[0]}')

    for i_rh, (runhistory, runhistory_path) in enumerate(zip(runhistories, runhistory_paths)):

        trajectory = create_trajectory(runhistory, bigger_is_better=True)
        # print_traj(trajectory)
        write_list_of_dicts_to_file(runhistory_path.parent / TRAJECTORY_V1_FILENAME, trajectory)

        trajectory = create_trajectory(runhistory, bigger_is_better=False)
        # print_traj(trajectory)
        write_list_of_dicts_to_file(runhistory_path.parent / TRAJECTORY_V2_FILENAME, trajectory)

    return 1


def write_list_of_dicts_to_file(output_file: Path, data: List[Dict]):
    with output_file.open('w') as fh:
        for dict_to_store in data:
            json.dump(dict_to_store, fh)
            fh.write(os.linesep)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='HPOBench Wrapper',
                                     description='Extract the trajectories from the runhistory of the optimization run')

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--debug', action='store_true', default=False, help="When given, enables debug mode logging.")

    args = parser.parse_args()
    extract_trajectory(output_dir=Path(args.output_dir), debug=args.debug)
