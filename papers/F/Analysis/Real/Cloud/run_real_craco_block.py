# Running this command: python ../py/build_real_cube.py -n 1 -m 3000 -o Output/craco_real1.csv --clobber -p ../Cubes/craco_real_cube.json

import argparse
import numpy as np
import subprocess

def main(pargs):
  start = pargs.start
  end = pargs.end
  nums = np.arange(start, end + 1, dtype="int")

  commands = []

  for number in nums:

      line = [
          "python",
          "../py/build_real_cube.py",
          "-n",
          f"{number}",
          "-m",
          "3000",
          "-o",
          f"Output/craco_real{number}.csv",
          "--clobber",
          "-p",
          f"../Cubes/craco_real_cube.json",
      ]

      commands.append(line)

  processes = []

  for command in commands:
      print(f"Running this command: {' '.join(command)}")
      pw = subprocess.Popen(command)
      processes.append(pw)

  for pw in processes:
      exit_code = pw.wait()
      print(exit_code)

  print("All done!")

def parse_option():
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        required=True,
        help="csv to start on",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        required=False,
        help="csv to end on (inclusive)",
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # get the argument of training.
    pargs = parse_option()
    main(pargs)