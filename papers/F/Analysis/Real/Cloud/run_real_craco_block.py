# Running this command: python ../py/build_real_cube.py -n 1 -m 3000 -o Output/craco_real1.csv --clobber -p ../Cubes/craco_real_cube.json

import numpy as np
import subprocess

start = 1
end = 41
nums = np.arange(start, end + 1, dtype="int")

commands = []

for number in nums:
    line = f"python ../py/build_real_cube.py -n {number} -m 3000 -o Output/craco_real{number}.csv --clobber -p ../Cubes/craco_real_cube.json"
    commands.append(line)

processes = []

for command in commands:
    print(f"Running this command: {' '.join(command)}")
    pw = subprocess.Popen(command)
    processes.append(pw)

for pw in processes:
    exit_code = pw.wait()
    print(exit_code)

