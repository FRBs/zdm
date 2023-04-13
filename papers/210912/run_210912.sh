#!/bin/bash

SNR=31.7

#command="python pz_given_dm.py -d 1234. -i 30.9 -s CRAFT/ICS -H 50 -o 210912_pzgdm -S $SNR"

command="python pz_given_dm.py -d 1458 -i 31.0 -s CRAFT/ICS -H 50 -o 220610_pzgdm -z 1.01"

echo $command
$command
