#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:44:34 2026

@author: Peter
"""
import matplotlib.pyplot as plt
from numpy import interp
import numpy as np
import json


with open("measurementsOUTPUT.json", "r") as msmfile:
    measurementdict = json.load(msmfile) 
with open("opticsOUTPUT.json", "r") as optcfile:
    opticsdict=json.load(optcfile)

SCL_DICTIONARY = {
    'SCL:Cav01': ['SCL:Cav01a', 'SCL:Cav01b', 'SCL:Cav01c'], 
    'SCL:Cav02': ['SCL:Cav02a', 'SCL:Cav02b', 'SCL:Cav02c'], 
    'SCL:Cav03': ['SCL:Cav03a', 'SCL:Cav03b', 'SCL:Cav03c'], 
    'SCL:Cav04': ['SCL:Cav04a', 'SCL:Cav04b', 'SCL:Cav04c'], 
    'SCL:Cav05': ['SCL:Cav05a', 'SCL:Cav05b', 'SCL:Cav05c'], 
    'SCL:Cav06': ['SCL:Cav06a', 'SCL:Cav06b', 'SCL:Cav06c'], 
    'SCL:Cav07': ['SCL:Cav07a', 'SCL:Cav07b', 'SCL:Cav07c'], 
    'SCL:Cav08': ['SCL:Cav08a', 'SCL:Cav08b', 'SCL:Cav08c'], 
    'SCL:Cav09': ['SCL:Cav09a', 'SCL:Cav09b', 'SCL:Cav09c'], 
    'SCL:Cav10': ['SCL:Cav10a', 'SCL:Cav10b', 'SCL:Cav10c'], 
    'SCL:Cav11': ['SCL:Cav11a', 'SCL:Cav11b', 'SCL:Cav11c'],
    'SCL:Cav12': ['SCL:Cav12a', 'SCL:Cav12b', 'SCL:Cav12c', 'SCL:Cav12d'], 
    'SCL:Cav13': ['SCL:Cav13a', 'SCL:Cav13b', 'SCL:Cav13c', 'SCL:Cav13d'], 
    'SCL:Cav14': ['SCL:Cav14a', 'SCL:Cav14b', 'SCL:Cav14c', 'SCL:Cav14d'], 
    'SCL:Cav15': ['SCL:Cav15a', 'SCL:Cav15b', 'SCL:Cav15c', 'SCL:Cav15d'], 
    'SCL:Cav16': ['SCL:Cav16a', 'SCL:Cav16b', 'SCL:Cav16c', 'SCL:Cav16d'], 
    'SCL:Cav17': ['SCL:Cav17a', 'SCL:Cav17b', 'SCL:Cav17c', 'SCL:Cav17d'], 
    'SCL:Cav18': ['SCL:Cav18a', 'SCL:Cav18b', 'SCL:Cav18c', 'SCL:Cav18d'], 
    'SCL:Cav19': ['SCL:Cav19a', 'SCL:Cav19b', 'SCL:Cav19c', 'SCL:Cav19d'], 
    'SCL:Cav20': ['SCL:Cav20a', 'SCL:Cav20b', 'SCL:Cav20c', 'SCL:Cav20d'], 
    'SCL:Cav21': ['SCL:Cav21a', 'SCL:Cav21b', 'SCL:Cav21c', 'SCL:Cav21d'], 
    'SCL:Cav22': ['SCL:Cav22a', 'SCL:Cav22b', 'SCL:Cav22c', 'SCL:Cav22d'], 
    'SCL:Cav23': ['SCL:Cav23a', 'SCL:Cav23b', 'SCL:Cav23c', 'SCL:Cav23d'],
    
    'SCL:Cav25': ['SCL:Cav25a', 'SCL:Cav25b', 'SCL:Cav25c', 'SCL:Cav25d'],
    
    'SCL:Cav27': ['SCL:Cav27a', 'SCL:Cav27b', 'SCL:Cav27c', 'SCL:Cav27d'],
    'SCL:Cav28': ['SCL:Cav28a', 'SCL:Cav28b', 'SCL:Cav28c', 'SCL:Cav28d'],
    'SCL:Cav29': ['SCL:Cav29a', 'SCL:Cav29b', 'SCL:Cav29c', 'SCL:Cav29d'],
    'SCL:Cav30': ['SCL:Cav30a', 'SCL:Cav30b', 'SCL:Cav30c', 'SCL:Cav30d'],
    'SCL:Cav31': ['SCL:Cav31a', 'SCL:Cav31b', 'SCL:Cav31c', 'SCL:Cav31d'],
    'SCL:Cav32': ['SCL:Cav32a', 'SCL:Cav32b', 'SCL:Cav32c', 'SCL:Cav32d']
} 
#Not sure why these are missing
#'SCL:Cav24': ['SCL:Cav24a', 'SCL:Cav24b', 'SCL:Cav24c', 'SCL:Cav24d'], 'SCL:Cav26': ['SCL:Cav26a', 'SCL:Cav26b', 'SCL:Cav26c', 'SCL:Cav26d'],

scl_cav_params = {}
for cavity in opticsdict:
    if "SCL:Cav" in cavity:
        scl_cav_params = scl_cav_params | {cavity: {'phase': opticsdict[cavity]['phase'], 'amp': opticsdict[cavity]['amp']}}



bpm_list = []
phase_list = []
for diag, meas in measurementdict.items():
    if 'SCL_Diag:BPM' in diag:
        if "23" in diag:
            pass
        elif "25" in diag:
            pass
        else:
            bpm_list.append(diag)
            phase_list.append(meas["phi_avg"])


cavitysetupdict = {}
indexx = 1 # start at bpm00b not a
for cryo in SCL_DICTIONARY:
    for cavity in SCL_DICTIONARY[cryo]:
        difference = scl_cav_params[cavity]['phase'] - 2* phase_list[indexx] + np.pi*1.5
        difference = (difference + np.pi) % (2 * np.pi) - np.pi
        
        cavitysetupdict = cavitysetupdict | {cavity: {"name": cavity, 
                                            "Q_L": 7.0e5, "R_over_Q":483.0, "amp":15e6,
                                            "phase": round(np.degrees(scl_cav_params[cavity]['phase']), 2),
                                            "BPM_ref": {bpm_list[indexx]:difference}}}

    indexx+=1


# Define the filename
filename = 'cavityparameters.json'

# Write the data to the file
with open(filename, 'w') as json_file:
    json.dump(cavitysetupdict, json_file)

print(f"Successfully created {filename}")



