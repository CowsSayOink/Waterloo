import math
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import random

# Set up the environment.
import scipy.signal
from IPython.display import display
import ipywidgets as widgets

from qblox_instruments import Cluster, PlugAndPlay




# Scan for available devices and display
with PlugAndPlay() as p:
    # get info of all devices
    device_list = p.list_devices()
    device_keys = list(device_list.keys())

# create widget for names and ip addresses
connect = widgets.Dropdown(
    options=[(device_list[key]["description"]["name"]) for key in device_list.keys()],
    description="Select Device",
)
print(
    "The following widget displays all the existing modules that are connected to your PC which includes the Pulsar modules as well as a Cluster. Select the device you want to run the notebook on."
)
display(connect)



# close all previous connections to the cluster
Cluster.close_all()

# Retrieve device name and IP address
device_name = connect.value
device_number = connect.options.index(device_name)
ip_address = device_list[device_keys[device_number]]["identity"]["ip"]

# connect to the cluster and reset
cluster = Cluster(device_name, ip_address)
cluster.reset()
print(f"{device_name} connected at {ip_address}")



# Find all QRM/QCM modules
available_slots = {}
for module in cluster.modules:
    # if module is currently present in stack
    if cluster._get_modules_present(module.slot_idx):
        # check if QxM is RF or baseband
        if module.is_rf_type:
            available_slots[f"module{module.slot_idx}"] = ["QCM-RF", "QRM-RF"][
                module.is_qrm_type
            ]
        else:
            available_slots[f"module{module.slot_idx}"] = ["QCM", "QRM"][
                module.is_qrm_type
            ]

# List of all QxM modules present
connect_qxm = widgets.Dropdown(options=[key for key in available_slots.keys()])

print(available_slots)
# display widget with cluster modules
print()
print("Select the QRM module from the available modules in your Cluster:")
display(connect_qxm)


# Connect to the cluster QRM
qrm = getattr(cluster, "module4")  # Connect to the module that you have chosen above
print("connected")
print(cluster.get_system_state())



# Waveform length parameter
waveform_length = 16  # nanoseconds

waveforms= {
     "block": {"data": [0.5 for i in range(0, waveform_length)], "index": 0},
}



# Acquisitions
acquisitions = {
    "ttl":   {"num_bins": 100, "index": 0},
}


# Sequence program for AWG.
seq_prog_awg = """
           wait_sync 4        #Wait for sequencers to synchronize and then wait another 4ns.
           move      5,R0     #Loop iterator.
loop:
           play      0,0,16   #Play a block on output path 0 and wait 16ns.
           wait      984      #Wait 984ns
           loop      R0, @loop #Repeat loop until R0 is 0

           stop               #Stop the sequence after the last iteration.
"""

# Sequence program for aqcuiring
seq_prog_acq = """
        wait_sync 4           #Wait for sequencers to synchronize and then wait another 4ns.
        wait 140              #Approximate time of flight
        acquire_ttl 0,0,1,4   #Turn on TTL acquire on input path 0 and wait 4ns.
        wait 6000             #Wait 6000ns.
        acquire_ttl 0,0,0,4   #Turn off TTL acquire on input path 0 and wait 4ns.

        stop                   #Stop sequencer.
"""


# Add sequence program, waveform and acquistitions to single dictionary.
sequence_awg = {
    "waveforms": waveforms,
    "weights": {},
    "acquisitions": {},
    "program": seq_prog_awg,
    }
sequence_acq= {
       "waveforms": {},
       "weights": {},
       "acquisitions": acquisitions,
       "program": seq_prog_acq,
   }


# Upload sequence.
qrm.sequencer0.sequence(sequence_awg)
qrm.sequencer1.sequence(sequence_acq)




# Map sequencer to specific outputs (but first disable all sequencer connections)
for sequencer in qrm.sequencers:
    for out in range(0, 2):
        sequencer.set("channel_map_path{}_out{}_en".format(out % 2, out), False)
qrm.sequencer0.channel_map_path0_out0_en(True)
qrm.sequencer0.channel_map_path1_out1_en(True)

# Enable sync
qrm.sequencer0.sync_en(True)
qrm.sequencer1.sync_en(True)

# Delete previous acquisition.
qrm.delete_acquisition_data(1, "ttl")

# Configure scope mode
qrm.scope_acq_sequencer_select(1)

# Choose threshold and input gain
threshold = 0.5
input_gain = 0

# Configure the TTL acquisition
qrm.sequencer1.ttl_acq_input_select(0)
qrm.sequencer1.ttl_acq_auto_bin_incr_en(False)

#Set input gain and threshold
qrm.in0_gain(input_gain)
qrm.sequencer1.ttl_acq_threshold(threshold)


# Arm and start sequencer.
qrm.arm_sequencer(0)
qrm.arm_sequencer(1)
qrm.start_sequencer()

# Print status of sequencer.
print(qrm.get_sequencer_state(0, 1))
print(qrm.get_sequencer_state(1, 1))









