
# Import ipython widgets
import json
import math
import os

import ipywidgets as widgets
import matplotlib.pyplot
import numpy

# Set up the environment.
import scipy.signal
from IPython.display import display
from ipywidgets import fixed, interact, interact_manual, interactive

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
    "The following widget displays all the existing modules that are connected to your PC which includes \nthe Cluster module. Select the device you want to run the notebook on."
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
print("Select the QxM module from the available modules in your Cluster:")
display(connect_qxm)


# Connect to the cluster QxM module
module = connect_qxm.value
qxm = getattr(cluster, module)
print(f"{available_slots[connect_qxm.value]} connected")
print(cluster.get_system_state())


# Waveform parameters
waveform_length = 2000  # nanoseconds

# Waveform dictionary (data will hold the samples and index will be used to select the waveforms in the instrument).
waveforms = {
    "gaussian": {
        "data": scipy.signal.gaussian(
            waveform_length, std=0.12 * waveform_length
        ).tolist(),
        "index": 0,
    },
    "block": {"data": [1.0 for i in range(0, waveform_length)], "index": 1},
}


# Sequence program.
seq_prog = """
       move      100,R0   #Loop iterator.
       move      20,R1    #Initial wait period in ns.
       wait_sync 4        #Wait for sequencers to synchronize and then wait another 4ns.

loop:  set_mrk   1        #Set marker output 1.
       play      0,1,4    #Play a gaussian and a block on output path 0 and 1 respectively and wait 4ns.
       set_mrk   0        #Reset marker output 1.
       upd_param 16       #Update parameters and wait the remaining 16ns of the waveforms.

       wait      R1       #Wait period.

       play      1,0,20   #Play a block and a gaussian on output path 0 and 1 respectively and wait 20ns.
       wait      1000     #Wait a 1us in between iterations.
       add       R1,20,R1 #Increase wait period by 20ns.
       loop      R0,@loop #Subtract one from loop iterator.

       stop               #Stop the sequence after the last iteration.
"""

# Add sequence to single dictionary and write to JSON file.
sequence = {
    "waveforms": waveforms,
    "weights": {},
    "acquisitions": {},
    "program": seq_prog,
}
with open("sequence.json", "w", encoding="utf-8") as file:
    json.dump(sequence, file, indent=4)
    file.close()

# Upload sequence.
qxm.sequencer0.sequence("sequence.json")
qxm.sequencer1.sequence("sequence.json")


# Configure the sequencers to synchronize.
qxm.sequencer0.sync_en(True)
qxm.sequencer1.sync_en(True)

# Map sequencers to specific outputs (but first disable all sequencer connections).
for sequencer in qxm.sequencers:
    for out in range(0, 4):
        if hasattr(sequencer, "channel_map_path{}_out{}_en".format(out % 2, out)):
            sequencer.set("channel_map_path{}_out{}_en".format(out % 2, out), False)

# If it is a QRM, we only map sequencer 0 to the outputs.
qxm.sequencer0.channel_map_path0_out0_en(True)
qxm.sequencer0.channel_map_path1_out1_en(True)
if qxm.is_qcm_type:
    qxm.sequencer1.channel_map_path0_out2_en(True)
    qxm.sequencer1.channel_map_path1_out3_en(True)


# Arm and start both sequencers.
qxm.arm_sequencer(0)
qxm.arm_sequencer(1)
qxm.start_sequencer()

# Print status of both sequencers.
print(qxm.get_sequencer_state(0))
print(qxm.get_sequencer_state(1))



# Stop both sequencers.
qxm.stop_sequencer()

# Print status of both sequencers (should now say it is stopped).
print(qxm.get_sequencer_state(0))
print(qxm.get_sequencer_state(1))
print()

# Uncomment the following to print an overview of the instrument parameters.
# Print an overview of the instrument parameters.
# print("Snapshot:")
# qxm.print_readable_snapshot(update=True)

# Close the instrument connection.
Cluster.close_all()





















