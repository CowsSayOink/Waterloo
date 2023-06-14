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
from scipy.fft import rfft, rfftfreq

from qblox_instruments import Cluster, PlugAndPlay, Pulsar


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
qrm = getattr(cluster, 'module4')  # Connect to the module that you have chosen above, module4 is the QRM address
#Module 4 is because readout is in the 4th doc in the physical hardware. if its in a different port, change the channel number


print(f"{available_slots[connect_qxm.value]} connected")
print(cluster.get_system_state())

# Waveforms
waveform_len = 1000
waveforms = {
    "gaussian": {
        "data": scipy.signal.gaussian(waveform_len, std=0.133 * waveform_len).tolist(),
        "index": 0,
    },
    "sine": {
        "data": [
            math.sin((2 * math.pi / waveform_len) * i) for i in range(0, waveform_len)
        ],
        "index": 1,
    },
    "sawtooth": {
        "data": [(1.0 / waveform_len) * i for i in range(0, waveform_len)],
        "index": 2,
    },
    "block": {"data": [1.0 for i in range(0, waveform_len)], "index": 3},
}






# plotting stuff
#
# time = numpy.arange(0, max(map(lambda d: len(d["data"]), waveforms.values())), 1)
# fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(10, 10 / 1.61))
#
# for wf, d in waveforms.items():
#     ax.plot(time[: len(d["data"])], d["data"], ".-", linewidth=0.5, label=wf)
#
# ax.legend(loc=4)
# ax.yaxis.grid()
# ax.xaxis.grid()
# ax.set_ylabel("Waveform primitive amplitude")
# ax.set_xlabel("Time (ns)")
#
# matplotlib.pyplot.draw()
# matplotlib.pyplot.show()



# Acquisitions
acquisitions = {"scope": {"num_bins": 1, "index": 0}}


# Number of sequencers per instrument
num_seq = 6

# Program
program = """
wait_sync 4
play      0,1,4
wait      140
acquire   0,0,16380
stop
"""

# Write sequence to file.
with open("sequence.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "waveforms": waveforms,
            "weights": waveforms,
            "acquisitions": acquisitions,
            "program": program,
        },
        file,
        indent=4,
    )
    file.close()

# Program sequencers.
for sequencer in qrm.sequencers:
    sequencer.sequence("sequence.json")


# Configure the sequencer to trigger the scope acquisition.
qrm.scope_acq_sequencer_select(0)
qrm.scope_acq_trigger_mode_path0("sequencer")
qrm.scope_acq_trigger_mode_path1("sequencer")

# Configure sequencer
qrm.sequencer0.sync_en(True)
qrm.sequencer0.gain_awg_path0(
    1.1 / num_seq
)  # The output range is 1.0 to -1.0, but we want to show what happens when the signals go out of range.
qrm.sequencer0.gain_awg_path1(1.1 / num_seq)
qrm.sequencer0.channel_map_path0_out0_en(True)
qrm.sequencer0.channel_map_path1_out1_en(True)

# Start the sequence
qrm.arm_sequencer(0)
qrm.start_sequencer()

# Wait for the sequencer to stop
qrm.get_acquisition_state(0, 1)

# Get acquisition data
qrm.store_scope_acquisition(0, "scope")
acq = qrm.get_acquisitions(0)

# Plot the results
fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(15, 15 / 2 / 1.61))

ax.plot(acq["scope"]["acquisition"]["scope"]["path0"]["data"][0:waveform_len])
ax.plot(acq["scope"]["acquisition"]["scope"]["path1"]["data"][0:waveform_len])
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Relative amplitude")

matplotlib.pyplot.show()
































