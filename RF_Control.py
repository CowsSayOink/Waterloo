# Import ipython widgets
import json
import math
import os

import ipywidgets as widgets
import matplotlib.pyplot
import numpy as np

# Set up the environment.
import scipy.signal
from IPython.display import display
from ipywidgets import fixed, interact, interact_manual, interactive

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
    "The following widget displays all the existing modules that are connected to your PC. Select the device you want to run the notebook on."
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
connect_qrm_rf = widgets.Dropdown(options=[key for key in available_slots.keys()])

print(available_slots)
# display widget with cluster modules
print()
print("Select the QRM-RF module from the available modules in your Cluster:")
display(connect_qrm_rf)


# Connect to the cluster QxM module
module = connect_qrm_rf.value
qrm_rf = getattr(cluster, 'module4')
print(f"{available_slots[connect_qrm_rf.value]} connected")

print(cluster.get_system_state())



# Parameters
no_averages = 10
integration_length = 1000
holdoff_length = 200
waveform_length = integration_length + holdoff_length

# Create DC waveform
waveforms = {"dc": {"data": [0.5 for i in range(0, waveform_length)], "index": 0}}


# Acquisitions
acquisitions = {"acq": {"num_bins": 1, "index": 0}}


# Sequence program.
seq_prog = """
      wait_sync 4
      move    0,R1         #Average iterator.
      nop

loop: play    0,0,4
      wait    {}           #Wait the hold-off time
      acquire 0,0,{}       #Acquire bins and store them in "avg" acquisition.
      add     R1,1,R1      #Increment avg iterator
      nop                  #Wait a cycle for R1 to be available.
      jlt     R1,{},@loop  #Run until number of average iterations is done.

      stop                 #Stop.
""".format(
    holdoff_length, integration_length, no_averages
)

# Add sequence to single dictionary and write to JSON file.
sequence = {
    "waveforms": waveforms,
    "weights": {},
    "acquisitions": acquisitions,
    "program": seq_prog,
}

'''
with open("sequence.json", "w", encoding="utf-8") as file:
    json.dump(sequence, file, indent=4)
    file.close()

qrm_rf.sequencer0.marker_ovr_en(True)
qrm_rf.sequencer0.marker_ovr_value(15)  # Enables output on QRM-RF

qrm_rf.out0_offset_path0(5.5)
qrm_rf.out0_offset_path1(5.5)

# Configure scope mode
qrm_rf.scope_acq_sequencer_select(0)
qrm_rf.scope_acq_trigger_mode_path0("sequencer")
qrm_rf.scope_acq_trigger_mode_path1("sequencer")

# Configure the sequencer
qrm_rf.sequencer0.mod_en_awg(True)
qrm_rf.sequencer0.demod_en_acq(True)
qrm_rf.sequencer0.nco_freq(100e6)
qrm_rf.sequencer0.integration_length_acq(integration_length)
qrm_rf.sequencer0.sync_en(True)

lo_sweep_range = np.linspace(2e9, 18e9, 200)

lo_data_0 = []
lo_data_1 = []

for lo_val in lo_sweep_range:
    # Update the LO frequency.
    qrm_rf.out0_in0_lo_freq(lo_val)

    # Upload sequence. This clears the acquisitions.
    qrm_rf.sequencer0.sequence("sequence.json")

    qrm_rf.arm_sequencer(0)
    qrm_rf.start_sequencer()

    # Wait for the sequencer to stop with a timeout period of one minute.
    qrm_rf.get_acquisition_state(0, 1)

    # Move acquisition data from temporary memory to acquisition list.
    qrm_rf.store_scope_acquisition(0, "acq")

    # Get acquisition list from instrument.
    data = qrm_rf.get_acquisitions(0)["acq"]

    # Store the acquisition data.
    lo_data_0.append(data["acquisition"]["bins"]["integration"]["path0"][0])
    lo_data_1.append(data["acquisition"]["bins"]["integration"]["path1"][0])

# The result still needs to be divided by the integration length to make sure
# the units are correct.
lo_data_0 = np.asarray(lo_data_0) / integration_length
lo_data_1 = np.asarray(lo_data_1) / integration_length


amplitude = np.sqrt(lo_data_0**2 + lo_data_1**2)
phase = np.arctan2(lo_data_1, lo_data_0)

fig, [ax1, ax2] = matplotlib.pyplot.subplots(2, 1, sharex=True, figsize=(15, 8))

ax1.plot(lo_sweep_range / 1e9, amplitude)
ax1.grid(True)
ax1.set_ylabel("Amplitude")

ax2.plot(lo_sweep_range / 1e9, phase)
ax2.grid(True)
ax2.set_ylabel("Phase (deg)")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_xticks(np.arange(1, 10, 1))
matplotlib.pyplot.show()
'''









cluster.reset()

# Sequence program.
seq_prog = """
      wait_sync 4

loop: play    0,0,1200
      jmp     @loop
"""

# Add sequence to single dictionary and write to JSON file.
sequence = {
    "waveforms": waveforms,
    "weights": {},
    "acquisitions": acquisitions,
    "program": seq_prog,
}
with open("sequence.json", "w", encoding="utf-8") as file:
    json.dump(sequence, file, indent=4)
    file.close()

qrm_rf.sequencer0.sequence("sequence.json")

# Configure the Local oscillator
qrm_rf.out0_in0_lo_freq(5e9 - 300e6)

qrm_rf.sequencer0.marker_ovr_en(True)
qrm_rf.sequencer0.marker_ovr_value(15)  # Enables output on QRM-RF

# Configure the sequencer
qrm_rf.sequencer0.mod_en_awg(True)
qrm_rf.sequencer0.nco_freq(300e6)
qrm_rf.sequencer0.sync_en(True)

qrm_rf.arm_sequencer(0)
qrm_rf.start_sequencer(0)

print("Status:")
print(qrm_rf.get_sequencer_state(0))

print('We need a spectrum analyser to see the peaks, centered at 4.7GHz')


def set_offset0(offset0):
    qrm_rf.out0_offset_path0(offset0)


def set_offset1(offset1):
    qrm_rf.out0_offset_path1(offset1)


def set_gain_ratio(gain_ratio):
    qrm_rf.sequencer0.mixer_corr_gain_ratio(gain_ratio)
    # Start
    qrm_rf.arm_sequencer(0)
    qrm_rf.start_sequencer(0)


def set_phase_offset(phase_offset):
    qrm_rf.sequencer0.mixer_corr_phase_offset_degree(phase_offset)
    # Start
    qrm_rf.arm_sequencer(0)
    qrm_rf.start_sequencer(0)


interact(
    set_offset0,
    offset0=widgets.FloatSlider(
        min=-14.0,
        max=14.0,
        step=0.001,
        start=0.0,
        layout=widgets.Layout(width="1200px"),
    ),
)
interact(
    set_offset1,
    offset1=widgets.FloatSlider(
        min=-14.0,
        max=14.0,
        step=0.001,
        start=0.0,
        layout=widgets.Layout(width="1200px"),
    ),
)
interact(
    set_gain_ratio,
    gain_ratio=widgets.FloatSlider(
        min=0.9, max=1.1, step=0.001, start=1.0, layout=widgets.Layout(width="1200px")
    ),
)
interact(
    set_phase_offset,
    phase_offset=widgets.FloatSlider(
        min=-45.0,
        max=45.0,
        step=0.001,
        start=0.0,
        layout=widgets.Layout(width="1200px"),
    ),
)







# Stop sequencer.
qrm_rf.stop_sequencer()

# Print status of sequencer.
print(qrm_rf.get_sequencer_state(0))
print()

# Uncomment the following to print an overview of the instrument parameters.
# Print an overview of the instrument parameters.
# print("Snapshot:")
# cluster.print_readable_snapshot(update=True)

# Close the instrument connection.
cluster.close()















