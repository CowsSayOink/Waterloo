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
from qcodes import Instrument
from qblox_instruments import Cluster, PlugAndPlay



with PlugAndPlay() as p:
    # get info of all devices
    device_list = p.list_devices()

# Scan for available devices and display
names = {
    dev_id: dev_info["description"]["name"] for dev_id, dev_info in device_list.items()
}
ip_addresses = {
    dev_id: dev_info["identity"]["ip"] for dev_id, dev_info in device_list.items()
}

# create widget for names and ip addresses
connect = widgets.Dropdown(
    options=[(names[dev_id] + " @" + ip_addresses[dev_id], dev_id)
             for dev_id in device_list.keys()],
    description="Select Device",
)
display(connect)



# Connect to device
dev_id = connect.value
# Close the chosen QCodes instrument as to prevent name clash.
try:
    Instrument.find_instrument(names[dev_id]).close()
except KeyError:
    pass

print(ip_addresses)
print(dev_id)
cluster = Cluster(name=names[dev_id], identifier=ip_addresses[dev_id])

print(f"{connect.label} connected")
print(cluster.get_system_state())



def select_module_widget(device, select_all=False, select_qrm_type: bool=True, select_rf_type: bool=False):
    """Create a widget to select modules of a certain type

    default is to show only QRM baseband

    Args:
        devices : Cluster we are currently using
        select_all (bool): ignore filters and show all modules
        select_qrm_type (bool): filter QRM/QCM
        select_rf_type (bool): filter RF/baseband
    """
    options = [[None, None]]


    for module in device.modules:
        if module.present():
            if select_all or (module.is_qrm_type == select_qrm_type and module.is_rf_type == select_rf_type):
                options.append(
                    [
                        f"{device.name} "
                        f"{module.short_name} "
                        f"({module.module_type}{'_RF' if module.is_rf_type else ''})",
                        module,
                    ]
                )
    widget = widgets.Dropdown(options=options)
    display(widget)

    return widget


print("Select the readout module from the available modules:")
select_qrm = select_module_widget(cluster, select_qrm_type=True, select_rf_type=False)


readout_module = select_qrm.value
print(f"{readout_module} connected")


# Waveform parameters
waveform_length = 120  # nanoseconds

# Waveform dictionary (data will hold the samples and index will be used to select the waveforms in the instrument).
waveforms = {
    "gaussian": {
        "data": scipy.signal.gaussian(
            waveform_length, std=0.12 * waveform_length
        ).tolist(),
        "index": 0,
    },
    "sine": {
        "data": [
            math.sin((2 * math.pi / waveform_length) * i)
            for i in range(0, waveform_length)
        ],
        "index": 1,
    },
}



#plotting:
# time = numpy.arange(0, max(map(lambda d: len(d["data"]), waveforms.values())), 1)
# fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(10, 10 / 1.61))
#
# for wf, d in waveforms.items():
#     ax.plot(time[: len(d["data"])], d["data"], ".-", linewidth=0.5, label=wf)
#
# ax.legend(loc=4)
# ax.yaxis.grid()
# ax.xaxis.grid()
# ax.set_ylabel("Waveform primitive amplitude", fontsize = 20)
# ax.set_xlabel("Time (ns)", fontsize = 20)
#
# matplotlib.pyplot.draw()
# matplotlib.pyplot.show()



# Acquisitions
acquisitions = {
    "single": {"num_bins": 1, "index": 0},
    "multiple_0": {"num_bins": 1, "index": 1},
    "multiple_1": {"num_bins": 1, "index": 2},
    "multiple_2": {"num_bins": 1, "index": 3},
    "avg": {"num_bins": 1, "index": 4},
}

# Sequence program.
seq_prog = """
play    0,1,4     #Play waveforms and wait 4ns. Parameters: waveform index (from dict) on path 0, waveform index (from dict) on path 1, wait (in ns)
acquire 0,0,16380 #Acquire waveforms and wait remaining duration of scope acquisition. Parameters: acquisition index (from dict), data bin, Duration of acq (in ns)
stop              #Stop.
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


# Upload sequence.
readout_module.sequencer0.sequence("sequence.json")

# Configure the sequencer to trigger the scope acquisition.
readout_module.scope_acq_sequencer_select(0)
readout_module.scope_acq_trigger_mode_path0("sequencer")
readout_module.scope_acq_trigger_mode_path1("sequencer")

# Map sequencer to specific outputs (but first disable all sequencer connections)
for sequencer in readout_module.sequencers:
    for out in range(0, 2):
        sequencer.set("channel_map_path{}_out{}_en".format(out % 2, out), False)
readout_module.sequencer0.channel_map_path0_out0_en(True)
readout_module.sequencer0.channel_map_path1_out1_en(True)

# Arm and start sequencer.
readout_module.arm_sequencer(0)
readout_module.start_sequencer()

# Print status of sequencer.
print("Status:")
print(readout_module.get_sequencer_state(0))


# Wait for the acquisition to finish with a timeout period of one minute.
readout_module.get_acquisition_state(0, 1)

# Move acquisition data from temporary memory to acquisition list.
readout_module.store_scope_acquisition(0, "single")

# Get acquisition list from instrument.
single_acq = readout_module.get_acquisitions(0)


# Plot acquired signal on both inputs.
fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(15, 15 / 2 / 1.61))
ax.plot(single_acq["single"]["acquisition"]["scope"]["path0"]["data"][130:290])
ax.plot(single_acq["single"]["acquisition"]["scope"]["path1"]["data"][130:290])
ax.set_xlabel("Time (ns)", fontsize = 20)
ax.set_ylabel("Relative amplitude", fontsize = 20)
matplotlib.pyplot.show()










