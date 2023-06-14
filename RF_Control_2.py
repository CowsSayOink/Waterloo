#Import ipython widgets
import json
import math
import os
from typing import List

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

# Set up the environment
import scipy.signal
from IPython.display import display

from ipywidgets import fixed, interact, interact_manual, interactive
from qcodes import Instrument

from qblox_instruments import Cluster, PlugAndPlay


# Scan for available devices and display
with PlugAndPlay() as p:
    # get info of all devices
    device_list = p.list_devices()

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
select_readout_module = select_module_widget(cluster, select_qrm_type=True, select_rf_type=True)


readout_module = select_readout_module.value
print(f"{readout_module} connected")

cluster.reset()
print(cluster.get_system_state())


def plot_spectrum(freq_sweep_range, I_data, Q_data):
    amplitude = np.sqrt(I_data**2 + Q_data**2)
    phase = np.arctan2(Q_data, I_data)*180/np.pi

    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(15, 7))

    ax1.plot(freq_sweep_range / 1e9, amplitude, color="#00839F", linewidth=2)
    ax1.set_ylabel("Amplitude (V)")

    ax2.plot(freq_sweep_range / 1e9, phase, color="#00839F", linewidth=2)
    ax2.set_ylabel("Phase ($\circ$)")
    ax2.set_xlabel("Frequency (GHz)")
    fig.tight_layout()
    plt.show()


# Parameters
no_averages = 10
integration_length = 1024
holdoff_length = 200
waveform_length = integration_length + holdoff_length




# Acquisitions
acquisitions = {"acq": {"num_bins": 1, "index": 0}}




seq_prog = f"""
      move    {no_averages},R0           # Average iterator.
      nop
      reset_ph
      set_awg_offs 10000, 10000          # set amplitude of signal
      nop
loop:
      wait     {holdoff_length}          # Wait time of flight
      acquire  0,0,{integration_length}  # Acquire data and store them in bin_n0 of acq_index.
      loop     R0,@loop                  # Run until number of average iterations is done.
      stop                               # Stop the sequencer
      """

# Add sequence to single dictionary and write to JSON file.
sequence = {
    "waveforms": {},
    "weights": {},
    "acquisitions": acquisitions,
    "program": seq_prog,
}
with open("sequence.json", "w", encoding="utf-8") as file:
    json.dump(sequence, file, indent=4)
    file.close()

# Upload sequence
readout_module.sequencer0.sequence("sequence.json")



readout_module.sequencer0.marker_ovr_en(True)
readout_module.sequencer0.marker_ovr_value(3)  # Enables output on QRM-RF

# Set offset in mV
readout_module.out0_offset_path0(5.5)
readout_module.out0_offset_path1(5.5)

# Configure scope mode
readout_module.scope_acq_sequencer_select(0)
readout_module.scope_acq_trigger_mode_path0("sequencer")
readout_module.scope_acq_trigger_mode_path1("sequencer")

# Configure the sequencer
readout_module.sequencer0.mod_en_awg(True)
readout_module.sequencer0.demod_en_acq(True)
readout_module.sequencer0.nco_freq(100e6)
readout_module.sequencer0.integration_length_acq(integration_length)
readout_module.sequencer0.sync_en(True)

# NCO delay compensation
readout_module.sequencer0.nco_prop_delay_comp_en(True)

lo_sweep_range = np.linspace(2e9, 18e9, 161)

lo_data_0 = []
lo_data_1 = []

for lo_val in lo_sweep_range:
    # Update the LO frequency.
    readout_module.out0_in0_lo_freq(lo_val)

    # Clear acquisitions
    readout_module.sequencer0.delete_acquisition_data("acq")

    readout_module.arm_sequencer(0)
    readout_module.start_sequencer()

    # Wait for the sequencer to stop with a timeout period of one minute.
    readout_module.get_acquisition_state(0, 1)

    # Move acquisition data from temporary memory to acquisition list.
    readout_module.store_scope_acquisition(0, "acq")

    # Get acquisition list from instrument.
    data = readout_module.get_acquisitions(0)["acq"]

    # Store the acquisition data.
    lo_data_0.append(data["acquisition"]["bins"]["integration"]["path0"][0])
    lo_data_1.append(data["acquisition"]["bins"]["integration"]["path1"][0])


# The result still needs to be divided by the integration length to make sure
# the units are correct.
lo_data_0 = np.asarray(lo_data_0) / integration_length
lo_data_1 = np.asarray(lo_data_1) / integration_length

# Plot amplitude/phase results
plot_spectrum(lo_sweep_range, lo_data_0, lo_data_1)


























