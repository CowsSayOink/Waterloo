import json
import warnings
from typing import Any, Callable, Dict

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from qblox_instruments import Cluster, ClusterType, PlugAndPlay
from qblox_instruments.ieee488_2 import DummyBinnedAcquisitionData
from quantify_core.analysis import base_analysis as ba
from quantify_core.measurement import MeasurementControl
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.instrument_monitor import InstrumentMonitor
from quantify_core.visualization.pyqt_plotmon import \
    PlotMonitor_pyqt as PlotMonitor
from quantify_core.visualization.SI_utilities import format_value_string
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import \
    BasicTransmonElement
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import \
    ClusterComponent

from qblox_instruments.ieee488_2 import DummyBinnedAcquisitionData
from quantify_core.data.handling import set_datadir
# warnings.simplefilter("always")


import warnings
import matplotlib.pyplot as plt
from scipy.signal import detrend
import ipywidgets as widgets
import numpy as np

from qcodes.instrument.parameter import ManualParameter
from quantify_scheduler.gettables import ScheduleGettable

from quantify_core.analysis.spectroscopy_analysis import ResonatorSpectroscopyAnalysis

from quantify_core.analysis.single_qubit_timedomain import (
    AllXYAnalysis,
   EchoAnalysis,
   RabiAnalysis,
   RamseyAnalysis,
   T1Analysis,
)

from quantify_scheduler.schedules import (
    heterodyne_spec_sched,
    two_tone_spec_sched,
    allxy_sched,
    echo_sched,
    rabi_sched,
    ramsey_sched,
    t1_sched,
    readout_calibration_sched
)
from quantify_scheduler.schedules.spectroscopy_schedules import heterodyne_spec_sched, two_tone_spec_sched

from quantify_core.data.handling import set_datadir

set_datadir("quantify_data")
# warnings.simplefilter("always")




#import ipynb #I dont think well need this!

from hello_world import qubit_0, measurement_control, transmon_chip, cluster
from hello_world import compiler
from hello_world import heterodyne_spec_kwargs, two_tone_spec_kwargs, rabi_kwargs
from hello_world import set_dummy_data_rabi, clear_dummy_data, heterodyne_spec_sched_with_dummy
from hello_world import QubitSpectroscopyAnalysis




np.asarray([qubit_0.clock_freqs.readout()])




def kwarg_wrapper(func: Callable[[BasicTransmonElement], Dict[str, Any]]) -> Callable:
    def inner(qubit: BasicTransmonElement, **kwargs):
        default = func(qubit)
        for key, value in kwargs.items():
            if key in default:
                default[key] = value
            else:
                raise RuntimeError
        return default

    return inner


@kwarg_wrapper
def nco_heterodyne_spec_kwargs(qubit: BasicTransmonElement) -> Dict[str, Any]:
    return {
        "pulse_amp": qubit.measure.pulse_amp(),
        "pulse_duration": qubit.measure.pulse_duration(),
        "frequencies": np.asarray([qubit.clock_freqs.readout()]),
        "acquisition_delay": qubit.measure.acq_delay(),
        "integration_time": qubit.measure.integration_time(),
        "init_duration": qubit.reset.duration(),
        "port": qubit.ports.readout(),
        "clock": qubit.name + ".ro",
    }


@kwarg_wrapper
def heterodyne_spec_kwargs(qubit: BasicTransmonElement) -> Dict[str, Any]:
    return {
        "pulse_amp": qubit.measure.pulse_amp(),
        "pulse_duration": qubit.measure.pulse_duration(),
        "frequency": qubit.clock_freqs.readout(),
        "acquisition_delay": qubit.measure.acq_delay(),
        "integration_time": qubit.measure.integration_time(),
        "init_duration": qubit.reset.duration(),
        "port": qubit.ports.readout(),
        "clock": qubit.name + ".ro",
    }


@kwarg_wrapper
def nco_two_tone_spec_kwargs(qubit: BasicTransmonElement, **kwargs) -> Dict[str, Any]:
    return {
        "spec_pulse_amp": 0.001,
        "spec_pulse_duration": 4e-6,
        "spec_pulse_frequencies": qubit.clock_freqs.f01(),
        "spec_pulse_port": qubit.ports.microwave(),
        "spec_pulse_clock": qubit.name + ".01",
        "ro_pulse_amp": qubit.measure.pulse_amp(),
        "ro_pulse_duration": qubit.measure.pulse_duration(),
        "ro_pulse_delay": 200e-9,
        "ro_pulse_port": qubit.ports.readout(),
        "ro_pulse_clock": qubit.name + ".ro",
        "ro_pulse_frequency": qubit.clock_freqs.readout(),
        "ro_acquisition_delay": qubit.measure.acq_delay(),
        "ro_integration_time": qubit.measure.integration_time(),
        "init_duration": qubit.reset.duration(),
    }



@kwarg_wrapper
def two_tone_spec_kwargs(qubit: BasicTransmonElement, **kwargs) -> Dict[str, Any]:
    return {
        "spec_pulse_amp": 0.001,
        "spec_pulse_duration": 4e-6,
        "spec_pulse_frequency": qubit.clock_freqs.f01(),
        "spec_pulse_port": qubit.ports.microwave(),
        "spec_pulse_clock": qubit.name + ".01",
        "ro_pulse_amp": qubit.measure.pulse_amp(),
        "ro_pulse_duration": qubit.measure.pulse_duration(),
        "ro_pulse_delay": 200e-9,
        "ro_pulse_port": qubit.ports.readout(),
        "ro_pulse_clock": qubit.name + ".ro",
        "ro_pulse_frequency": qubit.clock_freqs.readout(),
        "ro_acquisition_delay": qubit.measure.acq_delay(),
        "ro_integration_time": qubit.measure.integration_time(),
        "init_duration": qubit.reset.duration(),
    }


@kwarg_wrapper
def rabi_kwargs(qubit: BasicTransmonElement) -> Dict[str, Any]:
    return {
        "pulse_amp": qubit.rxy.amp180(),
        "pulse_duration": qubit.rxy.duration(),
        "frequency": qubit.clock_freqs.f01(),
        "qubit": qubit.name,
    }



'''
Resonator spectroscopy
The very first experiment for tuning a superconducting qubit is to find the resonance frequency of the readout resonator. 
As we will be sweeping the frequency, we first define a qcodes `Parameter` as placeholder, which will be filled with the sweep values during the experiment.
'''



#This compensates for electrical delay

cluster.module4.sequencer0.nco_prop_delay_comp_en(True)
cluster.module4.sequencer0.nco_prop_delay_comp(50)

transmon_chip.cfg_sched_repetitions(2048)

for att in np.arange(20, 4, -8):
    print(att)
    config = transmon_chip.hardware_config()
    config["cluster"]["cluster_module4"]["complex_output_0"]["output_att"] = att
    transmon_chip.hardware_config(config)

    qubit_0.measure.pulse_amp(0.25)

    freq = ManualParameter(name="frequency", unit="Hz", label="f")
    freq.batched = True

    measurement_control.settables(freq)
    measurement_control.setpoints(np.linspace(6.74e9, 6.76e9, 2001))
    gettable = ScheduleGettable(
        quantum_device=transmon_chip,
        schedule_function=heterodyne_spec_sched,
        schedule_kwargs=nco_heterodyne_spec_kwargs(qubit_0, frequencies=freq),
        real_imag=False,
        batched=True
    )
    measurement_control.gettables(gettable)

    res_spec_dset = measurement_control.run("ResonatorSpectroscopy")
    res_spec_result = ResonatorSpectroscopyAnalysis(
        dataset=res_spec_dset,
    ).run()
    plt.plot(res_spec_result.dataset_processed.x0, detrend(np.unwrap(np.angle(res_spec_result.dataset_processed.S21))))
    res_spec_result.display_figs_mpl()
    plt.show()


qubit_0.clock_freqs.readout(6.751e9)
config = transmon_chip.hardware_config()
config["cluster"]["cluster_module4"]["complex_output_0"]["output_att"] = 56
transmon_chip.hardware_config(config)
freq = ManualParameter(name="frequency", unit="Hz", label="f")
freq.batched = True



#Next, we need to set up MeasurementControl to use this Parameter. First, we set that the next measurement will be sweeping over the frequency:

measurement_control.settables(freq)

#We also need to set the actual values for the experiment. Here from 2-10 GHz, in 3 steps.

measurement_control.setpoints(np.linspace(6.75e9, 6.753e9, 300))

# We also need to define the actual measurement. To this end, we use a schedule function. This is simply a function that generates
# a schedule for a given frequency. As Resonator spectroscopy is a standard experiment, we can simply import the function from quantify.
# We will have a look at custom schedule functions later.

import inspect
heterodyne_spec_function = heterodyne_spec_sched
print(inspect.getsource(heterodyne_spec_sched))

# The schedule function itself is still an abstract object, it does not refer to the hardware config (e.g. cabling, ip addresses) yet.
# To fully define the measurements, we define a `ScheduleGettable`, which fully describes the experiment.

measurement_control.settables(freq)
gettable = ScheduleGettable(
    quantum_device=transmon_chip,
    schedule_function=heterodyne_spec_function,
    schedule_kwargs=nco_heterodyne_spec_kwargs(qubit_0, frequencies=freq),
    real_imag=False,
    batched=True
)


#We also need to connect the ScheduleGettable to MeasurementControl

measurement_control.gettables(gettable)
heterodyne_spec_function(**nco_heterodyne_spec_kwargs(qubit_0)).plot_circuit_diagram()
compiled = compiler.compile(heterodyne_spec_function(**nco_heterodyne_spec_kwargs(qubit_0)))
compiled.plot_pulse_diagram()

#Finally, we execute the experiment using `MeasurementControl`. The data is automatically saved, and also available immediately as `xarray.Dataset`

res_spec_dset = measurement_control.run("ResonatorSpectroscopy")

# The information we are interested in (the resonance frequency) is not immediately accessible, we need to analyze the data first.
# To this end, we can use an analysis class based on the `lmfit` package.
# For standard experiments like resonator spectroscopy, it is already provided by quantify:


res_spec_result = ResonatorSpectroscopyAnalysis(
    dataset=res_spec_dset,
).run()
# res_spec_result.plot_figures()
# res_spec_result.display_figs_mpl()  # use .plot(show_fit=True) instead. If fitting is not yet done, also fit.

plt.plot(res_spec_result.dataset_processed.x0, detrend(np.unwrap(np.angle(res_spec_result.dataset_processed.S21))))
res_spec_result.display_figs_mpl()

'''
## Power scans
As a resonator connected to a superconducting qubit will become nonlinear, we are also interested how the resonator behaves if we modify readout power. 
A two dimensional sweeps works very similar to before. We first define a new Parameter for the amplitude in addition to the one for frequency
'''

cluster.start_adc_calib(4)
qubit_0.clock_freqs.readout(6.7507982)
config = transmon_chip.hardware_config()
config["cluster"]["cluster_module4"]["complex_output_0"]["output_att"] = 30
transmon_chip.hardware_config(config)

amp = ManualParameter(name="amplitude", unit="V", label="amplitude")
freq.batched =True

#Now we prepare MeasurementControl to sweep both amplitude and frequency, and provide the values for both. MeasurementControl will then measure any combination of the two Parameters.

measurement_control.settables([freq, amp])
measurement_control.setpoints_grid(
    (np.linspace(6.745e9, 6.77e9, 300), np.linspace(0.001, 0.25, 30))
)

#Defining and executing the measurement works just like before.

# gettable = ScheduleGettable(
#     transmon_chip,
#     nco_heterodyne_spec_sched,
#     nco_heterodyne_spec_kwargs(qubit_0, frequencies=freq, pulse_amp=amp),
#     real_imag=False,
#     batched=True,
#     max_batch_size=16384
# )
# measurement_control.gettables(gettable)
# #amp.batched=True
# res_power_scan_dset = measurement_control.run("ResonatorPowerScan")


# Pulsed qubit spectroscopy

qubit_0.clock_freqs.readout(6.7507982e9)
config = transmon_chip.hardware_config()
config["cluster"]["cluster_module4"]["complex_output_0"]["output_att"] = 30
transmon_chip.hardware_config(config)
qubit_0.measure.integration_time(1e-6)
qubit_0.measure.pulse_amp(0.145)


cluster.module2.out0_att()
transmon_chip.cfg_sched_repetitions(1024)
freq = ManualParameter(name="freq", unit="Hz", label="Frequency")
freq.batched = True
amp = ManualParameter(name="amp", unit="mV", label="Amplitude")

qubit_0.reset.duration(100e-6)
qubit_0.measure.pulse_amp(0.14)
qubit_0.measure.pulse_duration(1e-6)

qubit_spec_sched_kwargs = nco_two_tone_spec_kwargs(qubit_0, spec_pulse_frequencies=freq, spec_pulse_duration=4e-6, spec_pulse_amp=amp, ro_pulse_amp=0.14)

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=two_tone_spec_sched,
    schedule_kwargs=qubit_spec_sched_kwargs,
    real_imag=False,
    batched=True
)
config = transmon_chip.hardware_config()
config["cluster"]["cluster_module2"]["complex_output_0"]["output_att"] = 0
config["cluster"]["cluster_module2"]["complex_output_0"]["lo_freq"] = 5.8e9
transmon_chip.hardware_config(config)
frequency_setpoints = np.linspace(5.95e9, 6.05e9, 20)

measurement_control.settables([freq, amp])
measurement_control.setpoints_grid(
    (frequency_setpoints, np.linspace(0.001, 0.25, 10))
)

#measurement_control.settables(freq)
#measurement_control.setpoints(frequency_setpoints)
measurement_control.gettables(gettable)

qubit_spec_dset = measurement_control.run("QubitSpectroscopy")
qubit_spec_sched_kwargs = two_tone_spec_kwargs(qubit_0, spec_pulse_frequency=6.0e9, spec_pulse_duration=50e-6, spec_pulse_amp=0.1, ro_pulse_amp=0.14)
compiled = compiler.compile(two_tone_spec_sched(**qubit_spec_sched_kwargs))
compiled.plot_pulse_diagram()
qubit_spec_results = QubitSpectroscopyAnalysis(
    label="QubitSpectroscopy",
    settings_overwrite={"mpl_transparent_background": False},
).run()
qubit_spec_results.display_figs_mpl()
plt.plot(qubit_spec_results.dataset.x0, detrend(np.unwrap(qubit_spec_results.dataset.y1)))




'''
# Amplitude Rabi
The next step is to find the amplitude required for a pi pulse. To do this, we do a simple amplitude Rabi measurement. 
We can visualize the provided schedule function at a circuit level:

'''

qubit_0.clock_freqs.f01(6.018e9)
qubit_0.rxy.duration(100e-9)
schedule = rabi_sched(**rabi_kwargs(qubit_0))
schedule.plot_circuit_diagram()

#If we compile the circuit for a specific quantum device, we can also visualize the pulse envelopes for every line


compiled = compiler.compile(
    schedule
)
compiled.plot_pulse_diagram()


#Qblox control hardware can modify the amplitude of a pulse at runtime,
# greatly speeding up the experiment.To speed up this measurement, we mark the amplitude parameter as `batched`.

pulse_amp = ManualParameter(name="pulse_amplitude", unit="V", label="amplitude")
pulse_amp.batched = True
duration = ManualParameter(name="duration", unit="s", label="duration")
duration.batched = False

#We also need to mark the schedule gettable as batched

transmon_chip.cfg_sched_repetitions(1024)
cluster.module2.out0_att()
qubit_0.reset.duration(100e-6)

# measurement_control.settables([pulse_amp, duration])
# measurement_control.setpoints_grid(
#     ( np.linspace(0, 0.25, 100), 20e-9*np.arange(1,30))
#     #( np.linspace(-0.25, 0.25, 50), np.linspace(6.0e9, 6.1e9, 50))
# )
gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=rabi_sched,
    schedule_kwargs=rabi_kwargs(qubit_0, pulse_amp=pulse_amp, pulse_duration=100e-9),#, frequency=freq),
    batched=True,
    real_imag=False,
)

#And then run the experiment the same way as before

amplitude_setpoints = np.linspace(-0.25, 0.25, 20)

measurement_control.settables(pulse_amp)
measurement_control.setpoints(amplitude_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("Rabi")

plt.plot(dset.x0, dset.y1)

plt.plot(dset.x0, dset.y1)

rabi_result = RabiAnalysis(
    label="Rabi"
).run()
rabi_result.display_figs_mpl()

qubit_0.rxy.amp180(43.4e-3)

from quantify_scheduler.schedules import trace_schedule

# f = qubit_0.clock_freqs.f01()
# print(f)
# f += 6e6
# qubit_0.clock_freqs.f01(f)
# qubit_0.clock_freqs.f01()



# Qubit Frequency - Ramsey
#qubit_0.rxy.amp180(11.962e-3)
tau = ManualParameter(name="tau", unit="s", label="Time")
tau.batched = True

ramsey_sched_kwargs = {
    "qubit": qubit_0.name,
    "times": tau,
    "artificial_detuning": 0e6,
}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=ramsey_sched,
    schedule_kwargs=ramsey_sched_kwargs,
    real_imag=False,
    batched=True,
)

tau_setpoints = np.arange(80e-9, 10e-6, 80e-7)

measurement_control.settables(tau)
measurement_control.setpoints(tau_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("Ramsey")

plt.plot(dset.x0, dset.y1)

plt.plot(dset.x0, np.angle(dset.y0+1j*dset.y1, deg=True))

ramsey_analysis = RamseyAnalysis(
    label="Ramsey", settings_overwrite={"mpl_transparent_background": False}
)
ramsey_result = ramsey_analysis.run(
    artificial_detuning=ramsey_sched_kwargs["artificial_detuning"]
)
ramsey_result.display_figs_mpl()

ramsey_result.fit_results["Ramsey_decay"].best_values["frequency"]


# T1

#f = qubit_0.clock_freqs.f01()
#qubit_0.clock_freqs.f01(f+197e3)
#qubit_0.reset.duration(200e-6)
tau = ManualParameter(name="tau_delay", unit="s", label="Delay")
tau.batched = True

t1_sched_kwargs = {"times": tau, "qubit": qubit_0.name}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=t1_sched,
    schedule_kwargs=t1_sched_kwargs,
    real_imag=False,
    batched=True,
)

delay_setpoints = np.arange(40e-9, 10e-6, 100e-9)

measurement_control.settables(tau)
measurement_control.setpoints(delay_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("T1 experiment")

plt.plot(dset.x0, dset.y0)

plt.plot(dset.x0, dset.y1)

t1_result = T1Analysis(
    label="T1", settings_overwrite={"mpl_transparent_background": False}
).run()
t1_result.display_figs_mpl()

# T2 - Ramsey

f = qubit_0.clock_freqs.f01()
qubit_0.clock_freqs.f01(f+200.9e3)
tau = ManualParameter(name="tau_delay", unit="s", label="Delay")
tau.batched = True

ramsey_sched_kwargs = {"times": tau, "qubit": qubit_0.name, "artificial_detuning": 0}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=ramsey_sched,
    schedule_kwargs=ramsey_sched_kwargs,
    real_imag=True,
    batched=True,
)

tau_setpoints = np.arange(1e-6, 20e-6, 200e-9)
#tau_setpoints = np.linspace(1e-6, 300e-6, 1000)

measurement_control.settables(tau)
measurement_control.setpoints(tau_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("Ramsey")

ramsey_analysis = RamseyAnalysis(
    label="Ramsey", settings_overwrite={"mpl_transparent_background": False}
)
ramsey_result = ramsey_analysis.run(
    artificial_detuning=ramsey_sched_kwargs["artificial_detuning"]
)
ramsey_result.display_figs_mpl()

# Echo

tau = ManualParameter(name="tau_delay", unit="s", label="Delay")
tau.batched = True

echo_sched_kwargs = {"times": tau, "qubit": qubit_0.name}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=echo_sched,
    schedule_kwargs=echo_sched_kwargs,
    real_imag=True,
    batched=True,
)

delay_setpoints = np.arange(1e-6, 30e-6, 200e-9)

measurement_control.settables(tau)
measurement_control.setpoints(delay_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("Echo experiment")

echo_result = EchoAnalysis(
    label="Echo", settings_overwrite={"mpl_transparent_background": False}
).run()
echo_result.display_figs_mpl()

# All XY

qubit_0.reset.duration(500e-6)
def show_allxy(idx: int = 0):
    schedule = allxy_sched(qubit=qubit_0.name, element_select_idx=idx)
    schedule.plot_circuit_diagram()


widgets.interact(show_allxy, idx=widgets.IntSlider(min=0, max=20, step=1))

element_idx = ManualParameter(name="idx", unit="", label="element")
element_idx.batched = True

allxy_sched_kwargs = {"element_select_idx": element_idx, "qubit": qubit_0.name}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=allxy_sched,
    schedule_kwargs=allxy_sched_kwargs,
    real_imag=True,
    batched=True,
)

element_idx_setpoints = np.arange(0, 21, 1)

measurement_control.settables(element_idx)
measurement_control.setpoints(element_idx_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("AllXY experiment")

allxy_result = AllXYAnalysis(
    label="AllXY", settings_overwrite={"mpl_transparent_background": False}
).run()
allxy_result.display_figs_mpl()

transmon_chip.cfg_sched_repetitions(1)
target_state = ManualParameter(name="target", unit="", label="target")
target_state.batched = True

readout_calibration_sched_kwargs = {"prepared_states": target_state, "qubit": qubit_0.name}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=readout_calibration_sched,
    schedule_kwargs=readout_calibration_sched_kwargs,
    real_imag=False,
    batched=True,
)

from scipy.optimize import minimize

transmon_chip.cfg_sched_repetitions(1)

def cost_function(pars):
    amplitude, frequency = pars
    qubit_0.measure.pulse_amp(amplitude)
    qubit_0.clock_freqs.readout(frequency)
    target_state_setpoints = np.zeros(1000, dtype=int)

    measurement_control.settables(target_state)
    measurement_control.setpoints(target_state_setpoints)
    measurement_control.gettables(gettable)

    dset0 = measurement_control.run("discrimination0")


    target_state_setpoints = np.ones(1000, dtype=int)

    measurement_control.settables(target_state)
    measurement_control.setpoints(target_state_setpoints)
    measurement_control.gettables(gettable)
    dset1 = measurement_control.run("discrimination1")
    return -np.abs(np.average(dset0.y1)-np.average(dset1.y1))


from scipy.optimize import minimize
res = minimize(cost_function, x0=(0.14, 6.750792e9), bounds=((0.1, 0.25), (6.7e9, 6.8e9)))

res

cost_function([1.39852543e-01, 6.75079200e+09])

amplitude, frequency = [1.39852543e-01, 6.75079200e+09]
qubit_0.measure.pulse_amp(amplitude)
qubit_0.clock_freqs.readout(frequency)
target_state_setpoints = np.zeros(1000, dtype=int)

measurement_control.settables(target_state)
measurement_control.setpoints(target_state_setpoints)
measurement_control.gettables(gettable)

dset0 = measurement_control.run("discrimination0")


target_state_setpoints = np.ones(1000, dtype=int)

measurement_control.settables(target_state)
measurement_control.setpoints(target_state_setpoints)
measurement_control.gettables(gettable)
dset1 = measurement_control.run("discrimination1")

print(np.average(dset0.y1)-np.average(dset1.y1))

plt.scatter(dset0.y0, dset0.y1, alpha=0.1)
plt.scatter(dset1.y0, dset1.y1, alpha=0.1)

import matplotlib.pyplot as plt
ground = np.where(dset["x0"]==0)
plt.scatter(dset.y0[ground], dset.y1[ground])
ground = np.where(dset["x0"]==1)
plt.scatter(dset.y0[ground], dset.y1[ground])

import matplotlib.pyplot as plt
ground = np.where(dset["x0"]==0)
plt.scatter(dset.y0[ground], dset.y1[ground])
ground = np.where(dset["x0"]==1)
plt.scatter(dset.y0[ground], dset.y1[ground])

dset

#Loading data
from quantify_core.data.handling import load_dataset
rabi_data = load_dataset("20230116-161745") #Change to your own data


r = RabiAnalysis(rabi_data).run(calibration_points=True)
r.display_figs_mpl()























































