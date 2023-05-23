import warnings

import ipywidgets as widgets
import numpy as np

from qcodes.instrument.parameter import ManualParameter
from quantify_scheduler.gettables import ScheduleGettable

#import ipynb



# warnings.simplefilter("always")

from quantify_core.analysis.spectroscopy_analysis import ResonatorSpectroscopyAnalysis

from quantify_core.analysis.single_qubit_timedomain import (
    AllXYAnalysis,
   EchoAnalysis,
   RabiAnalysis,
   RamseyAnalysis,
   T1Analysis,
)

from quantify_scheduler.schedules.timedomain_schedules import (
    allxy_sched,
    echo_sched,
    rabi_sched,
    ramsey_sched,
    t1_sched,
    readout_calibration_sched
)

from quantify_scheduler.schedules.spectroscopy_schedules import (
    heterodyne_spec_sched,
    two_tone_spec_sched,
)



#import ipynb
from hello_world import qubit_0, measurement_control, transmon_chip
from hello_world import compiler
from hello_world import heterodyne_spec_kwargs, two_tone_spec_kwargs, rabi_kwargs
from hello_world import set_dummy_data_rabi, clear_dummy_data, heterodyne_spec_sched_with_dummy
from hello_world import QubitSpectroscopyAnalysis

# Resonator spectroscopy
'''
The very first experiment for tuning a superconducting qubit is to find the resonance frequency of the readout resonator. 
As we will be sweeping the frequency, we first define a qcodes `Parameter` as placeholder, which will be filled with the sweep values during the experiment.
'''

freq = ManualParameter(name="frequency", unit="Hz", label="f")

#Next, we need to set up `MeasurementControl` to use this Parameter. First, we set that the next measurement will be sweeping over the frequency:

measurement_control.settables(freq)

#We also need to set the actual values for the experiment. Here from 2-10 GHz, in 3 steps.

measurement_control.setpoints(np.linspace(7.435e9, 7.45e9, 100))
'''
We also need to define the actual measurement. To this end, we use a schedule function. This is simply a function that generates a schedule for a given frequency. 
As Resonator spectroscopy is a standard experiment, we can simply import the function from quantify. We will have a look at custom schedule functions later.
'''


# TODO: We should show a schedule function somewhere
heterodyne_spec_function = heterodyne_spec_sched

'''
The schedule function itself is still an abstract object, it does not refer to the hardware config (e.g. cabling, ip addresses) yet. 
To fully define the measurements, we define a `ScheduleGettable`, which fully describes the experiment.
'''

gettable = ScheduleGettable(
    quantum_device=transmon_chip,
    schedule_function=heterodyne_spec_function,
    schedule_kwargs=heterodyne_spec_kwargs(qubit_0, frequency=freq),
    real_imag=False,
)


#We also need to connect the `ScheduleGettable` to MeasurementControl

measurement_control.gettables(gettable)

heterodyne_spec_function(**heterodyne_spec_kwargs(qubit_0)).plot_circuit_diagram()

compiled = compiler.compile(heterodyne_spec_function(**heterodyne_spec_kwargs(qubit_0)))
compiled.plot_pulse_diagram()

#Finally, we execute the experiment using `MeasurementControl`. The data is automatically saved, and also available immediately as `xarray.Dataset`

res_spec_dset = measurement_control.run("ResonatorSpectroscopy")
'''
The information we are interested in (the resonance frequency) is not immediately accessible, we need to analyze the data first. 
To this end, we can use an analysis class based on the `lmfit` package. For standard experiments like resonator spectroscopy, it is already provided by quantify:
'''

res_spec_result = ResonatorSpectroscopyAnalysis(
    dataset=res_spec_dset,
).run()

# res_spec_result.plot_figures()
# res_spec_result.display_figs_mpl()  # use .plot(show_fit=True) instead. If fitting is not yet done, also fit.

res_spec_result.display_figs_mpl()
'''
## Power scans
As a resonator connected to a superconducting qubit will become nonlinear, we are also interested how the resonator behaves if we modify readout power. 
A two dimensional sweeps works very similar to before. We first define a new Parameter for the amplitude in addition to the one for frequency
'''

amp = ManualParameter(name="amplitude", unit="V", label="amplitude")

#Now we prepare MeasurementControl to sweep both amplitude and frequency, and provide the values for both. MeasurementControl will then measure any combination of the two Parameters.

measurement_control.settables([freq, amp])
measurement_control.setpoints_grid(
    (np.linspace(7.432e9, 7.438e9, 50), np.linspace(0.001, 0.05, 20))
)

np.linspace(0.001, 0.05, 20)[3]

#Defining and executing the measurement works just like before.

gettable = ScheduleGettable(
    transmon_chip,
    heterodyne_spec_sched,
    heterodyne_spec_kwargs(qubit_0, frequency=freq, pulse_amp=amp),
    real_imag=False,
)
measurement_control.gettables(gettable)
#amp.batched=True
res_power_scan_dset = measurement_control.run("ResonatorPowerScan")


# Pulsed qubit spectroscopy

config = transmon_chip.hardware_config()
config["cluster"]["cluster_module1"]["complex_output_0"]

freq = ManualParameter(name="freq", unit="Hz", label="Amplitude")
transmon_chip.hardware_config
qubit_0.measure.pulse_amp(0.003736842105263157)
qubit_spec_sched_kwargs = two_tone_spec_kwargs(qubit_0, spec_pulse_frequency=freq)

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=two_tone_spec_sched,
    schedule_kwargs=qubit_spec_sched_kwargs,
    real_imag=False,
)

frequency_setpoints = np.linspace(5.06e9, 5.1e9, 100)

measurement_control.settables(freq)
measurement_control.setpoints(frequency_setpoints)
measurement_control.gettables(gettable)

qubit_spec_dset = measurement_control.run("QubitSpectroscopy")

qubit_spec_results = QubitSpectroscopyAnalysis(
    label="QubitSpectroscopy",
    settings_overwrite={"mpl_transparent_background": False},
).run()
qubit_spec_results.display_figs_mpl()


# Amplitude Rabi
#The next step is to find the amplitude required for a pi pulse. To do this, we do a simple amplitude Rabi measurement. We can visualize the provided schedule function at a circuit level:

qubit_0.clock_freqs.f01(5.08170e9)
schedule = rabi_sched(**rabi_kwargs(qubit_0))
schedule.plot_circuit_diagram()

#If we compile the circuit for a specific quantum device, we can also visualize the pulse envelopes for every line

compiled = compiler.compile(
    schedule
)
compiled.plot_pulse_diagram()

#Qblox control hardware can modify the amplitude of a pulse at runtime, greatly speeding up the experiment.To speed up this measurement, we mark the amplitude parameter as `batched`.

pulse_amp = ManualParameter(name="pulse_amplitude", unit="V", label="amplitude")
pulse_amp.batched = True

#We also need to mark the schedule gettable as batched

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=rabi_sched,
    schedule_kwargs=rabi_kwargs(qubit_0, pulse_amp=pulse_amp),
    batched=True,
    real_imag=True,
)

#And then run the experiment the same way as before

amplitude_setpoints = np.linspace(-0.05, 0.05, 1000)

measurement_control.settables(pulse_amp)
measurement_control.setpoints(amplitude_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("Rabi")

rabi_result = RabiAnalysis(
    label="Rabi"
).run()
rabi_result.display_figs_mpl()

from quantify_scheduler.schedules import trace_schedule

# Qubit Frequency - Ramsey

qubit_0.rxy.amp180(11.962e-3)
tau = ManualParameter(name="tau", unit="s", label="Time")
tau.batched = True

ramsey_sched_kwargs = {
    "qubit": qubit_0.name,
    "times": tau,
    "artificial_detuning": 1e6,
}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=ramsey_sched,
    schedule_kwargs=ramsey_sched_kwargs,
    real_imag=True,
    batched=True,
)

tau_setpoints = np.arange(120e-9, 5e-6, 40e-9)

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

ramsey_result.fit_results["Ramsey_decay"].best_values["frequency"]

# T1



f = qubit_0.clock_freqs.f01()
qubit_0.clock_freqs.f01(f+197e3)
qubit_0.reset.duration(200e-6)
tau = ManualParameter(name="tau_delay", unit="s", label="Delay")
tau.batched = True

t1_sched_kwargs = {"times": tau, "qubit": qubit_0.name}

gettable = ScheduleGettable(
    transmon_chip,
    schedule_function=t1_sched,
    schedule_kwargs=t1_sched_kwargs,
    real_imag=True,
    batched=True,
)

delay_setpoints = np.arange(40e-9, 40e-6, 400e-9)

measurement_control.settables(tau)
measurement_control.setpoints(delay_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("T1 experiment")

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
    real_imag=True,
    batched=True,
)

target_state_setpoints = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])

measurement_control.settables(target_state)
measurement_control.setpoints(target_state_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("discrimination2")

import matplotlib.pyplot as plt
ground = np.where(dset["x0"]==0)
plt.scatter(dset.y0[ground], dset.y1[ground])
ground = np.where(dset["x0"]==1)
plt.scatter(dset.y0[ground], dset.y1[ground])

dset

from quantify_core.data.handling import load_dataset
rabi_data = load_dataset("20230116-161745")


r = RabiAnalysis(rabi_data).run(calibration_points=True)
r.display_figs_mpl()

































