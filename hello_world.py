
print("Starting hello world")

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
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt as PlotMonitor
from quantify_core.visualization.SI_utilities import format_value_string
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from quantify_scheduler.schedules.spectroscopy_schedules import (
    heterodyne_spec_sched,
    two_tone_spec_sched)

from qblox_instruments.ieee488_2 import DummyBinnedAcquisitionData
from quantify_core.data.handling import set_datadir
# warnings.simplefilter("always")

set_datadir("quantify_data")

with PlugAndPlay() as p:
    p.print_devices()


cluster = Cluster("cluster", "Ontario")# Either Erie or Ontario depending on the cluster
# Reset
cluster.reset()

ic = InstrumentCoordinator("IC")
ic.add_component(ClusterComponent(cluster))


measurement_control = MeasurementControl("QubitCalibration")

#There is a problem with the following block of code, but i think its just for plotting
'''
plotmon = PlotMonitor("PlotMonitor")
instmon = InstrumentMonitor("InstrumentMonitor")
measurement_control.instr_plotmon(plotmon.name)
plotmon.tuids_max_num(1)
'''


'''The central component for all experiments is the QuantumDevice. It connects information about the qubits and their connectivity with the involved control hardware, 
instrument coordinator and measurement control. First we define the device, and connect it to instrument coordinator and measurement control.
'''

transmon_chip = QuantumDevice("transmon_chip")
transmon_chip.instr_instrument_coordinator("IC")
transmon_chip.instr_measurement_control("QubitCalibration")

#Next, we add information about the control electronics and how they are connected to the QuantumDevice.

hardware_config_file = "./hw_config_rf.json"

with open(hardware_config_file, "r") as f:
    hardware_cfg = json.load(f)
transmon_chip.hardware_config(hardware_cfg)

transmon_chip.cfg_sched_repetitions(1024)


#We also need to provide information about the qubits. In this example, we will use a single qubit called `qubit_0`.


qubit_0 = BasicTransmonElement("q0")
transmon_chip.add_element(qubit_0)

qubit_0.clock_freqs.readout(6.74e9)
qubit_0.clock_freqs.f01(6.01e9)

qubit_0.measure.pulse_amp(0.006)
qubit_0.measure.pulse_duration(1000e-9)
qubit_0.measure.integration_time(1600e-9)
qubit_0.measure.acq_delay(164e-9)

qubit_0.reset.duration(100e-6)

qubit_0.rxy.duration(120e-9)
qubit_0.rxy.amp180(0.2)
qubit_0.rxy.motzoi(0.1)



'''Now that the QuantumDevice is fully defined, we can use it. To automate the compilation process, we will define a compiler where the 
transmon_chip is added as default value. This enables us to use
compiler.compile(schedule), whenever we want to compile a generic schedule to this specific quantum device.
'''


#compiler = SerialCompiler("compiler")
compiler = SerialCompiler("compiler", quantum_device=transmon_chip)









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
def heterodyne_spec_kwargs(qubit: BasicTransmonElement) -> Dict[str, Any]:
    return {
        "pulse_amp": qubit.measure.pulse_amp(),
        "pulse_duration": qubit.measure.pulse_duration(),
        "frequencies": [qubit.clock_freqs.readout()],
        "acquisition_delay": qubit.measure.acq_delay(),
        "integration_time": qubit.measure.integration_time(),
        "init_duration": qubit.reset.duration(),
        "port": qubit.ports.readout(),
        "clock": qubit.name + ".ro",
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




'''If we want to run the tutorial notebook without hardware and still use the fitting routines, we need to provide the (fake) 
results for each experiment. These helper functions facilitate doing so.
'''


class heterodyne_spec_sched_with_dummy:
    def __init__(self):
        dummy_data = [
            [DummyBinnedAcquisitionData(data=(0.1, 0.2), thres=1, avg_cnt=0)],
            [DummyBinnedAcquisitionData(data=(0.3, 0.4), thres=1, avg_cnt=0)],
            [DummyBinnedAcquisitionData(data=(0.5, 0.6), thres=1, avg_cnt=0)],
        ]
        self.dummy_data_iter = iter(dummy_data)

    def __call__(self, **kwargs):
        print("New schedule generation")
        print(kwargs)
        cluster.set_dummy_binned_acquisition_data(
            slot_idx=16,
            sequencer=0,
            acq_index_name="0",
            data=next(self.dummy_data_iter),
        )
        return heterodyne_spec_sched(**kwargs)


def set_dummy_data_rabi():
    dummy_data_rabi = [
        DummyBinnedAcquisitionData(data=(np.cos(2 * np.pi * i / 20), np.sin(2 * np.pi * i / 20)), thres=1, avg_cnt=0)
        for i in range(50)
    ]

    cluster.set_dummy_binned_acquisition_data(
        slot_idx=16,
        sequencer=0,
        acq_index_name="0",
        data=dummy_data_rabi,
    )


def clear_dummy_data():
    cluster.set_dummy_binned_acquisition_data(
        slot_idx=16,
        sequencer=0,
        acq_index_name="0",
        data=[],
    )






'''Quantify provides analysis classes for many standard experiments. 
But sometimes we want to define our own. Below we show how this is done on the example of two tone spectroscopy on a superconducting qubit.
'''

class QubitSpectroscopyAnalysis(ba.BaseAnalysis):
    """
    Fits a Lorentzian function to qubit spectroscopy data and finds the
    0-1 transistion frequency of the qubit
    """

    def process_data(self):
        """
        Populates the :code:`.dataset_processed`.
        """
        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.

        self.dataset_processed["Magnitude"] = self.dataset.y0
        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.Magnitude.attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset_processed["x0"] = self.dataset.x0
        self.dataset_processed = self.dataset_processed.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a Lorentzian function to the data.
        """
        mod = LorentzianModel()

        magnitude = np.array(self.dataset_processed["Magnitude"])
        frequency = np.array(self.dataset_processed.x0)
        guess = mod.guess(magnitude, x=frequency)
        fit_result = mod.fit(magnitude, params=guess, x=frequency)

        self.fit_results.update({"Lorentzian_peak": fit_result})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """
        fit_result = self.fit_results["Lorentzian_peak"]
        fit_warning = ba.wrap_text(ba.check_lmfit(fit_result))

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True

            text_msg = "Summary\n"
            text_msg += format_value_string(
                "Frequency 0-1",
                fit_result.params["x0"],
                unit="Hz",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Peak width",
                fit_result.params["width"],
                unit="Hz",
                end_char="\n",
            )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["frequency_01"] = ba.lmfit_par_to_ufloat(
            fit_result.params["x0"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """Creates qubit spectroscopy figure"""

        fig_id = "qubit_spectroscopy"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        qpl.plot_textbox(ax, self.quantities_of_interest["fit_msg"])

        self.dataset_processed.Magnitude.plot(ax=ax, marker=".", linestyle="")

        qpl.plot_fit(
            ax=ax,
            fit_res=self.fit_results["Lorentzian_peak"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        qpl.set_ylabel(ax, r"Output voltage", self.dataset_processed.Magnitude.units)
        qpl.set_xlabel(
            ax, self.dataset_processed.x0.long_name, self.dataset_processed.x0.units
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset, "S21")


def lorentzian(
    x: float,
    x0: float,
    width: float,
    A: float,
    c: float,
) -> float:
    r"""
    A Lorentzian function.

    Parameters
    ----------
    x:
        independent variable
    x0:
        horizontal offset
    width:
        Lorenztian linewidth
    A:
        amplitude
    c:
        vertical offset

    Returns
    -------
    :
        Lorentzian function


    .. math::

        y = \frac{A*\mathrm{width}}{\pi(\mathrm{width}^2 + (x - x_0)^2)} + c

    """

    return A * width / (np.pi * ((x - x0) ** 2) + width**2) + c


class LorentzianModel(lmfit.model.Model):
    """
    Model for data which follows a Lorentzian function.
    """

    # pylint: disable=empty-docstring
    # pylint: disable=abstract-method
    # pylint: disable=too-few-public-methods

    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later.

        super().__init__(lorentzian, *args, **kwargs)

        self.set_param_hint("x0", vary=True)
        self.set_param_hint("A", vary=True)
        self.set_param_hint("c", vary=True)
        self.set_param_hint("width", vary=True)

    # pylint: disable=missing-function-docstring
    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        x = kws.get("x", None)

        if x is None:
            return None

        # Guess that the resonance is where the function takes its maximal
        # value
        x0_guess = x[np.argmax(data)]
        self.set_param_hint("x0", value=x0_guess)

        # assume the user isn't trying to fit just a small part of a resonance curve.
        xmin = x.min()
        xmax = x.max()
        width_max = xmax - xmin

        delta_x = np.diff(x)  # assume f is sorted
        min_delta_x = delta_x[delta_x > 0].min()
        # assume data actually samples the resonance reasonably
        width_min = min_delta_x
        width_guess = np.sqrt(width_min * width_max)  # geometric mean, why not?
        self.set_param_hint("width", value=width_guess)

        # The guess for the vertical offset is the mean absolute value of the data
        c_guess = np.mean(data)
        self.set_param_hint("c", value=c_guess)

        # Calculate A_guess from difference between the peak and the backround level
        A_guess = np.pi * width_guess * (np.max(data) - c_guess)
        self.set_param_hint("A", value=A_guess)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)




print('Finished hello world\n')


