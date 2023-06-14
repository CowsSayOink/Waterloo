# %%
import json
import warnings
from typing import Any, Callable

from qblox_instruments import Cluster, ClusterType, PlugAndPlay
from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.visualization.instrument_monitor import InstrumentMonitor
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt as PlotMonitor
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from typing import Dict

warnings.simplefilter("always")
# Setting experiment data directory
set_datadir("quantify-data")
print(f'Data will be saved in:\n\t"{get_datadir()}"')

# %%

with PlugAndPlay() as p:
    p.print_devices()

# %%
dummy_cfg = {
    "2": ClusterType.CLUSTER_QCM_RF,
    "4": ClusterType.CLUSTER_QRM_RF,
    "6": ClusterType.CLUSTER_QRM_RF,
}

cluster = Cluster("cluster", "Ontario")# Either Erie or Ontario depending on the cluster

# Reset
cluster.reset()

# %%

ic = InstrumentCoordinator("IC")
ic.add_component(ClusterComponent(cluster))

# %%

measurement_control = MeasurementControl("Qubit Calibration")

plotmon = PlotMonitor("Plot Monitor")
instmon = InstrumentMonitor("Instrument Monitor")
measurement_control.instr_plotmon(plotmon.name)
plotmon.tuids_max_num(1)

# %%

hardware_config_file = "../hw_config_rf.json"

with open(hardware_config_file, "r") as f:
    hardware_cfg = json.load(f)

# %%
transmon_chip = QuantumDevice("transmon_chip")

qubit_0 = BasicTransmonElement("q0")
transmon_chip.add_element(qubit_0)

transmon_chip.hardware_config(hardware_cfg)
transmon_chip.instr_instrument_coordinator("IC")
transmon_chip.instr_measurement_control("Qubit Calibration")

# %%
transmon_chip.cfg_sched_repetitions(1024)


# %%
# Transmon Element parameters
qubit_0.clock_freqs.readout(7.01e9)
qubit_0.clock_freqs.f01(4.865e9)

qubit_0.measure.pulse_amp(0.03)
qubit_0.measure.pulse_duration(2000e-9)
qubit_0.measure.integration_time(1000e-9)
qubit_0.measure.acq_delay(500e-9)

qubit_0.reset.duration(2e-6)

qubit_0.rxy.duration(120e-9)
qubit_0.rxy.amp180(0.2)
qubit_0.rxy.motzoi(0.1)


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
        "frequency": qubit.clock_freqs.readout(),
        "acquisition_delay": qubit.measure.acq_delay(),
        "integration_time": qubit.measure.integration_time(),
        "init_duration": qubit.reset.duration(),
        "port": qubit.ports.readout(),
        "clock": qubit.name + ".ro",
    }


@kwarg_wrapper
def two_tone_spec_kwargs(qubit: BasicTransmonElement, **kwargs) -> Dict[str, Any]:
    return {
        "spec_pulse_amp": 0.01,
        "spec_pulse_duration": 1e-6,
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


compiler = SerialCompiler("compiler")






print("Shared has run")


















