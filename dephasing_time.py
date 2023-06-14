# %%
import numpy as np
from qcodes.instrument import Instrument
from qcodes.instrument.parameter import ManualParameter
from quantify_core.analysis.single_qubit_timedomain import RamseyAnalysis
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.schedules.timedomain_schedules import echo_sched, ramsey_sched
from quantify_scheduler.visualization.pulse_diagram import pulse_diagram_plotly

from shared import (
    measurement_control,
    transmon_chip,
    qubit_0,
    compiler,
)

# %%

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

# %%
tau_setpoints = np.arange(1e-6, 30e-6, 100e-9)

measurement_control.settables(tau)
measurement_control.setpoints(tau_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("Ramsey")

# %%

try:
    ramsey_analysis = RamseyAnalysis(
        label="Ramsey", settings_overwrite={"mpl_transparent_background": False}
    )
    ramsey_result = ramsey_analysis.run(
        artificial_detuning=ramsey_sched_kwargs["artificial_detuning"]
    )
    ramsey_result.display_figs_mpl()
except IndexError:
    print("Fit failed.")

# %%
help(echo_sched)

# %%

schedule = echo_sched(times=400e-9, repetitions=10, qubit=qubit_0.name)
schedule = compiler.compile(schedule, transmon_chip.generate_compilation_config())

pulse_diagram_plotly(schedule)

# %%

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

# %%


delay_setpoints = np.arange(1e-6, 30e-6, 200e-9)

measurement_control.settables(tau)
measurement_control.setpoints(delay_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("Echo experiment")

# %%
from quantify_core.analysis.single_qubit_timedomain import EchoAnalysis

try:
    echo_result = EchoAnalysis(
        label="Echo", settings_overwrite={"mpl_transparent_background": False}
    ).run()
    echo_result.display_figs_mpl()
except ValueError:
    print("Fit failed.")

# %%


Instrument.close_all()





































