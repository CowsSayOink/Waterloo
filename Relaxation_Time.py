# %%
from qcodes.instrument import Instrument
from qcodes.instrument.parameter import ManualParameter
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.schedules.timedomain_schedules import t1_sched
from quantify_scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib
from quantify_scheduler.visualization.pulse_diagram import pulse_diagram_plotly

from shared import (
    measurement_control,
    transmon_chip,
    qubit_0,
    compiler,
)

# %%

help(t1_sched)

# %% [markdown]
# Below we generate such an example schedule and show its circuit diagram.

# %%


schedule = t1_sched(times=1, repetitions=10, qubit=qubit_0.name)

circuit_diagram_matplotlib(schedule)

# %%


schedule = t1_sched(times=50e-6, repetitions=10, qubit=qubit_0.name)
schedule = compiler.compile(schedule, transmon_chip.generate_compilation_config())
pulse_diagram_plotly(schedule)


pulse_diagram_plotly(schedule)


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

# %%
import numpy as np

delay_setpoints = np.arange(40e-9, 40e-6, 400e-9)

measurement_control.settables(tau)
measurement_control.setpoints(delay_setpoints)
measurement_control.gettables(gettable)

dset = measurement_control.run("T1 experiment")

# %%
from quantify_core.analysis.single_qubit_timedomain import T1Analysis

try:
    t1_result = T1Analysis(
        label="T1", settings_overwrite={"mpl_transparent_background": False}
    ).run()
    t1_result.display_figs_mpl()
except ValueError:
    print("Fit failed.")

# %%


Instrument.close_all()

# %%














































