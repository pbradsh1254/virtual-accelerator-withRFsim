from __future__ import annotations

import json
import numpy as np

from dataclasses import dataclass
from typing import Optional
from virtaccl.cavity_model import (
    AllCavitySpecs,
    ControllerParams,
    RFSourceParams,
    PIController,
    RFSource,
    FeedforwardSchedule,
    StateSpaceParams,
    StateSpaceCavityModel
)

import math

TWO_PI = 2 * np.pi

_OUTPUT_KEYS = (
    "t",
    "cav_amp",
    "cav_phase",
    "cav_iq",
    "fwd_iq",
    "fwd_amp",
    "refl_iq",
    "refl_amp",
    "P_fwd_W",
    "P_refl_W",
    "P_beam_W",
    "detuning_total_Hz",
    "residual_iq",
    "E_acc_MVm",
    "beam_iq",
    "u_ff",
)



def _empty_accumulators(names: list[str]) -> dict[str, dict[str, list]]:
    return {name: {k: [] for k in _OUTPUT_KEYS} for name in names}

def _finalize(acc: dict[str, dict[str, list]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {k: np.asarray(v) for k, v in inner.items()}
        for name, inner in acc.items()
    }


@dataclass
class SimulationParams:
    dt: float
    fill_duration: float
    flattop_duration: float
    beam_on_time: float

class CavityChain:
    def __init__(
            self, simParams: SimulationParams, 
            CavitySpecList: list[AllCavitySpecs],
            ):
        if not CavitySpecList:
            raise ValueError("CavityChain requires at least one CavitySpec.")
        if not simParams:
            raise ValueError("CavityChain requires Simulation Parameters.")
        self._pulse_time = 0.0
        self.dt = simParams.dt
        self.fill_duration = simParams.fill_duration
        self.flattop_duration = simParams.flattop_duration
        self.beam_on_time = simParams.beam_on_time
        
        #establish chain dictionaries
        self._plants: dict[str, StateSpaceCavityModel] = {}
        self._ff_schedules: dict[str, FeedforwardSchedule] = {}
        self._setpoints: dict[str, complex]            = {}
        self._pulse_times: dict[str, float]            = {}
        self._bpm_offsets: dict[str, dict]             = {}
        

        for cavityspecs in CavitySpecList:
            name = cavityspecs.name
            tau = cavityspecs._state_space_params.tau  # already computed
            self._setpoints[name]    = cavityspecs._RF_source_params.setpoint_phasor
            self._ff_schedules[name] = FeedforwardSchedule(setpoint=self._setpoints[name], 
                                                           dt=self.dt, tau=tau, 
                                                           fill_duration=self.fill_duration, 
                                                           flattop_duration=self.flattop_duration)
            self._pulse_times[name]  = 0.0
            self._bpm_offsets[name]  = cavityspecs.BPM_ref or {}
            self._plants[name] = StateSpaceCavityModel(allcavityspecs=cavityspecs, dt=self.dt)
        
    @classmethod
    def from_json(cls, simParams: SimulationParams, path:str, **kwargs) -> "CavityChain":
        with open(path) as f:
            data = json.load(f)
        cavityspecs = []
        for params in data.values():
            params = dict(params)
            if "setpoint_V" in params:
                params["amp"] = params.pop("setpoint_V")
            if "setpoint_phi" in params:
                params["phase"] = params.pop("setpoint_phi")
            cavityspecs.append(AllCavitySpecs(**params))
        return cls(simParams, cavityspecs, **kwargs)
        

    def _append_step(self, acc: dict[str, list], record: dict) -> None:
        for k in _OUTPUT_KEYS:
            acc[k].append(record[k])

    def closed_loop_step(
            self,
            name: str,
            beam_current: Optional[float] = None,
            beam_phase: Optional[float] = None,
            ):
        plant    = self._plants[name]
        setpoint = self._setpoints[name]
        ff_schedule = self._ff_schedules[name]
        t = self._pulse_times[name]
        
        klystron_active = True
        
        if ff_schedule.is_active == True:
            u_ff  = ff_schedule.get_ff()   # chain owns the schedule
        else:
            klystron_active = False
            u_ff = 0.0 + 0.0j

        raw = plant.step(
            self.dt, u_ff, setpoint=setpoint,
            beam_current=beam_current,
            beam_phase=beam_phase,
            klystron_active = klystron_active
        )

        self._pulse_times[name] += self.dt

        return dict(
            t=t,
            cav_phasor=raw.V_cav,
            fwd_phasor=raw.V_fwd,
            refl_phasor=raw.V_refl,
            P_fwd=raw.P_fwd,
            P_refl=raw.P_refl,
            error=plant._controller._error,
            u_ff=u_ff,
            detuning_total_Hz=raw.total_detuning_Hz,
            beam_iq=raw.I_b,
        )
    
    def switch_feedback(self, new: bool):
        for cavity in self._plants:
            controller = self._plants[cavity]._controller
            controller.params.enable_feedback = new
            
    def _record_to_output(self, name: str, record: dict) -> dict:
        cav = record["cav_phasor"]
        fwd = record["fwd_phasor"]
        refl = record["refl_phasor"]
        L_active = self._plants[name].cavParams.L_active
        return {
            "t":                 record["t"],
            "cav_amp":           abs(cav),
            "cav_phase":         float(np.angle(cav)),
            "cav_iq":            cav,
            "fwd_iq":            fwd,
            "fwd_amp":           abs(fwd),
            "refl_iq":           refl,
            "refl_amp":          abs(refl),
            "P_fwd_W":           record["P_fwd"],
            "P_refl_W":          record["P_refl"],
            "P_beam_W":          record.get("P_beam", 0.0),
            "detuning_total_Hz": record.get("detuning_total_Hz", 0.0),
            "residual_iq":       record["error"],
            "E_acc_MVm":         abs(cav) / (L_active * 1e6),
            "beam_iq":           record.get("beam_iq", 0.0 + 0.0j),
            "u_ff":              record["u_ff"],   # ← add this
        }
    
    def fill(self) -> dict[str, dict[str, np.ndarray]]:
        """Step all cavities through the fill portion of the feedforward schedule.
        Open-loop (feedback state is whatever enable_feedback is set to),
        no beam. Advances the schedule index through all fill steps.
        """
        acc = _empty_accumulators(list(self._plants.keys()))
        # All schedules have the same n_fill_steps (same dt, fill_duration)
        n_fill = next(iter(self._ff_schedules.values())).n_fill_steps

        for _ in range(n_fill):
            for name in self._plants:
                record = self.closed_loop_step(name, beam_current=None, beam_phase=None)
                self._append_step(acc[name], self._record_to_output(name, record))
            self._pulse_time += self.dt

        return _finalize(acc)

    def flattop_step(
            self,
            beam_currents: dict[str, float] = None,
            beam_phases: dict[str, float] = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Advance all cavities by exactly one flat-top timestep.
        Beam current and phase are supplied externally from the beam-dynamics
        simulation. Returns single-element arrays in _OUTPUT_KEYS format.
        """
        acc = _empty_accumulators(list(self._plants.keys()))
        for name in self._plants:
            if beam_currents is not None:
                record = self.closed_loop_step(name, beam_current=beam_currents[name], beam_phase=beam_phases[name])
            else:
                record = self.closed_loop_step(name)
            self._append_step(acc[name], self._record_to_output(name, record))
        self._pulse_time += self.dt
        return _finalize(acc)

    def decay(self, n_tau: float = 5.0) -> dict[str, dict[str, np.ndarray]]:
        """Gate off the klystron and let all cavities ring down.
        Runs for n_tau cavity time constants of the slowest cavity in the chain.
        u_ff=0 with the schedule exhausted, so closed_loop_step already
        passes zero drive — no special handling needed.
        """
        tau_max = max(p.cavParams.tau for p in self._plants.values())
        n_steps = max(1, int(round(n_tau * tau_max / self.dt)))
        acc = _empty_accumulators(list(self._plants.keys()))

        for _ in range(n_steps):
            for name in self._plants:
                record = self.closed_loop_step(name, beam_current=None, beam_phase=None)
                self._append_step(acc[name], self._record_to_output(name, record))
            self._pulse_time += self.dt
        return _finalize(acc)