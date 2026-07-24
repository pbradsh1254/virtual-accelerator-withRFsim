#imports
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from scipy.integrate import solve_ivp
from scipy.linalg import expm

#constants
TWO_PI = 2*np.pi


@dataclass
class AllCavitySpecs:
    """
    Aggregation of all the outputs for a single cavity model.
    Splits the parameters into their individual dataclasses for the cavity, controller, and RF source.
    """
    #--required--
    name: str

    R_over_Q: float
    amp: float
    phase: float
    BPM_ref: dict[str, float]

    #--OPTIONAL--
    #--cavity params--
    f0: float  = 805e6          # Resonant frequency (Hz)
    Q0: float  = 7.0e9          # Unloaded quality factor
    Q_L: float = 7.5e5          # Loaded quality factor
    L_active: float = 0.906     # Active Length [m]
    Vmax: float = 2e7           # Maximum cavity field (V/m)
    alpha_beam: float = 645/945     # beam loading including chopper time removal
    
    #--controller params--
    enable_feedback: bool = False
    Kp: float = 20
    Ki: float = 5e6                       # there may be more controller params that can be added here
    tau_leak: float = 1.5e-5               # slight leak, tau/dt timesteps original data will be divided by e
    
    #--rf source params--
    transport_delay: float = 6.5e-6         # LLRF→klystron system transport delay [s]. This should be calculated I feel
    P_max: float = 550e3                    # max output of Klystron is 550 KW 
    fill_duration: float = 250e-6           # fill duration [s]
    flattop_duration: float = 1000e-6       # flattop duration [s]

    #--noise params--
    enable_process_noise: bool = False
    process_noise_rms_Vps: float = 0.0      # additive envelope forcing rms [V/s]
    process_noise_bw_Hz: float = 1.0e4
    enable_measurement_noise: bool = False
    meas_noise_rms_V: float = 0.0           # additive readout I/Q noise rms [V]

    #--detuning params--
    enable_static: bool = False
    static_Hz: float = 0.0

    enable_lfd: bool = False
    lfd_mode: str = "mechanical"            # "mechanical" | "algebraic"
    lfd_modes: dict = field(default_factory=lambda: {
        "f_m_Hz": 200.0,
        "Q_m": 100.0,
        "K_m_HzMV2": 1.0
    })
    lfd_algebraic_K_HzMV2: float = 1.0      # used when lfd_mode == "algebraic"
    persist_lfd: bool = False               # keep mechanical state across pulses

    enable_microphonics: bool = False
    mic_rms_Hz: float = 5.0                 # broadband RMS detuning [Hz]
    mic_bw_Hz: float = 30.0                 # 1st-order low-pass cutoff [Hz]
    #mic_lines: Sequence[tuple] = field(default_factory=lambda: [(60.0, 3.0),(120.0, 1.5)])
    enable_random_detuning: bool = False
    random_detuning_rms_Hz: float = 0.0     # per-step white detuning jitter [Hz]


    def __post_init__(self) -> None:
        phase_rad = math.radians(self.phase)
        
        self._state_space_params = StateSpaceParams(
            name=self.name,
            R_over_Q=self.R_over_Q,
            BPM_phase_offset=self.BPM_ref.get("phase_offset", 0.0),
            f0=self.f0,
            Q0=self.Q0,
            Q_L=self.Q_L,
            L_active=self.L_active,
            Vmax=self.Vmax,
        )
        self._controller_params = ControllerParams(
            enable_feedback=self.enable_feedback,
            Kp=self.Kp,
            Ki=self.Ki,
            tau_leak = self.tau_leak
        )
        self._RF_source_params = RFSourceParams(
            name=self.name,
            setpoint_amp=self.amp,
            setpoint_phase=phase_rad,
            BPM_phase_offset=self.BPM_ref.get("phase_offset", 0.0),
            transport_delay=self.transport_delay,
            P_max=self.P_max,
            fill_duration=self.fill_duration,
            flattop_duration=self.flattop_duration,
            R_L=self.R_over_Q*self.Q_L
        )
        self._noise_params = NoiseParams(
            enable_process_noise=self.enable_process_noise, 
            process_noise_rms_Vps=self.process_noise_rms_Vps, 
            process_noise_bw_Hz=self.process_noise_bw_Hz,
            enable_measurement_noise=self.enable_measurement_noise,
            meas_noise_rms_V=self.meas_noise_rms_V,
            )
        self._detuning_params = DetuningParams(
            enable_static=self.enable_static, 
            static_Hz=self.static_Hz,
            enable_lfd=self.enable_lfd,
            lfd_mode=self.lfd_mode,
            lfd_modes=self.lfd_modes,
            lfd_algebraic_K_HzMV2=self.lfd_algebraic_K_HzMV2,
            persist_lfd=self.persist_lfd,
            enable_microphonics=self.enable_microphonics,
            mic_rms_Hz=self.mic_rms_Hz,
            mic_bw_Hz=self.mic_bw_Hz,
            enable_random_detuning=self.enable_random_detuning,
            random_detuning_rms_Hz=self.random_detuning_rms_Hz,
        )



class FeedforwardSchedule:
    def __init__(self,
                setpoint: complex, 
                dt: float, 
                tau: float, 
                fill_duration: float, 
                flattop_duration: float,
                ):
        """
        Create a feedforward schedule for the RF source. Returns list of (time, phasor) tuples.
        Used only if prior feedforward schedule does not exist.
        May need to make update so that a loaded feedforward schedule can be rediscretized for different dt
        """
        # Create a default schedule based on the provided parameters
        time_points = np.arange(-fill_duration, flattop_duration, dt)
        phasor_values = np.zeros_like(time_points, dtype=complex)
        # Fill phasor values during fill duration with calculated exponential rise
        phasor_values[time_points <= 0] = setpoint / max(1.0 - np.exp(-fill_duration / tau), 1e-12)
        # Maintain phasor values during flattop duration
        phasor_values[(time_points > 0) & (time_points <= flattop_duration)] = setpoint
        self.schedule = phasor_values
        self._current_index = 0
        self.is_active = True
        self.n_fill_steps = int(np.sum(time_points <= 0))

    def get_ff(self):
        result = self.schedule[self._current_index]
        self._current_index += 1
        if self._current_index >= len(self.schedule):
            self.is_active = False
        return result

@dataclass
class ControllerParams:
    #Complex PI controller which adds to the feedforward system
    
    #required
    enable_feedback: bool
    Kp: float 
    Ki: float
    # ── Integral gating (stability) ────────────────────────────────────────────
    # Integrator is only active when ALL of these conditions are satisfied:
    #   1. t > fill_time_s + integral_guard_s        (post-fill settle)
    #   2. t < rf_off_time_s - integral_guard_s      (pre-decay settle)
    #   3. |e| < integral_error_gate_frac * |V_set|  (not in a large transient)
    #   4. |e| > integral_deadband_frac * |V_set|    (above noise floor)
    #   5. drive not saturated                        (anti-windup)
    # These are passed in at compute() time via gate_integral flag. 
    anti_windup_backcalc: bool = True               # back-calculate on saturation
    drive_max: Optional[float] = None               # forward-drive amplitude sat [V]
    drive_slew_max: Optional[float] = None          # max |dV_fwd/dt| [V/s]
    tau_leak: Optional[float] = None                # integral leak time constant [s^-1], prioritizes recent signal, avoids windup
    # integral_guard_s: float = 0.0        # dead-time after fill onset and before decay [s]; 0 = disabled
    # integral_error_gate_frac: float = 1.0  # gate off when |e| > this frac of setpoint; 1.0 = disabled
    # integral_deadband_frac: float = 0.0    # deadband fraction; 0 = disabled

class PIController:
    def __init__(self, PIparams: ControllerParams,):
        self.params=PIparams
        self.reset()

    def reset(self) -> None:
        self._suppress_fb_next_step = False
        self._integral: complex = 0.0 + 0.0j
        self._last_u: complex = 0.0 + 0.0j

    def reset_integrator(self) -> None:
        self._integral = 0.0 + 0.0j

    
    def compute_feedback(self, dt: float, 
                         setpoint: complex, V_meas: complex):
        error: complex = setpoint-V_meas
        params = self.params
        self._error = error

        
        if params.enable_feedback == True:
            setpoint_amp = abs(setpoint)
            
            #leaky integrator
            tau_leak = params.tau_leak
            if tau_leak is not None and float(tau_leak) > 0.0:
                leak = math.exp(-dt / float(tau_leak))
                self._integral *= leak

            self._integral += error * dt
            u_fb: complex = params.Kp * error + params.Ki * self._integral

        else:
            u_fb = 0.0 + 0.0j
        
        return u_fb

@dataclass
class RFSourceParams:
    #All parameters needed for the RF drive system
    
    #--required--
    name: str
    setpoint_amp: float             # target cavity voltage[V]
    setpoint_phase: float           # target cavity phase [radians]
    BPM_phase_offset: float         # Phase offset for BPM (rad)
    transport_delay: float          # LLRF→klystron system transport delay [s]. This should be calculated I feel
    fill_duration: float            # fill duration [s]
    flattop_duration: float         # flattop duration [s]
    P_max: float                    # maximum power
    R_L: float                 # shunt impedance
    slewing_limit: Optional[float] = None  # slewing (dV/dt) limit [V/s]

    def __post_init__(self):
        self.setpoint_phasor = self.setpoint_amp*np.exp(1j*self.setpoint_phase)
        self.V_max = math.sqrt(self.P_max*4*self.R_L)

class RFSource:
    def __init__(self, RFparams: RFSourceParams, dt: float,):
        self.name = RFparams.name
        self.setpoint_phasor = RFparams.setpoint_phasor
        self._delay = RFparams.transport_delay
        self._len_buffer = int(math.floor(self._delay / dt+ 1e-9))
        self._buffer = []
        self.V_max = RFparams.V_max
        self.slewing_limit = RFparams.slewing_limit
        self.fill_duration = RFparams.fill_duration
        self.last_cmd: complex = 0.0 + 0.0j
        self.last_out: complex = 0.0 + 0.0j
        self.last_err: complex = 0.0 + 0.0j
        self.saturated: bool = False
        self.slewing: bool = False
        self.reset()


    def reset(self) -> None:
        self._buffer = []
        self._ind_buf = 0
        self._s1: complex = 0.0 + 0.0j      # first-pole state
        self._s2: complex = 0.0 + 0.0j      # second-pole state
        self.last_cmd = 0.0 + 0.0j
        self.last_out = 0.0 + 0.0j
        self.last_err = 0.0 + 0.0j
        self.saturated = False
        self.slewing = False

    # ---- fractional-delay read/write -----------------------------------------
    def buffer(self, u_fb: Optional[complex]=None, setpoint: Optional[float]=None) -> complex: 
        #Must have either u_fb or setpoint filled
        if u_fb is not None and setpoint is not None:
            raise ValueError("requires either u_cmd or setpoint")
        if len(self._buffer) < self._len_buffer:
            # Not yet full — push the sample but output zero
            self._buffer.append(u_fb)
            return 0.0 + 0.0j
        else:
            # Ring buffer write+read with index arithmetic
            self._buffer[self._ind_buf] = u_fb
            out = self._buffer[(self._ind_buf - self._len_buffer) % self._len_buffer]
            self._ind_buf = (self._ind_buf + 1) % self._len_buffer
        
        return out
    
    # ---- one step: u_cmd -> u_rf ---------------------------------------------
    def drive(self, dt:float, setpoint: complex, V_meas: complex, 
              u_ff: complex, controller: PIController
              ) -> complex:
        """Advance the actuator one step; return delivered drive u_rf."""

        u_fb = controller.compute_feedback(dt=dt, setpoint=setpoint, V_meas=V_meas, )
        u_fb = self.buffer(u_fb=u_fb)
        
        u_cmd = u_ff + u_fb
        
        # --- first-order bandwidth pole (exact ZOH on held input) ---
        # if self._tau1 is not None:
        #     a1 = math.exp(-dt / self._tau1)
        #     self._s1 = a1 * self._s1 + (1.0 - a1) * u
        #     u = self._s1
        # if self._tau2 is not None:
        #     a2 = math.exp(-dt / self._tau2)
        #     self._s2 = a2 * self._s2 + (1.0 - a2) * u
        #     u = self._s2

        # --- amplitude saturation (phase-preserving) ---
        u = u_cmd
        self.saturated = False
        if self.V_max < float("inf"):
            mag = abs(u_cmd)
            if mag > self.V_max and mag > 0:
                u = u_cmd * (self.V_max / mag)
                self.saturated = True
                print("Cavity "+str(self.name)+" is saturated")

        # --- slew-rate limit (phase-vector rate cap) ---
        self.slewing = False
        smax = self.slewing_limit
        if smax is not None and smax > 0 and dt > 0:
            max_step = float(smax) * dt
            du = u - self.last_out
            if abs(du) > max_step and abs(du) > 0:
                u = self.last_out + du / abs(du) * max_step
                self.slewing = True
                print("Cavity "+str(self.name)+" is slewing")

        # --- optional simple AM/PM (linear phase push), disabled by default ---
        # ampm = self.p.am_pm_rad_per_V
        # if ampm is not None and ampm != 0.0:
        #     push = float(ampm) * (abs(u) - float(self.p.am_pm_ref_V))
        #     u = u * complex(math.cos(push), math.sin(push))

        self.last_out = u
        self.last_cmd_err = u_cmd - u   #determines how much was lost through delay, saturation, slewing, etc
        
        return u

@dataclass
class NoiseParams:
    """Generic process/measurement noise (physics + observation, no hardware).

    - process noise: additive band-limited forcing d on the envelope [V/s]
    - measurement noise: additive Gaussian on the *read-out* cavity I/Q [V]
    These replace the old master-oscillator framing; they are generic stochastic
    terms for residual / SINDy realism, not a hardware chain.
    """
    enable_process_noise: bool 
    process_noise_rms_Vps: float       # additive envelope forcing rms [V/s]
    process_noise_bw_Hz: float 
    enable_measurement_noise: bool 
    meas_noise_rms_V: float            # additive readout I/Q noise rms [V]



@dataclass
class DetuningParams:
    """Detuning sub-model flags and coefficients.  All effects toggle here."""
    enable_static: bool
    static_Hz: float

    enable_lfd: bool
    lfd_mode: str
    lfd_modes: list[float, float, float]
    lfd_algebraic_K_HzMV2: float
    persist_lfd: bool 

    enable_microphonics: bool 
    mic_rms_Hz: float 
    mic_bw_Hz: float 
    #mic_lines: Sequence[tuple] 
    enable_random_detuning: bool 
    random_detuning_rms_Hz: float 

@dataclass
class StateSpaceParams:
    """Parameters for a single cavity."""
    #--required--
    name: str
    R_over_Q: float             # Shunt impedance over Q (Ohms)
    f0: float                   # Resonant frequency (Hz)
    Q0: float                   # Unloaded quality factor
    Q_L: float                  # Loaded quality factor
    L_active: float             # Active Length [m]
    Vmax: float                 # Maximum cavity field (V/m)
    BPM_phase_offset: float     # BPM phase reference for each cavity
    
    def __post_init__(self) -> None:
        self.omega0: float = TWO_PI*self.f0                       #natural angular resonant frequency
        self.omega_half: float = self.omega0/(2*self.Q_L)       #half bandwidth (rad/s)
        self.tau: float = 1/self.omega_half                     #field fill/decay time [inverse seconds]
        self.bandwidth_Hz: float = self.omega0/(self.Q_L*TWO_PI)  #cavity bandwidth
        self.R_L: float = self.R_over_Q * self.Q_L              #loaded shunt impedance
        self.Q_ext = 1.0 / (1.0 / self.Q_L - 1.0 / self.Q0)     #external quality factor
        self.beta_c = self.Q0 / self.Q_ext                      #coupling factor beta
        self.Q0_eff = self.Q0                                   #currently effective quality factor is ideal ##########
        self.refl_factor: float = (2.0*self.beta_c)/(1+self.beta_c) #reflection scaling factor (2 for beta_c>>1)

#self._rng, self.det, self.noise, self._lfd, self._lfd_alg, self._mic_lp, self._proc_lp

@dataclass
class StepOutput:
    """Everything produced by one model step.  SI units throughout."""
    t: float
    V_cav: complex
    V_fwd: complex
    V_refl: complex
    V_beam: complex
    I_b: complex
    P_fwd: float
    P_refl: float
    P_beam: float

    static_Hz: float  
    lfd_Hz: float 
    mic_Hz: float
    dyn_Hz: float
    total_detuning_Hz: float
    beam_forcing: complex
    noise_forcing: float

#State Space Cavity Plant
class StateSpaceCavityModel:
    def __init__(
                self,
                allcavityspecs: AllCavitySpecs,
                dt: float,
                rng: Optional[np.random.Generator] = None,
                ):
        self._name = allcavityspecs.name
        rfParams = allcavityspecs._RF_source_params
        self._rf_source = RFSource(RFparams=rfParams, dt=dt)
        self._controller = PIController(PIparams=allcavityspecs._controller_params)
        self.det = allcavityspecs._detuning_params
        self.noise = allcavityspecs._noise_params
        self._rng = rng if rng is not None else np.random.default_rng()

        self.all_specs   = allcavityspecs
        self._setpoint = rfParams.setpoint_phasor
        self.cavParams = allcavityspecs._state_space_params
        self.omega_half  = self.cavParams.omega_half
        self.refl_factor = self.cavParams.refl_factor
        self.x = np.zeros(2, dtype=float)
        self.t = 0.0
        self._proc_lp: complex = 0.0 + 0.0j
        self._mic_lp:  float   = 0.0
        self._lfd_alg: float = 0.0
        self._lfd:     list  = []   # populated when mechanical LFD modes are loaded
        

#Andrew's Primary Code Starts here
    def _detuning_components(self, dt: float) -> tuple:
        static_Hz = self.det.static_Hz if self.det.enable_static else 0.0

        lfd_Hz = 0.0
        if self.det.enable_lfd:
            if self.det.lfd_mode == "mechanical":
                lfd_Hz = sum(s[0] for s in self._lfd) / TWO_PI
            else:
                lfd_Hz = self._lfd_alg

        mic_Hz = 0.0
        rand_Hz = 0.0
        if self.det.enable_microphonics:
            tau_lp = 1.0 / (TWO_PI * self.det.mic_bw_Hz)
            alpha = dt / (tau_lp + dt)
            white = self._rng.standard_normal() * self.det.mic_rms_Hz \
                * math.sqrt(2.0 * tau_lp / dt) if dt > 0 else 0.0
            self._mic_lp += alpha * (white - self._mic_lp)
            mic_Hz += self._mic_lp
            for f_line, amp in self.det.mic_lines:
                mic_Hz += amp * math.sin(TWO_PI * f_line * self.t)
        if self.det.enable_random_detuning and self.det.random_detuning_rms_Hz > 0:
            rand_Hz += self._rng.standard_normal() * self.det.random_detuning_rms_Hz
        return static_Hz, lfd_Hz, mic_Hz, rand_Hz
    
    def lfd_mode_detunings_Hz(self) -> np.ndarray:
        """Return individual LFD mode detuning contributions in Hz."""
        if not self.det.enable_lfd:
            return np.zeros(len(self.det.lfd_modes), dtype=float)
    
        if self.det.lfd_mode == "mechanical":
            return np.array([s[0] / TWO_PI for s in self._lfd], dtype=float)
    
        # Algebraic LFD has no modal split, so expose it as one pseudo-mode.
        return np.array([self._lfd_alg], dtype=float)

    def _process_forcing(self, dt: float) -> complex:
    # ---- process noise -------------------------------------------------------
        if not (self.noise.enable_process_noise and self.noise.process_noise_rms_Vps > 0):
            return 0.0 + 0.0j
        tau_lp = 1.0 / (TWO_PI * self.noise.process_noise_bw_Hz)
        alpha = dt / (tau_lp + dt)
        scale = self.noise.process_noise_rms_Vps * math.sqrt(2.0 * tau_lp / dt) if dt > 0 else 0.0
        w = (self._rng.standard_normal() + 1j * self._rng.standard_normal()) * scale
        self._proc_lp += alpha * (w - self._proc_lp)
        return self._proc_lp
    
    def A_matrix(self, dw: float) -> np.ndarray:
        oh = self.cavParams.omega_half
        return np.array([[-oh, -dw], [dw, -oh]], dtype=float)

    @property
    def B_matrix(self) -> np.ndarray:
        return self.cavParams.omega_half * np.eye(2)
    
    def _discretize_zoh(self, A: np.ndarray, dt: float):
        """Exact zero-order-hold discretization via a block matrix exponential.

        Builds M = [[A, I],[0, 0]] (4x4) so that expm(M dt) = [[Phi, Gamma],
        [0, I]], giving both the state-transition matrix Phi = expm(A dt) and
        the forcing-integral matrix Gamma = int_0^dt expm(A s) ds in one call.
        This avoids inverting A (which is singular only if omega_half = 0).
        """
        n = A.shape[0]
        M = np.zeros((2 * n, 2 * n), dtype=float)
        M[:n, :n] = A
        M[:n, n:] = np.eye(n)
        E = expm(M * dt)
        Phi = E[:n, :n]
        Gamma = E[:n, n:]
        return Phi, Gamma
    
    def _propagate(self, x: np.ndarray, A: np.ndarray, B: np.ndarray,
                u: complex, V_beam: complex, d: complex, dt: float):
        """Advance x by dt under constant forcing; return (x_new, xdot_new)."""
        f = (B @ np.array([u.real, u.imag])
                + B @ np.array([V_beam.real, V_beam.imag])
                + np.array([d.real, d.imag]))

        # Cache key must include ALL parameters of A:
        #   A[0,0] = A[1,1] = -omega_half  (changes when Q_L changes)
        #   A[0,1] = -dw, A[1,0] = +dw     (changes with detuning)
        # The original key omitted A[0,0], so a Q_L change during fitting
        # would reuse a Phi/Gamma computed for a different omega_half while
        # f = B @ u used the new omega_half — mismatched propagation.
        key = (round(dt, 18), round(A[0, 0], 12), round(A[0, 1], 12))
        cache = getattr(self, "_zoh_cache", None)
        if cache is None or cache[0] != key:
            Phi, Gamma = self._discretize_zoh(A, dt)
            self._zoh_cache = (key, Phi, Gamma)
        else:
            _, Phi, Gamma = cache
        x_new = Phi @ x + Gamma @ f
        
        xdot_new = A @ x_new + f      # analytic derivative at the new state

        return x_new, xdot_new

    def _advance_lfd(self, V_mag: float, dt: float) -> None:
        E2 = (V_mag / self.cavParams.L_active / 1.0e6) ** 2     # (MV/m)^2
        if self.det.lfd_mode == "mechanical":
            for s, mode in zip(self._lfd, self.det.lfd_modes):
                wm = TWO_PI * mode.f_m_Hz
                drive = -TWO_PI * mode.K_m_HzMV2 * wm * wm * E2   # K>0 -> df<0
                accel = drive - (wm / mode.Q_m) * s[1] - wm * wm * s[0]
                s[1] += accel * dt
                s[0] += s[1] * dt
        else:
            self._lfd_alg = -self.det.lfd_algebraic_K_HzMV2 * E2
    
    @property
    def V_cav(self) -> complex:
        return self.x[0] + 1j * self.x[1]

    def step(self, 
             dt: float, 
             u_ff,
             setpoint: complex, 
             klystron_active: bool,
             beam_current: Optional[float]=None,
             beam_phase: Optional[float]=None, 
             ) -> dict[str, dict[str, float]]:
        if klystron_active:
            I_beam = 0.0 + 0.0j
            V_beam = 0.0 + 0.0j
            if beam_current is not None:
                I_beam = beam_current * np.exp(1j*beam_phase)
                V_beam = -self.all_specs.alpha_beam * self.all_specs._state_space_params.R_L * I_beam

            static_Hz, lfd_Hz, mic_Hz, rand_Hz = self._detuning_components(dt)
            lfd_modes_Hz = self.lfd_mode_detunings_Hz() #This doesn't do anything?

            dyn_Hz = lfd_Hz + mic_Hz + rand_Hz
            total_detuning_Hz = static_Hz + dyn_Hz
            dw = TWO_PI * total_detuning_Hz
            d = self._process_forcing(dt) #what does this do

            A = self.A_matrix(dw)
            B = self.B_matrix
            # phasors
            V_fwd = self._rf_source.drive(dt=dt, setpoint=setpoint, 
                                V_meas=self.V_cav, 
                                u_ff=u_ff, 
                                controller=self._controller,)
        else:
            I_beam = 0.0 + 0.0j
            V_beam = 0.0 + 0.0j
            static_Hz, lfd_Hz, mic_Hz, rand_Hz = self._detuning_components(dt)
            dyn_Hz = lfd_Hz + mic_Hz + rand_Hz
            total_detuning_Hz = static_Hz + dyn_Hz   
            dw = TWO_PI * total_detuning_Hz          
            d = self._process_forcing(dt)
            A = self.A_matrix(dw)
            B = self.B_matrix
            V_fwd = 0.0

        # State-space propagation over the step (method selected at construction:
        # "expm" exact ZOH | "rk45" scipy solve_ivp | "rk4" legacy fixed-step).
        self.x, xdot = self._propagate(self.x, A, B, V_fwd, V_beam, d, dt)
        self._advance_lfd(abs(self.V_cav), dt)
        self.t += dt
        V_cav = self.V_cav
        V_refl = self.refl_factor * V_cav - V_fwd
        R_L = self.all_specs._state_space_params.R_L
        # powers
        P_fwd  = np.abs(((V_fwd) ** 2) / (4.0 * R_L))
        P_refl = np.abs(((V_refl) ** 2) / (4.0 * R_L))
        # beam power: (1/2) Re(V_c I_b*) — uses I_b directly (externally-set phasor)
        S_beam = 0.5 * V_cav * I_beam
        P_beam = np.real(S_beam)

        return StepOutput(
            t=self.t,
            V_cav=V_cav, V_fwd=V_fwd, V_refl=V_refl, V_beam=V_beam, I_b=I_beam,
            P_fwd=P_fwd, P_refl=P_refl, P_beam=P_beam,
            static_Hz=static_Hz, mic_Hz=mic_Hz, lfd_Hz=lfd_Hz,
            dyn_Hz=dyn_Hz, total_detuning_Hz=total_detuning_Hz,
            beam_forcing=self.omega_half * V_beam, noise_forcing=d,
        )