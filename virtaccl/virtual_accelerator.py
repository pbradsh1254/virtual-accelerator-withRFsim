import sys
import time
import argparse
import numpy as np
from datetime import datetime
from importlib.metadata import version
from typing import Dict, Any, List, TypeVar, Generic

from virtaccl.server import Server, not_ctrlc
from virtaccl.beam_line import BeamLine
from virtaccl.model import Model
from virtaccl.rf_sim_lib import CavityChain, SimulationParams
from virtaccl.graphing import LiveDashboard, record_cav_pulse_step, record_beam_pulse_step, block_until_closed
import json

class VA_Parser:
    def __init__(self):
        self._va_arguments_: Dict[str, Dict[str, Any]] = {}
        self.model_arguments: Dict[str, Dict[str, Any]] = {}
        self.server_arguments: Dict[str, Dict[str, Any]] = {}
        self.custom_arguments: Dict[str, Dict[str, Any]] = {}
        self.__all_arguments__ = {'va': self._va_arguments_, 'model': self.model_arguments,
                                  'server': self.server_arguments, 'custom': self.custom_arguments}
        self.__all_argument_keys__ = set()

        self.version = version('virtaccl')
        self.description = 'Run the Virac virtual accelerator server.'

        add_va_arguments(self)

    def __find_argument_dict__(self, name) -> Dict[str, Dict[str, Any]]:
        for argument_group, arguments in self.__all_arguments__.items():
            if name in arguments:
                return arguments

    def set_description(self, new_description: str):
        self.description = new_description

    def add_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self.custom_arguments[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def add_va_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self._va_arguments_[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def add_model_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self.custom_arguments[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def add_server_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self.custom_arguments[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def remove_argument(self, name: str):
        if name not in self.__all_argument_keys__:
            print(f'Warning: Argument name "{name}" was not found.')
        else:
            arguments = self.__find_argument_dict__(name)
            del arguments[name]
            self.__all_argument_keys__.remove(name)

    def edit_argument(self, name: str, new_options: Dict[str, Any]):
        if name not in self.__all_argument_keys__:
            print(f'Warning: Argument name "{name}" was not found.')
        else:
            arguments = self.__find_argument_dict__(name)
            for option_key, new_value in new_options.items():
                arguments[name]['optional'][option_key] = new_value

    def change_argument_default(self, name: str, new_value: Any):
        arguments = self.__find_argument_dict__(name)
        arguments[name]['optional']['default'] = new_value

    def change_argument_help(self, name: str, new_help: Any):
        arguments = self.__find_argument_dict__(name)
        arguments[name]['optional']['help'] = new_help

    def initialize_arguments(self) -> Dict[str, Any]:
        va_parser = argparse.ArgumentParser(
            description = self.description + ' Version ' + self.version,
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        for group_key, argument_group in self.__all_arguments__.items():
            for argument_name, argument_dict in argument_group.items():
                va_parser.add_argument(*argument_dict['positional'], **argument_dict['optional'])
        return vars(va_parser.parse_args())


def add_va_arguments(va_parser: VA_Parser) -> VA_Parser:
    # Number (in Hz) determining the update rate for the virtual accelerator.
    va_parser.add_va_argument('--refresh_rate', default = 1.0, type = float,
                              help = 'Rate (in Hz) at which the virtual accelerator updates.')
    va_parser.add_va_argument('--sync_time', dest= 'sync_time', action = 'store_true',
                              help = "Synchronize timestamps for server parameters.")

    # Desired amount of output.
    va_parser.add_va_argument('--debug', dest = 'debug', action = 'store_true',
                              help = "Some debug info will be printed.")
    va_parser.add_va_argument('--production', dest = 'debug', action = 'store_false',
                              help = "DEFAULT: No additional info printed.")

    va_parser.add_server_argument('--print_server_keys', action = 'store_true',
                                  help = "Will print all server keys for the server. Will NOT run the virtual "
                                       "accelerator.")
    va_parser.add_server_argument('--print_settings', action = 'store_true',
                                  help = "Will only print setting keys for the server. Will NOT run the virtual "
                                       "accelerator.")

    return va_parser


# Define a TypeVar constrained to Model
ModelType = TypeVar('ModelType', bound = 'Model')
ServerType = TypeVar('ServerType', bound = 'Server')


class VirtualAcceleratorBuilder(Generic[ModelType, ServerType]):
    def __init__(self, model: ModelType, beam_line: BeamLine, server: ServerType, **kwargs):
        self.model = model
        self.beam_line = beam_line
        self.server = server
        self.options = kwargs

    def get_model(self) -> ModelType:
        return self.model

    def get_beamline(self) -> BeamLine:
        return self.beam_line

    def get_server(self) -> ServerType:
        return self.server

    def build(self) -> 'VirtualAccelerator[ModelType, ServerType]':
        return VirtualAccelerator(self.model, self.beam_line, self.server, **self.options)


class VirtualAccelerator(Generic[ModelType, ServerType]):
    def __init__(self, model: ModelType, beam_line: BeamLine, server: ServerType, **kwargs):
        if not kwargs:
            kwargs = VA_Parser().initialize_arguments()

        if kwargs['print_settings']:
            for key in beam_line.get_setting_keys():
                print(key)
            sys.exit()

        if kwargs['print_server_keys']:
            for key in beam_line.get_all_keys():
                print(key)
            sys.exit()

        self.sync_time = kwargs['sync_time']
        self.update_period = 1 / kwargs['refresh_rate']

        self.model = model
        self.beam_line = beam_line
        self.server = server

        sever_parameters = beam_line.get_server_parameter_definitions()
        server.add_parameters(sever_parameters)
        beam_line.reset_devices()

        if kwargs['debug']:
            print(server)

        self.track()

    def get_model(self) -> ModelType:
        return self.model

    def get_beamline(self) -> BeamLine:
        return self.beam_line

    def get_server(self) -> ServerType:
        return self.server

    def set_value(self, server_key: str, new_value):
        self.server.set_parameter(server_key, new_value)
        self.track()

    def set_values(self, new_settings: Dict[str, Any]):
        self.server.set_parameters(new_settings)
        self.track()

    def get_value(self, *server_key: str):
        if len(server_key) == 1:
            return self.server.get_parameter(server_key[0])
        else:
            return tuple(self.server.get_parameter(key) for key in server_key)

    def get_values(self, value_keys: List[str] = None) -> Dict[str, Any]:
        if value_keys is not None:
            return_dict = {}
            for key in value_keys:
                return_dict |= {key: self.server.get_parameter(key)}
        else:
            return_dict = self.server.get_parameters()
        return return_dict

    def track(self, timestamp: datetime = None, server_optics = None):

        server_params = self.server.get_parameters()
        self.beam_line.update_settings_from_server(server_params)

        if server_optics is not None:
            self.model.update_optics(server_optics)
        else:
            server_optics = self.beam_line.get_model_optics()
            self.model.update_optics(server_optics)        

        self.beam_line.update_readbacks()
        self.model.track()

        server_measurements = self.model.get_measurements()
        self.beam_line.update_measurements_from_model(server_measurements)
        new_server_values = self.beam_line.get_parameters_for_server()
        self.server.set_parameters(new_server_values, timestamp = timestamp)

    def start_server(self):
        self.server.start()
        print(f"Server started.")
        now = None

        # Our new data acquisition routine
        while not_ctrlc():
            loop_start_time = time.time()

            if self.sync_time:
                now = datetime.now()
            self.track(timestamp = now)
            self.server.update()

            loop_time_taken = time.time() - loop_start_time
            sleep_time = self.update_period - loop_time_taken
            if sleep_time < 0.0:
                print('Warning: Update took longer than refresh rate.')
            else:
                time.sleep(sleep_time)

        print('Exiting. Thank you for using our virtual accelerator!')

    def start_server_withRF(self):
        self.server.start()
        print(f"Server started.")
        
        simParams = SimulationParams(
        dt = 3e-6,
        fill_duration = 250e-6,
        flattop_duration = 300e-6,
        beam_on_time = 300e-6,
        )
        chain = CavityChain.from_json(simParams, "/home/hitesh/virtual-accelerator-withRFsim/cavityparameters.json")
        fill_data = chain.fill()

        now = None
        server_optics = self.beam_line.get_model_optics()
        self.track(timestamp = now, server_optics = server_optics)
        self.server.update()

        all_cav_hist = {}
        all_BPM_hist = {}

        for cav, data in fill_data.items():
            refl_IQ = np.atleast_1d(data['refl_iq'])
            all_cav_hist[cav] = {
                't':          list(np.atleast_1d(data['t'])),
                'amp':        list(np.abs(np.atleast_1d(data['cav_iq']))),
                'phase':      list(np.angle(np.atleast_1d(data['cav_iq']))),
                'fwd_phase':  list(np.angle(np.atleast_1d(data['fwd_iq']))),
                'fwd_amp':    list(np.abs(np.atleast_1d(data['fwd_iq']))),
                'refl_amp':   list(np.abs(refl_IQ)),
                'refl_phase': list(np.angle(refl_IQ)),
            }

        cav_name = list(fill_data.keys())

        server_measurements = self.model.get_measurements()

        BPM_name = [name for name in server_measurements.keys() if 'SCL_Diag:BPM' in name]

        dashboard_cav = LiveDashboard(
            cav_name,
            titles=['amp', 'Phase', 'fwd_amp', 'fwd_phase', 'refl_amp', 'refl_phase'],
            colors=['blue', 'orange', 'green', 'red', 'purple', 'brown'],
            data_keys=['amp', 'phase', 'fwd_amp', 'fwd_phase', 'refl_amp', 'refl_phase'],
            phase_keys={'phase', 'fwd_phase', 'refl_phase'},
            layout=(3, 2),
            window_title="Cavity Dashboard",
        )
        dashboard_beam = LiveDashboard(
            BPM_name,
            titles=['phase_avg', 'current'],
            colors=['blue', 'red'],
            data_keys=['phase_avg', 'current'],
            phase_keys={'phase_avg'},
            layout=(2, 1),
            window_title="Beam Dashboard",
        )

        dashboard_cav.push(all_cav_hist)
        dashboard_beam.push(all_BPM_hist)

        trn_count = 0
        beam_start = simParams.beam_on_time
        chain.switch_feedback(True)

        while chain._pulse_time < (simParams.fill_duration + simParams.flattop_duration) and not_ctrlc():
            loop_start_time = time.time()
            # print(chain._pulse_time)
            # print()
            # print(simParams.fill_duration + simParams.flattop_duration)
            # print()
            if self.sync_time:
                now = datetime.now()

            if chain._pulse_time < beam_start:
                step_data = chain.flattop_step()   
            else:
                server_measurements = self.model.get_measurements()
                server_optics = self.beam_line.get_model_optics()
                beam_cur = {}
                beam_phi = {}
                
                for cav in cav_name:
                    if 'SCL:Cav' in cav:
                        offset_dict = chain._bpm_offsets[cav]
                        print(offset_dict)                
                        for bpmPV in offset_dict:
                            beam_cur = beam_cur | {cav: offset_dict[bpmPV]}
                            cav_phi_deg = np.degrees(2* server_measurements[bpmPV] + offset_dict[bpmPV])- 20.5
                            cav_phi = (cav_phi_deg + np.pi) % (2 * np.pi) - np.pi
                            beam_phi = beam_phi | {cav: cav_phi} #must change from 402.5 to 805 MHz, offset is structured this way as well in createcavlists.

                step_data = chain.flattop_step(beam_currents = beam_cur, beam_phases = beam_phi)

            for cav, data in step_data.items():
                server_optics[cav]['amp'] = np.abs(data['cav_iq']) / 15e6 #This needs to be fixed to actually represent a real normal value
                server_optics[cav]['phase'] = np.angle(data['cav_iq'])

            self.track(timestamp = now, server_optics = server_optics)
            self.server.update()

            server_measurements = self.model.get_measurements()
            all_BPM_hist = record_beam_pulse_step(server_measurements, chain._pulse_time, all_BPM_hist)
            dashboard_beam.push(all_BPM_hist)

            all_cav_hist = record_cav_pulse_step(step_data, all_cav_hist)
            dashboard_cav.push(all_cav_hist)

            trn_count += 1
            if (trn_count) % 10 == 0:
                print(str(trn_count) + ' timesteps completed')
# =============================================================================
#       loop_time_taken = time.time() - loop_start_time
#       sleep_time = self.update_period - loop_time_taken
#       if sleep_time < 0.0:
#           print('Warning: Update took longer than refresh rate.')
#       else:
#           time.sleep(sleep_time)
# =============================================================================      

        chain.switch_feedback(False)  
        decay_data  = chain.decay(n_tau=5)
        #add data to graphing list
        #graphing functions based on cavity  
        for cav, data in decay_data.items():
            cav_iq_arr = np.atleast_1d(data['cav_iq'])     
            fwd_iq_arr = np.atleast_1d(data['fwd_iq'])     
            refl_iq_arr = np.atleast_1d(data['refl_iq'])    
            t_arr = np.atleast_1d(data['t'])

            hist = all_cav_hist[cav]
            hist['t'].extend(t_arr)
            hist['amp'].extend(np.abs(cav_iq_arr))
            hist['phase'].extend(np.angle(cav_iq_arr))
            hist['fwd_phase'].extend(np.angle(fwd_iq_arr))
            hist['fwd_amp'].extend(np.abs(fwd_iq_arr))
            hist['refl_amp'].extend(np.abs(refl_iq_arr))
            hist['refl_phase'].extend(np.angle(refl_iq_arr))

        # finalize() converts the accumulated lists to numpy arrays and does one
        # last full-quality render; block_until_closed() keeps both windows open.
        dashboard_cav.finalize(all_cav_hist)
        dashboard_beam.finalize(all_BPM_hist)

        block_until_closed()

        print('Exiting. Thank you for using our virtual accelerator!')


