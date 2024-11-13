import numpy as np
import uproot as up
import os
import torch
import time
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''RUNTIME PROFILING'''
import cProfile
import pstats
from functools import wraps
from io import StringIO
from contextlib import contextmanager

def profile_function(func):
    """
    Decorator to profile a specific function using cProfile
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(func, *args, **kwargs)
        finally:
            s = StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats(20)  # Print top 20 time-consuming operations
            print(s.getvalue())
    return wrapper

'''END RUNTIME PROFILING'''
'''BEGIN MEMORY PROFILING'''
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

'''END MEMORY PROFILING'''
'''NORMALIZING FLOW STUFF'''
import normflows as nf

def get_compiled_NF_model(model_path):
    run_num = 7
    run_num_str = str(run_num)

    #NF Stuff

    K = 8 #num flows

    latent_size = 1 #dimension of PDF
    hidden_units = 256 #nodes in hidden layers
    hidden_layers = 26
    context_size = 3 #conditional variables for PDF
    num_context = 3

    K_str = str(K)
    batch_size= 2000
    hidden_units_str = str(hidden_units)
    hidden_layers_str = str(hidden_layers)
    batch_size_str = str(batch_size)
    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                                 num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set base distribution
    q0 = nf.distributions.DiagGaussian(1, trainable=False)

    # Construct flow model
    model = nf.ConditionalNormalizingFlow(q0, flows)

    model.load(model_path)
    return torch.compile(model,mode = "reduce-overhead").to(device)

'''Some util functions'''
c = 299792458 # 2.998 * 10 ^ 8 m/s
c_n = 1 #c = 1 in natural units
def time_func(p,m,dx):
    p_div_m = p / m
    vc = p_div_m * np.sqrt(1 / (1 + ((p_div_m) ** 2) * (1 / (c_n ** 2)))) # in terms of c
    v = vc * c #now in m/s
    v_mm = v * 1000 # in mm/s
    v_mmpns = v_mm / (10 ** (9)) # in mm/ns
    return dx / v_mmpns
import datetime

def print_w_time(message):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"{current_time} {message}")
'''END some util functions'''

'''SIPM SIGNAL PROCESSING'''
class SiPMSignalProcessor:
    def __init__(self, 
                 sampling_rate=40e9,  # 40 GHz sampling rate
                 tau_rise=1e-9,       # 1 ns rise time
                 tau_fall=10e-9,      # 50 ns fall time
                 window=200e-9,       # 200 ns time window
                 cfd_delay=5e-9,      # 5 ns delay for CFD
                 cfd_fraction=0.3):   # 30% fraction for CFD
        
        self.sampling_rate = sampling_rate
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall
        self.window = window
        self.cfd_delay = cfd_delay
        self.cfd_fraction = cfd_fraction
        
        # Time array for single pulse shape
        self.time = np.arange(0, self.window, 1/self.sampling_rate)
        
        # Generate single pulse shape
        self.pulse_shape = self._generate_pulse_shape()
    
    def _generate_pulse_shape(self):
        """Generate normalized pulse shape for a single photon"""
        shape = (1 - np.exp(-self.time/self.tau_rise)) * np.exp(-self.time/self.tau_fall)
        return shape / np.max(shape)  # Normalize
    
    def generate_waveform(self, photon_times):
        """Generate waveform from list of photon arrival times"""
        # Initialize waveform array
        waveform = np.zeros_like(self.time)
        
        # Add pulse for each photon
        for t in photon_times:
            if 0 <= t < self.window:
                idx = int(t * self.sampling_rate)
                remaining_samples = len(self.time) - idx
                waveform[idx:] += self.pulse_shape[:remaining_samples]
        
        return self.time, waveform
    
    def integrate_charge(self, waveform, integration_start=0, integration_time=100e-9):
        """Integrate charge in specified time window"""
        start_idx = int(integration_start * self.sampling_rate)
        end_idx = int((integration_start + integration_time) * self.sampling_rate)
        
        # Integrate using trapezoidal rule
        charge = np.trapezoid(waveform[start_idx:end_idx], dx=1/self.sampling_rate)
        return charge
    def constant_threshold_timing(self,waveform,threshold):
        for i in range(len(self.time)):
            if(waveform[i] > threshold):
                return self.time[i]
        return -1
        
    def apply_cfd(self, waveform, use_interpolation=True):
        """Apply Constant Fraction Discrimination to the waveform.

        Parameters:
        -----------
        waveform : numpy.ndarray
            Input waveform to process
        use_interpolation : bool, optional
            If True, use linear interpolation for sub-sample precision
            If False, return the sample index of zero crossing
            Default is True

        Returns:
        --------
        tuple (numpy.ndarray, float)
            CFD processed waveform and the zero-crossing time in seconds.
            If use_interpolation is False, zero-crossing time will be aligned
            to sample boundaries.
        """
        # Calculate delay in samples
        delay_samples = int(self.cfd_delay * self.sampling_rate)

        # Create delayed and attenuated versions of the waveform
        delayed_waveform = np.pad(waveform, (delay_samples, 0))[:-delay_samples]
        attenuated_waveform = -self.cfd_fraction * waveform

        # Calculate CFD waveform
        cfd_waveform = delayed_waveform + attenuated_waveform

        # Find all zero crossings
        zero_crossings = np.where(np.diff(np.signbit(cfd_waveform)))[0]

        if len(zero_crossings) < 2:  # Need at least two crossings for valid CFD
            return cfd_waveform, None

        # Find the rising edge of the original pulse
        pulse_start = np.where(waveform > np.max(waveform) * 0.1)[0]  # 10% threshold
        if len(pulse_start) == 0:
            return cfd_waveform, None
        pulse_start = pulse_start[0]

        # Find the first zero crossing that occurs after the pulse starts
        valid_crossings = zero_crossings[zero_crossings > pulse_start]
        if len(valid_crossings) == 0:
            return cfd_waveform, None

        crossing_idx = valid_crossings[0]

        if not use_interpolation:
            # Simply return the sample index converted to time
            crossing_time = crossing_idx / self.sampling_rate
        else:
            # Use linear interpolation for sub-sample precision
            y1 = cfd_waveform[crossing_idx]
            y2 = cfd_waveform[crossing_idx + 1]

            # Calculate fractional position of zero crossing
            fraction = -y1 / (y2 - y1)

            # Calculate precise crossing time
            crossing_time = (crossing_idx + fraction) / self.sampling_rate

        return cfd_waveform, crossing_time


    def get_pulse_timing(self, waveform, threshold=0.1):
        """Get pulse timing using CFD method with additional validation.
        
        Parameters:
        -----------
        waveform : numpy.ndarray
            Input waveform to analyze
        threshold : float
            Minimum amplitude threshold for valid pulses (relative to max amplitude)
            
        Returns:
        --------
        float or None
            Timestamp of the pulse in seconds, or None if no valid pulse found
        """
        # Check if pulse amplitude exceeds threshold
        max_amplitude = np.max(waveform)
        if max_amplitude < threshold:
            return None
            
        # Apply CFD
        _, crossing_time = self.apply_cfd(waveform)
        
        return crossing_time
    '''SIPM END'''
'''Main function'''
@profile_function
def generateSiPMOutput(processed_data, normalizing_flow, batch_size=1024, device='cuda',pixel_threshold = 3):
    out_columns = ['event_idx','stave_idx','layer_idx','segment_idx','trueID','truePID','hitID','hitPID','P','Theta','Phi','strip_x','strip_y','strip_z','Charge1','Time1','Charge2','Time2']
    rows = []
    processor = SiPMSignalProcessor()
    normflow_input = []
    num_pixel_list = ["num_pixels_high_z","num_pixels_low_z"]
    print_w_time("Processing data in generateSiPMOutput...")
    got_row_indexes = False
    labels = []
    for row_idx, row in processed_data.iterrows():
        for SiPM_idx in range(2):
            num_pixel_tag = num_pixel_list[SiPM_idx] #get the name of the number of pixels for this SiPM
            #create a new row for each of the photons/pixels
            row_copy = row.copy()
            row_copy['SiPM_idx'] = SiPM_idx
            if(not got_row_indexes):
                labels = row_copy.index.to_list()
                got_row_indexes = True
            normflow_input.append(torch.tensor(row_copy.values,dtype=torch.float32).repeat(int(row_copy[num_pixel_tag]), 1))
    #Create a dict like{label : index in row}
    label_dict = {}
    for i in range(len(labels)):
        label_dict[labels[i]] = i
        
    normflow_input_tensor = torch.cat(normflow_input)

    data = []
    print_w_time(f"starting sampling")
    begin = time.time()
#     for i in tqdm(range(0, len(normflow_input_tensor), batch_size)):
    for i in range(0, len(normflow_input_tensor), batch_size):
        print_w_time(f"Starting batch # {(i / batch_size) + 1} / {int(np.ceil(len(normflow_input_tensor) / batch_size))}")
        batch_end = min(i + batch_size, len(normflow_input_tensor))
        batch_rows = normflow_input_tensor[i:batch_end]
        context_indexes = [label_dict["z_pos"],label_dict["hittheta"],label_dict["hitmomentum"]]
        time_index = label_dict["time"]
        batch_context = batch_rows[:,context_indexes].to(device)
        batch_particle_hit_times = batch_rows[:,time_index]
        
        with torch.no_grad():
            samples = abs(normalizing_flow.sample(num_samples=len(batch_context), context=batch_context)[0]).squeeze(1).cpu() + batch_particle_hit_times
        batch_combined_data = torch.cat((batch_rows,samples.unsqueeze(-1)),dim = 1)
        data.extend(batch_combined_data)
    end = time.time()
    print_w_time(f"sampling took {(end - begin) / 60} minutes")
    print_w_time("creating df")
    labels.append("photon_time")
    row_df = pd.DataFrame(data,columns = labels)
    print_w_time("Beginning pulse process")
    running_index = 0
    for (event_idx, stave_idx, layer_idx, segment_idx),group in row_df.groupby(['event_idx', 'stave_idx', 'layer_idx','segment_idx']):
#         print(f"Starting row # {running_index}\nevent #{event_idx}, stave #{stave_idx}, layer #{layer_idx}, segment #{segment_idx}")
        charge_times = torch.tensor([[0.0,0.0],[0.0,0.0]])
        set_event_details = False
        trigger = False
        trueID_list_len_max = -1
        for SiPM_idx, SiPM_group in group.groupby(['SiPM_idx']):
            #need to see if we get more than 1 hit in a segment - greg says to label this as noise
            trueID_list_len = len(set(SiPM_group["trueID"]))
            trueID_list_len_max = max(trueID_list_len,trueID_list_len_max)
            photon_times = torch.tensor(sorted(SiPM_group['photon_time'])) * 10 **(-9)
            #get relative times
            if(len(photon_times) > 0):
                #calculate time and charge
                time_arr,waveform = processor.generate_waveform(photon_times)
                timing = processor.get_pulse_timing(waveform,threshold = pixel_threshold)
                if(timing is not None):
                    curr_charge = processor.integrate_charge(waveform) * 1e6
                    curr_timing = timing * 1e8

                    charge_times[SiPM_idx][0] = processor.integrate_charge(waveform) * 1e6
                    charge_times[SiPM_idx][1] = timing * 1e8
#                                 print(f"SiPM idx {SiPM_idx} triggered, (time,charge) : ({curr_timing},{curr_charge})")
                    trigger = True
                else: #no trigger, don't set details yet
                    continue
                if(not set_event_details):
                    #take the 0th one - if there are multiple, then these are noise anyways
                    P = SiPM_group['truemomentum'][0]
                    trueID = SiPM_group['trueID'][0]
                    truePID = SiPM_group['truePID'][0]
                    hitID = SiPM_group['hitID'][0]
                    hitPID = SiPM_group['hitPID'][0]
                    theta = SiPM_group['truetheta'][0]
                    phi = SiPM_group['truephi'][0]
                    strip_x = SiPM_group['strip_x'][0]
                    strip_y = SiPM_group['strip_y'][0]
                    strip_z = SiPM_group['strip_z'][0]
                    set_event_details = True
            else: #no photons, no data
                continue
        if(not set_event_details):
            continue
        if (not trigger):
            continue;
        noise = False
        if(trueID_list_len_max > 1):
            noise = True
        if(not noise):
            if(trueID_dict[event_idx][trueID.item()] == -1):
                trueID_dict[event_idx][trueID.item()] = trueID_dict_running_idx
                trueID_dict_running_idx += 1
            translated_trueID = trueID_dict[event_idx][trueID.item()]
        else:
            translated_trueID = -1
        new_row = { 
           out_columns[0] : event_idx,
           out_columns[1] : stave_idx,
           out_columns[2] : layer_idx,
           out_columns[3] : segment_idx,
           out_columns[4] : translated_trueID, 
           "original_trueID" : trueID.item(), 
           out_columns[5] : truePID.item(), 
           out_columns[6] : hitID.item(),
           out_columns[7] : hitPID.item(),
           out_columns[8] : P.item(), 
           out_columns[9] : theta.item(), 
          out_columns[10] : phi.item(), 
          out_columns[11] : strip_z.item(), 
          out_columns[12] : strip_x.item(), 
          out_columns[13] : strip_y.item(), 
          out_columns[14] : charge_times[0,0].item(), 
          out_columns[15] : charge_times[0,1].item(), 
          out_columns[16] : charge_times[1,0].item(), 
          out_columns[17] : charge_times[1,1].item(),
        }
        rows.append(new_row)
        running_index += 1
    return pd.DataFrame(rows,columns = out_columns)


    
