import torch
import torch.nn as nn
from lif.spike_activation import SpikeActivation
import torch.nn.functional as F



class LILayer(nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 batch_size,
                 leak=0.9,
                 ):
        
        super(LILayer, self).__init__()

        self.register_buffer('membrane_potential', torch.zeros(batch_size, output_size))
        self.leak = leak
        self.fc = nn.Linear(input_size, output_size, bias=False) 
        self.recurrent =  False

    def _resize_state(self, B: int, device) -> None:
        if self.membrane_potential.shape[0] != B:
            self.membrane_potential = torch.zeros(B, self.membrane_potential.shape[1], device=device)

    def forward(self, x):
        B = x.size(0)
        self._resize_state(B, x.device)
        input_current = self.fc(x)
        self.membrane_potential = self.leak * self.membrane_potential + input_current
        return self.membrane_potential
    
    def reset(self):
        self.membrane_potential = torch.zeros_like(
            self.membrane_potential,
            device=self.membrane_potential.device,
        )

class LIFLayer(nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 batch_size,
                 vth = 1.0, 
                 leak_m=0.9,
                 leak_t=0.9,
                 recurrent = False,
                 reset_type = "soft",
                 surrogate_type = "2",
                 surrogate_scale = 1.0,
                 ):
        
        super(LIFLayer, self).__init__()

        self.register_buffer('membrane_potential', torch.zeros(batch_size, output_size))
        self.register_buffer('previous_spike', torch.zeros(batch_size, output_size))
        self.register_buffer('trace', torch.zeros(batch_size, output_size))

        self.reset_type = reset_type
        self.vth = vth
        self.surrogate_type = surrogate_type
        self.activation = SpikeActivation(self.vth, self.surrogate_type, surrogate_scale, output_size)
        self.recurrent = recurrent

        self.leak_m = leak_m
        self.leak_t = leak_t
        self.fc = nn.Linear(input_size, output_size, bias=False) 

        if self.recurrent:
            self.rc = nn.Linear(output_size, output_size, bias = False)
            

    def _resize_state(self, B: int, device) -> None:
        if self.membrane_potential.shape[0] != B:
            H = self.membrane_potential.shape[1]
            self.membrane_potential = torch.zeros(B, H, device=device)
            self.trace              = torch.zeros(B, H, device=device)
            self.previous_spike     = torch.zeros(B, H, device=device)

    def forward(self, x):
        B = x.size(0)
        self._resize_state(B, x.device)
        input_current = self.fc(x)

        if self.recurrent: 
            input_current += self.rc(self.previous_spike)
        
        # 1) Leak + drive
        membrane_potential_pre = self.leak_m * self.membrane_potential + input_current
        # 2) Fire
        spike = self.activation(membrane_potential_pre)
        tmp = spike
            
        # 3) Reset (soft vs hard)
        if self.reset_type == "soft":
            membrane_potential_new = membrane_potential_pre - spike * self.vth
        else:  # hard reset
            membrane_potential_new = membrane_potential_pre * (1 - spike)
        # 4) Update membrane potential for next time step 
        self.membrane_potential = membrane_potential_new
        # 5) Save spike for recurent connection to be integrated in the next time step
        self.previous_spike = spike.clone()
        # 6) Compute trace
        self.trace = self.trace * self.leak_t + tmp

        return spike, membrane_potential_new, self.trace
    
    def reset(self):
        self.membrane_potential = torch.zeros_like(
            self.membrane_potential,
            device=self.membrane_potential.device,
        )
        self.trace = torch.zeros_like(
            self.trace,
            device=self.trace.device,
        )
        self.previous_spike = torch.zeros_like(
            self.membrane_potential,
            device=self.trace.device,
        )

