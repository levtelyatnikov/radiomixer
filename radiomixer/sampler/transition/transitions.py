import numpy as np
from radiomixer.sampler.transition.transition import TransitionSampler
from radiomixer.transforms.transform import TransformType


class TransitionOverlapedSegmentsParametersSampler(TransitionSampler):
    """Class aims to generate parameters of transition for an audio"""

    def __init__(self, configs):
        super().__init__(TransformType.TRANSITIONSOVERLAPEDSEGMENTSPARAMETERSSAMPLER)
        self.configs = configs
      
    def _sampler(self, signal):
        """Transition generator

        Generate transitions parameters for each signal independently.
        Hence each signal has its own paratmeters of transition, which 
        will be applied and then signals will be aggregated with sum.
        """
        transition_params = {}

        segment = signal.parameters['segment']
        sd = segment[1] - segment[0]
    
        max_stable_period = int(self.configs["max_stable_period"] * sd)
        low_fade_in  = int(self.configs['min_fade_in'] * sd)
        high_fade_in = int(self.configs['max_fade_in'] * sd)

        low_fade_out = int(self.configs['min_fade_out'] * sd)
        high_fade_out = int(self.configs['max_fade_out'] * sd)

        # Check properties of sampled transitions
        assert max_stable_period + high_fade_in + high_fade_out \
            <= sd, f"Chanhe next parameters in config such that\
            {max_stable_period} + {high_fade_in} + {high_fade_out}\
            <= {sd}"

        assert low_fade_in < high_fade_in,\
            "min_fade_in grater than max_fade_in"
        assert low_fade_out < high_fade_out,\
            "min_fade_out grater than max_fade_out"

    
        transition_params['segment_duration'] = sd
        # Fade in number of samples
        transition_params["fade_in"] = np.random.randint(low = low_fade_in, high = high_fade_in)

        # Fade out number of sampels
        transition_params["fade_out"] = np.random.randint(low = low_fade_out, high = high_fade_out) 

        # Generate stable point duration
        high = np.min([max_stable_period, sd - transition_params["fade_in"] - transition_params["fade_out"]])
        transition_params["stable_duration"] = np.random.randint(low = 0, high =  high + 1)
        
        # Generate silence period
        high = sd - transition_params["fade_in"]\
               - transition_params["stable_duration"]\
               - transition_params["fade_out"] 
        transition_params["silence_duration"] = np.random.randint(low = 0, high = high + 1)

        # Generate transition type
        transition_params['in_transition_type'] = np.random.choice(
            self.configs['in_transition_type'], size = 1) 
        transition_params['out_transition_type'] = np.random.choice(
            self.configs['out_transition_type'], size = 1)

        # Update parameters dict in Signal
        signal.parameters['transition_parameters'] = transition_params
        
        return signal
