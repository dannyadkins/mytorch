import torch as pytorch 
import mytorch 
import numpy as np
from abc import ABC
from time import time 
from typing import List, Tuple
from prettytable import PrettyTable

# The first function should be the canonical implementation, which we use as the "correct" solution.
def run_testers(experiments: List[Tuple[callable, List[str]]]):
    # for each of pytorch and mytorch, creates an instance and calls setup 

    for experiment in experiments:
        TesterClass, fn_impls = experiment
        tester = TesterClass()
        tester.setup()

        args = TesterClass.get_random_args()
        canonical_fn_path = TesterClass.get_canonical_fn_path()

        # outputs contain name, output, speed 
        outputs: List[dict] = []

        cloned_args = [arg.clone() if isinstance(arg, pytorch.Tensor) else arg for arg in args]

        start_time = time()
        pytorch_output = tester.test(pytorch, canonical_fn_path, *cloned_args)
        end_time = time()
        baseline_speed = end_time - start_time
        outputs.append({
            'name': 'pytorch.' + canonical_fn_path,
            'correct': True,
            'speed': baseline_speed,
            'speed_ratio': 1.0,
            'error': None,
            'result': pytorch_output
        })


        for fn_impl in fn_impls:
            cloned_args = [arg.clone() if isinstance(arg, pytorch.Tensor) else arg for arg in args]

            error: Exception = None
            mytorch_output = None

            start_time = time()

            try: 
                mytorch_output = tester.test(mytorch, fn_impl, *cloned_args)
            except Exception as e:
                error = e

            end_time = time()

            solution_matches = True 
        
            try: 
                assert pytorch_output.allclose(mytorch_output)
            except Exception as e:
                solution_matches = False
            
            outputs.append({
                'name': "mytorch." + fn_impl,
                'correct': solution_matches,
                'speed': end_time - start_time,
                'speed_ratio': (end_time - start_time) / baseline_speed,
                'error': error,
                'result': mytorch_output
            })
        print_table(outputs)

def print_table(outputs: List[dict]):
    table = PrettyTable(['Name', 'Correct', 'Speed', 'Speed Ratio %', 'Error', 'Result'])
    for output in outputs:
        speed_ratio_percent_str = "{:.2f}%".format(output['speed_ratio'] * 100)
        # table.add_row([output['name'], output['correct'], output['speed'], speed_ratio_percent_str])
        # if speed % is an improvement, make it green, otherwise red 
        if output['speed_ratio'] < 1.0 and output['correct']:
            speed_ratio_percent_str = "\033[92m" + speed_ratio_percent_str + "\033[0m"
        elif output['speed_ratio'] > 1.0: 
            speed_ratio_percent_str = "\033[91m" + speed_ratio_percent_str + "\033[0m"

        # now for correctness, if it's correct, make it green, otherwise red
        if output['correct']:
            output['correct'] = "\033[92m" + "True" + "\033[0m"
        else: 
            output['correct'] = "\033[91m" + "False" + "\033[0m"
        
        if output['error'] is not None:
            output['error'] = "\033[91m" + str(output['error']) + "\033[0m"

        # make the result pretty, e.g. just print the first few characters/first number
        if isinstance(output['result'], pytorch.Tensor):
            # the first number in the whole tensor, regardless of dimensionality
            output['result'] = str(output['result'].flatten()[0]) + ' ' + str(output['result'].shape) + ' ' + str(output['result'].dtype)

        table.add_row([output['name'], output['correct'], output['speed'], speed_ratio_percent_str, output['error'], output['result']])
    print(table)

# abstract TestBase class, using ABC: 
class TestBase(ABC):
    def __init__(self, args: tuple, kwargs: dict):
        self.args = args 
        self.kwargs = kwargs 
        self.name = self.__class__.__name__

    # static get_random_args method
    @staticmethod
    def get_random_args():
        pass

    @staticmethod 
    def get_canonical_fn_path():
        pass

    def setup(self):
        pass

    def test(self, torch_impl, *args):
        pass 

def get_fn(parent_module, fn_path):
    fn_parts = fn_path.split('.')
    fn = parent_module
    for part in fn_parts:
        fn = getattr(fn, part)
    return fn

# setup function, which does any prep work 
class ReLUTester: 
    def setup(self):
        pass
    
    # gets some random relu-compatible arguments where some are negative
    @staticmethod
    def get_random_args():
        return [pytorch.randn(100, 100),]
    
    # gets the canonical implementation PATH of relu
    @staticmethod
    def get_canonical_fn_path():
        return 'nn.functional.relu'

    def test(self, torch_impl, fn_path, *args):
        # run relu based on the provided path
        fn = get_fn(torch_impl, fn_path)
        return fn(*args)

class Conv2DTester: 
    def setup(self):
        pass
    
    # gets some random conv2d-compatible arguments, make the kernel entirely positive
    @staticmethod
    def get_random_args():
        t = pytorch.randn(1, 1, 28, 28)
        # t[t < 0] = 0

        kernel = pytorch.randn(1, 1, 3, 3)
        # kernel[kernel < 0] = 0

        return [t, kernel]
    
    @staticmethod
    def get_canonical_fn_path():
        return 'nn.functional.conv2d'

    def test(self, torch_impl, fn_path, *args):
        fn = get_fn(torch_impl, fn_path)
        return fn(*args)
    
class MaxPool2DTester: 
    def setup(self):
        pass
    
    # gets some random maxpool2d compatible arguments
    # input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
    @staticmethod
    def get_random_args():
        return [pytorch.randn(1, 1, 28, 28), (2, 2), (2, 2), (0, 0)]
    
    @staticmethod
    def get_canonical_fn_path():
        return 'nn.functional.max_pool2d'

    def test(self, torch_impl, fn_path, *args):
        fn = get_fn(torch_impl, fn_path)
        return fn(*args)
    
class BatchNorm2DTester: 
    def setup(self):
        pass
    
    # gets some random batchnorm2d compatible arguments
    # inputs: (input_tensor, running_mean, running_var, weight (optional), bias (optional), training (optional), momentum (optional), eps (optional))
    @staticmethod
    def get_random_args():
        return [pytorch.randn(1, 1, 28, 28), pytorch.Tensor([0]), pytorch.Tensor([1]), pytorch.Tensor([1]), pytorch.Tensor([0]), False, 0.1, 1e-5]
    
    @staticmethod
    def get_canonical_fn_path():
        return 'nn.functional.batch_norm'

    def test(self, torch_impl, fn_path, *args):
        fn = get_fn(torch_impl, fn_path)
        return fn(*args)

if __name__ == "__main__":

    experiments = [
        (ReLUTester, ['nn.functional.relu_naive', 'nn.functional.relu_naive_inplace', 'relu_cython_naive', 'nn.functional.relu_vectorized_numpy', 'nn.functional.relu_naive_cuda', 'nn.functional.relu_naive_triton', 'nn.functional.bad_relu']),
        (Conv2DTester, ['nn.functional.conv2d_naive']),
        (MaxPool2DTester, ['nn.functional.max_pool2d_naive']),
        (BatchNorm2DTester, ['nn.functional.batch_norm_naive']),
    ]

    run_testers(experiments)