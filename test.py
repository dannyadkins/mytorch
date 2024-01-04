import torch as pytorch 
import mytorch 
import numpy as np
from abc import ABC
from time import time 
from typing import List, Tuple

# The first function should be the canonical implementation, which we use as the "correct" solution.
def run_testers(experiments: List[Tuple[callable, List[str]]]):
    # for each of pytorch and mytorch, creates an instance and calls setup 

    for experiment in experiments:
        TesterClass, fn_impls = experiment
        print("Running test for", TesterClass.__name__)
        tester = TesterClass()
        tester.setup()

        args = TesterClass.get_random_args()
        canonical_fn_path = TesterClass.get_canonical_fn_path()

        # outputs contain name, output, speed 
        outputs: List[dict] = []

        start_time = time()
        pytorch_output = tester.test(pytorch, canonical_fn_path, *args)
        end_time = time()
        outputs.append({
            'name': 'pytorch.' + canonical_fn_path,
            'correct': True,
            'speed': end_time - start_time,
        })

        for fn_impl in fn_impls:
            start_time = time()
            mytorch_output = tester.test(mytorch, fn_impl, *args)
            end_time = time()

            solution_matches = True 
        
            try: 
                assert pytorch_output.allclose(mytorch_output)
            except Exception:
                solution_matches = False
            
            outputs.append({
                'name': "mytorch." + fn_impl,
                'correct': solution_matches,
                'speed': end_time - start_time,
            })

        # print the outputs in a really pretty colorful table
        print("Output:")
        print("Name\t\t\tCorrect\t\tSpeed")
        for output in outputs:
            name = output['name']
            correct = output.get('correct', None)
            speed = output['speed']
            if correct is None:
                print(f"{name}\t\t{speed}")
            else:
                print(f"{name}\t\t{correct}\t\t{speed}")
        

        print(f"Test {tester.__class__.__name__} passed!")

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
    
    # gets some random relu-compatible arguments 
    @staticmethod
    def get_random_args():
        return [pytorch.rand(10, 10)]
    
    # gets the canonical implementation PATH of relu
    @staticmethod
    def get_canonical_fn_path():
        return 'nn.functional.relu'

    def test(self, torch_impl, fn_path, *args):
        # run relu based on the provided path
        fn = get_fn(torch_impl, fn_path)
        return fn(*args)

# Here is a list that maps every major building block in deep learning, progressively getting bigger.
# It points to the pytorch implementation/callpath, gives a set of example inputs, 
# and also points to a mytorch callpath if that exists. We should be able to specify many implementations/callpaths, since I might try multiple for mytorch.

if __name__ == "__main__":

    experiments = [
        (ReLUTester, ['nn.functional.relu_naive', 'nn.functional.relu_naive_cuda', 'nn.functional.relu_naive_triton']),
    ]

    run_testers(experiments)