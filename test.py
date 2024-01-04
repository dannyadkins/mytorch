import torch as pytorch 
import mytorch 
import numpy as np
from abc import ABC

from typing import List, Tuple


# The first function should be the canonical implementation, which we use as the "correct" solution.
def run_testers(experiments: List[Tuple[callable, List[str]]]):
    # for each of pytorch and mytorch, creates an instance and calls setup 
    tester_classes, fn_impls = experiments

    for TesterClass in tester_classes:
        print("Running test for", TesterClass.__name__)
        tester = TesterClass()
        tester.setup()
        args = tester.get_random_args()
        pytorch_output = tester.test(pytorch, **args)
        mytorch_output = tester.test(mytorch, **args)
        assert pytorch_output.allclose(mytorch_output)
        print(f"Test {tester.name} passed!")

# abstract TestBase class, using ABC: 
class TestBase(ABC):
    def __init__(self, args: tuple, kwargs: dict):
        self.args = args 
        self.kwargs = kwargs 

    # static get_random_args method
    @staticmethod
    def get_random_args():
        pass

    @staticmethod 
    def get_canonical_fn_path():
        pass

    def setup(self):
        pass

    def test(self, torch_impl, **args):
        pass 

# setup function, which does any prep work 
class ReLUTester: 
    def setup(self):
        pass
    
    # gets some random relu-compatible arguments 
    @staticmethod
    def get_random_args():
        return {'input': pytorch.rand(10, 10)}
    
    # gets the canonical implementation PATH of relu
    @staticmethod
    def get_canonical_fn_path():
        return 'nn.functional.relu'

    def test(self, torch_impl, fn_path, **args):
        # run relu based on the provided path
        fn = getattr(torch_impl, fn_path)
        return fn(**args)

# Here is a list that maps every major building block in deep learning, progressively getting bigger.
# It points to the pytorch implementation/callpath, gives a set of example inputs, 
# and also points to a mytorch callpath if that exists. We should be able to specify many implementations/callpaths, since I might try multiple for mytorch.

if __name__ == "__main__":

    experiments = [
        ([ReLUTester], ['nn.functional.relu']),
    ]

    run_testers()