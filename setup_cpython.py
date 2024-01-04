from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import numpy
import torch 

# Find all .pyx files in the 'mytorch' directory
pyx_files = []
for root, dirs, files in os.walk("mytorch"):
    for file in files:
        if file.endswith(".pyx"):
            module_path = os.path.join(root, file)
            module_name = module_path.replace('/', '.').replace('\\', '.').replace('.pyx', '')
            pyx_files.append(Extension(module_name, [module_path]))

# Configure compiler directives
compiler_directives = {'language_level': 3}

setup(
    name='mytorch_cython_modules',
    ext_modules=cythonize(pyx_files, compiler_directives=compiler_directives, annotate=True),
    include_dirs=[numpy.get_include(), torch.get_include(), torch.utils.cpp_extension.include_paths()[0]],
    script_args=['build_ext', '--inplace'],
    zip_safe=False,
)
