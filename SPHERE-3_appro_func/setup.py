# setup.py
import os
from setuptools import setup

if os.environ.get('SPHERE_NO_CYTHON'):
    setup()
else:
    try:
        from setuptools import Extension
        from Cython.Build import cythonize
        import numpy as np

        extensions = [
            Extension(
                'sphere_appro.ldf_core',
                ['sphere_appro/ldf_core.pyx'],
                include_dirs=[np.get_include()],
            ),
        ]

        setup(
            ext_modules=cythonize(extensions, compiler_directives={
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
            }),
        )
    except Exception:
        setup()
