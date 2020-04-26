from setuptools import setup
from distutils.command.build import build
from setuptools.command.install import install

from setuptools.command.develop import develop

import os
import subprocess
BASEPATH = os.path.dirname(os.path.abspath(__file__))

class custom_develop(develop):
    def run(self):
        original_cwd = os.getcwd()

        # build custom tf ops
        folders = [
            os.path.join(BASEPATH, 'gtrans/graphnet'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])

        os.chdir(original_cwd)

        super().run()


setup(name='gtrans',
      py_modules=['gtrans'],
      cmdclass={
          'develop': custom_develop,
        }
)
