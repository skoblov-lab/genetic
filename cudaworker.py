"""
This is a temporary module intended for testing purposes; we will move it later
to sklab-util
"""

import base64
import importlib.util
import inspect
import os
import shutil
import subprocess as sp
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Queue
from types import ModuleType
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union

import click
import dill

from genetic.base import Executor, Ord

Individual = TypeVar('Individual')


# TODO: use https://pypi.org/project/enlighten/ for progress bars and logging
# TODO: consider importing __all__

class CudaSubprocessExecutor(Generic[Individual], Executor):

    def __init__(self,
                 gpus: Iterable[Union[int, str]],
                 default: Ord,
                 init: Optional[ModuleType] = None,
                 init_name: Optional[str] = None,
                 verbose=False):
        """
        :param gpus: an iterable of GPU device IDs
        :param init: optional path to a module to import within the worker space
        :param default: default value to return in case of worker failure
        """
        self._devices = list(gpus)
        if not self._devices:
            raise ValueError('no GPU ids')
        self._gpus: Queue = Queue()
        for device in self._devices:
            self._gpus.put(device)
        if init and not isinstance(init, ModuleType):
            raise ValueError
        if init_name and not init:
            raise ValueError
        self._init_path = inspect.getsourcefile(init) if init else None
        self._init_name = (
            init_name if init_name else
            init.__name__ if init else
            None
        )
        self._default = default
        # set path to this module
        self._path = inspect.getsourcefile(inspect.currentframe())
        self._verbose = verbose

    def map(self,
            func: Callable[[Individual], Ord],
            individuals: Iterable[Individual]) -> List[Ord]:
        with ThreadPoolExecutor(len(self._devices)) as workers:
            return list(workers.map(partial(self._submit_task, func), individuals))

    def __str__(self):
        return f'{type(self).__name__} running on GPUs {str(self._devices)}'

    def _submit_task(self, f: Callable[..., Ord], *args, **kwargs) -> Ord:
        # make a package to transport
        package = self._init_name, self._init_path, f, args, kwargs
        encoded = base64.encodebytes(dill.dumps(package))
        # get a gpu device and call the function
        gpu = self._gpus.get()
        command = [
            shutil.which('python3'), self._path, '--device', str(gpu)
        ]
        process = sp.run(command, input=encoded, stdout=sp.PIPE, stderr=sp.PIPE)
        # release the gpu and return
        self._gpus.put(gpu)
        # TODO use proper logging
        if process.returncode:
            if self._verbose:
                print(process.stderr.decode(), file=sys.stderr)
            return self._default
        return dill.loads(base64.decodebytes(process.stdout))


@click.command('cudaworker')
@click.option('--device', type=int)
def cudaworker(device: int):
    # TODO validate device ID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    # decode the package
    decoded_package = base64.decodebytes(sys.stdin.buffer.read())
    init_name, init_path, func, args, kwargs = dill.loads(decoded_package)
    # import stuff from init
    if init_name and init_path:
        init_spec = importlib.util.spec_from_file_location(init_name, init_path)
        init_module = importlib.util.module_from_spec(init_spec)
        init_spec.loader.exec_module(init_module)
        globals()[init_name] = init_module
    # run the function, encode the output and write to stdout
    output = func(*args, **kwargs)
    encoded_output = base64.encodebytes(dill.dumps(output))
    sys.stdout.buffer.write(encoded_output)


if __name__ == '__main__':
    cudaworker()
