"""
This is a temporary module intended for testing purposes; we will move it later
to sklab-util
"""

import importlib.util
import inspect
import os
import shutil
import subprocess as sp
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from queue import Queue
from types import ModuleType
from typing import Callable, Generic, Iterable, List, Optional, TypeVar

import click
import dill
from fn import F

from genetic.base import Executor, Ord

Individual = TypeVar('Individual')


# TODO: use https://pypi.org/project/enlighten/ for progress bars and logging
# TODO: consider importing __all__

class CudaSubprocessExecutor(Generic[Individual], Executor):

    def __init__(self,
                 devices: List[List[int]],
                 default: Ord,
                 init: Optional[ModuleType] = None,
                 init_name: Optional[str] = None,
                 verbose=False):
        """
        :param devices: an iterable of device ID sets. Since each
        active worker must have exclusive access to a group of GPUs, GPU ID sets
        cannot intersect. Empty sets are allowed and can be repeated (workers
        receiving empty ID sets will simply run on CPU alone). GPU ID sets can
        differ in size. Examples: [[1, 2], [3, 4]], [[1, 2], [3], [4], []]
        :param init: an optional module to import within the worker space
        :param init_name: optional name for the module, otherwise its complete
        default name will be used (this option is handy if you want to mimic
        the `... import as ...` mechanism or if you want to import a module
        that contains forbidden symbols).
        :param default: default value to return in case of worker failure
        """
        if not devices:
            raise ValueError('no device groups ids')
        unique_devices = (F(map, set) >> (reduce, set.union))(devices)
        n_devices = (F(map, len) >> sum)(devices)
        if n_devices > len(unique_devices):
            raise ValueError(
                'At least one GPU ID is repeated across multiple device groups'
            )
        # CPU device groups become empty strings
        self._devices = (F(map, ','.join) >> list)(devices)
        # load a device queue
        self._device_queue: Queue = Queue()
        for device in self._devices:
            self._device_queue.put(device)
        if init and not isinstance(init, ModuleType):
            raise ValueError('`init` is not a valid module')
        if init_name and not init:
            raise ValueError('`init_name` passed without any `init` module')
        self._init_path = inspect.getsourcefile(init) if init else None
        self._init_name = (
            init_name if init_name else
            init.__name__ if init else
            None
        )
        # TODO should we catch and send an error instead?
        if not isinstance(default, Ord):
            raise ValueError('`default` is not an instance of Ord')
        self._default = default
        # set path to this module
        self._path = inspect.getsourcefile(inspect.currentframe())
        self._verbose = verbose

    def map(self,
            func: Callable[[Individual], Ord],
            individuals: Iterable[Individual]) -> List[Ord]:

        with ThreadPoolExecutor(len(self._devices)) as workers:
            return list(
                workers.map(F(self._submit_task, func), individuals)
            )

    def __str__(self):
        return (
            f'{type(self).__name__} running on device groups '
            f'{str(self._devices)}'
        )

    def _submit_task(self, f: Callable[..., Ord], *args, **kwargs) -> Ord:
        # make a package to transport
        package = self._init_name, self._init_path, f, args, kwargs
        encoded = dill.dumps(package)
        # get an encoded device group and call `f`
        devices = self._device_queue.get()
        with tempfile.NamedTemporaryFile() as buffer:
            buffer.write(encoded)
            buffer.flush()
            command = [
                shutil.which('python3'), self._path,
                '--devices', devices,
                '--exchange', buffer.name
            ]
            process = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
            # release the device group and return
            self._device_queue.put(devices)
            # TODO use proper logging
            if process.returncode:
                if self._verbose:
                    print(process.stderr.decode(), file=sys.stderr)
                return self._default
            buffer.seek(0)
            output_encoded = buffer.read()
            return dill.loads(output_encoded)


@click.command('cudaworker')
@click.option('--devices', type=str)
@click.option('--exchange', type=click.Path(resolve_path=True, exists=True))
def cudaworker(devices: str, exchange: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    with open(exchange, 'rb') as buffer:
        package = buffer.read()
    # decode the package
    init_name, init_path, func, args, kwargs = dill.loads(package)
    # import stuff from init
    if init_name and init_path:
        init_spec = importlib.util.spec_from_file_location(init_name, init_path)
        init_module = importlib.util.module_from_spec(init_spec)
        init_spec.loader.exec_module(init_module)
        globals()[init_name] = init_module
    # run the function, encode the output and write to stdout
    output = func(*args, **kwargs)
    output_encoded = dill.dumps(output)
    with open(exchange, 'wb') as buffer:
        buffer.write(output_encoded)


# @contextmanager
# def temp_fifo(tmp: str = tempfile.gettempdir(), suffix: str='fifo'):
#     """
#     Create a temporary named pipe (FIFO)
#     :param tmp: temporary directory; defaults to system TMPDIR
#     :param suffix: FIFO name suffix; defaults to 'fifo'
#     """
#     tmpdir = tempfile.mkdtemp(dir=tmp)
#     filename = os.path.join(tmpdir, suffix)
#     os.mkfifo(filename)  # create the FIFO
#     yield filename
#     # cleanup
#     os.unlink(filename)
#     os.rmdir(tmpdir)


if __name__ == '__main__':
    cudaworker()
