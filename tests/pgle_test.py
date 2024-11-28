# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import filecmp
from functools import partial
import glob
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from absl.testing import absltest
import jax
from jax._src import api
from jax._src import compilation_cache as cc
from jax._src import config
from jax._src import monitoring
from jax._src import pjit
from jax._src import profiler
from jax._src import test_util as jtu
from jax.experimental import profiler as exp_profiler
from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np

try:
  import portpicker
except ImportError:
  portpicker = None

jax.config.parse_flags_with_absl()


def get_fdo_profiles(dump_dir):
  jit_f_fdo_profiles = [
      x
      for x in os.listdir(dump_dir)
      if 'jit_f' in x and x.endswith('.fdo_profile')
  ]
  return jit_f_fdo_profiles


@jtu.pytest_mark_if_available('multiaccelerator')
class PgleTest(jtu.JaxTestCase):

  def setUp(self):
    if 'SUBPROCESS' in os.environ:
      raise unittest.SkipTest('Subtest is required.')

    super().setUp()
    cc.set_cache_dir(None)
    cc.reset_cache()

  def tearDown(self):
    cc.set_cache_dir(None)
    super().tearDown()

  def testPGLEProfilerGetFDOProfile(self):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
        compiler_options={'xla_gpu_enable_latency_hiding_scheduler': 'True'},
    )
    def f(x, y):
      return x @ y

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x, y)
      compiled = f_lowered.compile()

    pgle_profiler = profiler.PGLEProfiler(1, 90)
    with config.enable_pgle(False):
      with profiler.PGLEProfiler.trace(pgle_profiler):
        compiled(x, y)

    fdo_profile = pgle_profiler.consume_fdo_profile()
    self.assertIsNotNone(fdo_profile)
    self.assertIn(b'custom', fdo_profile)

  def testPGLEProfilerGetFDOProfileLarge(self):
    mesh = jtu.create_mesh((2,), ('x',))
    its = 500

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
        compiler_options={
            'xla_gpu_enable_latency_hiding_scheduler': 'True',
            # TODO(patrios): Remove this flag once b/376647494 is fixed.
            'xla_gpu_graph_min_graph_size': '100000',
        },
    )
    def f(x):
      agg = x
      for _ in range(its):
        agg = agg @ x
      return agg

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)

    pgle_profiler = profiler.PGLEProfiler(1, 90)
    with config.enable_pgle(False):
      with profiler.PGLEProfiler.trace(pgle_profiler):
        f(x)
    fdo_profile = pgle_profiler.consume_fdo_profile()
    self.assertEqual(fdo_profile.count(b'custom'), its)

  def testAutoPgle(self):
    mesh = jtu.create_mesh((2,), ('x',))

    with tempfile.TemporaryDirectory() as dump_dir:

      @partial(
          jax.jit,
          in_shardings=NamedSharding(mesh, PartitionSpec('x')),
          out_shardings=NamedSharding(mesh, PartitionSpec('x')),
          compiler_options={
              'xla_gpu_enable_latency_hiding_scheduler': 'True',
              # TODO(patrios): Remove this flag once b/376647494 is fixed.
              'xla_gpu_graph_min_graph_size': '100000',
              'xla_dump_to': dump_dir,
              'xla_gpu_experimental_dump_fdo_profiles': 'True',
          },
      )
      def f(x):
        return x * 2

      shape = (16, 16)
      x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
      expected = x * 2

      with config.pgle_profiling_runs(2), config.enable_pgle(True):
        # Run 1: Module should be compiled without FDO. Two modules are expected
        # One is the funtion f, the other one is multi slice module
        with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertEqual(cache_miss_count[0], 2)

        # Run 2: Second PGLE run. Profile should be empty.
        with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertEqual(cache_miss_count[0], 2)
        fdo_profiles_before_pgle = get_fdo_profiles(dump_dir)
        # One for before and one for after optimization.
        self.assertLen(fdo_profiles_before_pgle, 2)
        # The FDO profile file should be empty.
        self.assertEqual(
            os.path.getsize(
                os.path.join(dump_dir, fdo_profiles_before_pgle[0])
            ),
            0,
        )

        # Run 3: The module should be recompiled with FDO profiles
        with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertEqual(cache_miss_count[0], 2)
        fdo_profiles_after_pgle = get_fdo_profiles(dump_dir)
        # One for before and one for after optimization.
        self.assertLen(fdo_profiles_after_pgle, 4)

        for fdo_profile in fdo_profiles_after_pgle:
          if fdo_profile not in fdo_profiles_before_pgle:
            self.assertGreater(
                os.path.getsize(os.path.join(dump_dir, fdo_profile)), 0
            )

        # Run 4: Fast-path should be used after PGLE is done
        with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertLess(cache_miss_count[0], 2)

  def testAutoPgleWithAot(self):
    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    expected = x * 2

    f_lowered = f.lower(x)
    serialized, in_tree, out_tree = serialize(f_lowered.compile())
    compiled = deserialize_and_load(serialized, in_tree, out_tree)

    with config.pgle_profiling_runs(1), config.enable_pgle(True):
      # Run 1
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(compiled(x), expected)
      self.assertEqual(cache_miss_count[0], 0)

      # Run 2
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(compiled(x), expected)
      self.assertEqual(cache_miss_count[0], 0)

  def testAutoPgleWithPersistentCache(self):
    its = 50
    mesh = jtu.create_mesh((2,), ('x',))

    with tempfile.TemporaryDirectory() as dump_dir:

      @partial(
          jax.jit,
          in_shardings=NamedSharding(mesh, PartitionSpec('x')),
          out_shardings=NamedSharding(mesh, PartitionSpec('x')),
          compiler_options={
              'xla_gpu_enable_latency_hiding_scheduler': 'True',
              # TODO(patrios): Remove this flag once b/376647494 is fixed.
              'xla_gpu_graph_min_graph_size': '100000',
              'xla_dump_to': dump_dir,
              'xla_gpu_experimental_dump_fdo_profiles': 'True',
          },
      )
      def f(x):
        agg = x
        for _ in range(its):
          agg = agg @ x
        return agg

      shape = (16, 16)
      x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)

      with (
          config.enable_compilation_cache(True),
          config.enable_pgle(True),
          config.raise_persistent_cache_errors(True),
          config.raise_persistent_cache_errors(True),
          config.persistent_cache_min_entry_size_bytes(0),
          config.persistent_cache_min_compile_time_secs(0),
          config.pgle_profiling_runs(2),
          tempfile.TemporaryDirectory() as cache_dir,
      ):
        cc.reset_cache()
        cc.set_cache_dir(cache_dir)
        # Run 1: Module should be compiled without FDO
        with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
          f(x)
        self.assertGreater(cache_miss_count[0], 0)

        # Non-pgle profiled version of module should be saved
        non_pgle_profiled_files = os.listdir(cache_dir)
        self.assertNotEmpty(non_pgle_profiled_files)

        # Run 2: Compilation should not be called
        with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
          f(x)
        self.assertGreater(cache_miss_count[0], 0)

        fdo_profiles_before_pgle = get_fdo_profiles(dump_dir)
        # Run 3: Module should be compiled with FDO and stored to persistent cache
        with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
          f(x)
        self.assertGreater(cache_miss_count[0], 0)

        # Check if FDO profile file of the biggest module is not empty
        fdo_profiles_after_pgle = [
            x
            for x in get_fdo_profiles(dump_dir)
            if x not in fdo_profiles_before_pgle
        ]
        self.assertNotEmpty(fdo_profiles_after_pgle)

        # Check if FDO profile file in dump directory is not empty
        for fdo_profile in fdo_profiles_after_pgle:
          self.assertGreater(
              os.path.getsize(os.path.join(dump_dir, fdo_profile)), 0
          )

        for pgle_profiler in pjit._pgle_profiler_dict.values():
          self.assertTrue(pgle_profiler.is_enabled())
          self.assertTrue(pgle_profiler.is_fdo_consumed())

        files_after_pgle_profile = os.listdir(cache_dir)
        self.assertGreater(
            len(files_after_pgle_profile), len(non_pgle_profiled_files)
        )

        # Removing non-pgle profiled module from cache to check that later pgle
        # profiled version will be used.
        for non_pgle_file in non_pgle_profiled_files:
          path = os.path.join(cache_dir, non_pgle_file)
          if os.path.isfile(path):
            os.remove(path)
          elif os.path.isdir(path):
            shutil.rmtree(path)

        api.clear_caches()
        pjit._pgle_profiler_dict.clear()

        # Run 4: Persistent compilation cache should be hit PGLE profiler should
        # be disabled
        cache_hit = 0

        def check_if_cache_hit(event):
          nonlocal cache_hit
          if event == '/jax/compilation_cache/cache_hits':
            cache_hit += 1

        monitoring.register_event_listener(check_if_cache_hit)
        f(x)
        monitoring._unregister_event_listener_by_callback(check_if_cache_hit)

        self.assertGreater(cache_hit, 0)

  def testPassingFDOProfile(self):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
        compiler_options={'xla_gpu_enable_latency_hiding_scheduler': 'True'},
    )
    def f(x, y):
      return x @ y

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x, y)
      compiled = f_lowered.compile()

    with tempfile.TemporaryDirectory() as cache_dir:
      jax.profiler.start_trace(cache_dir)
      compiled(x, y)
      jax.profiler.stop_trace()
      directories = glob.glob(os.path.join(cache_dir, 'plugins/profile/**/'))
      directories = [d for d in directories if os.path.isdir(d)]
      rundir = directories[-1]
      logging.info('rundir: %s', rundir)
      fdo_profile = exp_profiler.get_profiled_instructions_proto(rundir)

    if jtu.test_device_matches(['gpu']) and jtu.is_device_cuda():
      self.assertIn(b'custom', fdo_profile)

    logging.info('fdo_profile: %s', fdo_profile)
    # Test pass fdo_profile as compiler_options API works.
    f_lowered.compile(compiler_options={'fdo_profile': fdo_profile})

  def testMultiProcessFDOProfileSync(self):
    if not jtu.is_running_under_pytest():
      raise unittest.SkipTest('Multiprocess test is only supported in pytest.')

    if not portpicker:
      raise unittest.SkipTest('Tests requires portpicker.')

    port = portpicker.pick_unused_port()
    num_gpus_per_task = 1

    def run_sub_process(task_id: int, exit_stack, dump_dir, out_file, err_file):
      env = os.environ.copy()
      env['JAX_PORT'] = str(port)
      env['TASK'] = str(task_id)
      env['DUMP_DIR'] = dump_dir
      env['SUBPROCESS'] = 'True'
      env['CUDA_VISIBLE_DEVICES'] = ','.join(
          str((task_id * num_gpus_per_task) + i)
          for i in range(num_gpus_per_task)
      )

      args = [sys.executable, __file__]
      proc = subprocess.Popen(
          args,
          env=env,
          stdout=out_file,
          stderr=err_file,
      )
      exit_stack.enter_context(proc)
      return proc

    def wait_and_log(proc, out_file, err_file):
      proc.wait()

      out_file.seek(0)
      err_file.seek(0)
      res_out = out_file.read()
      res_err = err_file.read()
      print(res_out.decode('utf-8'))
      print(res_err.decode('utf-8'))

      self.assertEqual(proc.returncode, 0)

    with (
        contextlib.ExitStack() as exit_stack,
        tempfile.TemporaryDirectory() as first_dump,
        tempfile.TemporaryDirectory() as second_dump,
        tempfile.TemporaryFile() as first_out,
        tempfile.TemporaryFile() as first_err,
        tempfile.TemporaryFile() as second_out,
        tempfile.TemporaryFile() as second_err,
        run_sub_process(
            task_id=0,
            exit_stack=exit_stack,
            dump_dir=first_dump,
            out_file=first_out,
            err_file=first_err,
        ) as first_proc,
        run_sub_process(
            task_id=1,
            exit_stack=exit_stack,
            dump_dir=second_dump,
            out_file=second_out,
            err_file=second_err,
        ) as second_proc,
    ):
      print('----Wait for the second process----')
      wait_and_log(second_proc, second_out, second_err)
      print('----Wait for the first process----')
      wait_and_log(first_proc, first_out, first_err)

      # Check that FDO profiles exists and they are equals
      for first_profile_name, second_profile_name in zip(
          get_fdo_profiles(first_dump), get_fdo_profiles(second_dump)
      ):
        first_profile = os.path.join(first_dump, first_profile_name)
        second_profile = os.path.join(second_dump, second_profile_name)
        self.assertTrue(filecmp.cmp(first_profile, second_profile))


class PgleMultiProcessTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_running_under_pytest():
      raise unittest.SkipTest('Multiprocess test is only supported in pytest.')
    if 'SUBPROCESS' not in os.environ:
      raise unittest.SkipTest('Subtest is required.')
    super().setUp()

  def testMultiProcessFDOProfileSyncSubprocess(self):
    port = os.environ['JAX_PORT']
    num_tasks = 2
    task_id = int(os.environ['TASK'])
    dump_dir = os.environ['DUMP_DIR']
    coordinator_address = f'localhost:{port}'
    jax.distributed.initialize(coordinator_address, num_tasks, task_id)
    mesh = jtu.create_mesh((2,), ('x',))

    its = 500

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
        compiler_options={
            'xla_gpu_enable_latency_hiding_scheduler': 'True',
            # TODO(patrios): Remove this flag once b/376647494 is fixed.
            'xla_gpu_graph_min_graph_size': '100000',
            'xla_dump_to': dump_dir,
            'xla_gpu_experimental_dump_fdo_profiles': 'True',
        },
    )
    def f(x):
      agg = x
      for _ in range(its):
        agg = agg @ x
      return agg

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    num_runs = 2
    with (
        config.pgle_profiling_runs(num_runs),
        config.enable_pgle(True),
        config.enable_compilation_cache(True),
        config.raise_persistent_cache_errors(True),
        config.raise_persistent_cache_errors(True),
        config.persistent_cache_min_entry_size_bytes(0),
        config.persistent_cache_min_compile_time_secs(0),
    ):
      for _ in range(num_runs + 1):
        f(x)
      fdo_profiles = get_fdo_profiles(dump_dir)
      # One for before and one for after optimization.
      self.assertLen(fdo_profiles, 4)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
