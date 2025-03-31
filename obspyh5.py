# Copyright 2013-2016 Tom Eulenfeld, MIT license
"""
obspyh5
=======
HDF5 write/read support for obspy
---------------------------------

Welcome!

Writes and reads ObsPy streams to/from HDF5 files.
Stats attributes are preserved if they are numbers, strings,
UTCDateTime objects or numpy arrays.
Its best used as a plugin to obspy.

For some examples have a look at the README.rst_.

.. _README.rst: https://github.com/trichter/obspyh5

This version attempts some speed improvements,
e.g. non-recursive traversal, parallel processing,
     mem optimization, caching, and some HDF5 stuff

"""

import json
from pathlib import Path
from warnings import warn
from typing import Dict, List, Union, Iterator, Optional, Any, Tuple
import functools

import numpy as np
from obspy.core import Trace, Stream, UTCDateTime as UTC
from obspy.core.util import AttribDict
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    
# For parallel processing
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

__version__ = '0.7.0'

_IGNORE = ('endtime', 'sampling_rate', 'npts', '_format')

_INDEXES = {
    'standard': (
        'waveforms/{trc_num:03d}_{id}_'
        '{starttime.datetime:%Y-%m-%dT%H:%M:%S}_{duration:.1f}s'),
    'flat': (
        'waveforms/{id}_'
        '{starttime.datetime:%Y-%m-%dT%H:%M:%S}_{duration:.1f}s'),
    'nested': (
        'waveforms/{network}.{station}/{location}.{channel}/'
        '{starttime.datetime:%Y-%m-%dT%H:%M:%S}_{duration:.1f}s'),
    'xcorr': (
        'waveforms/{network1}.{station1}-{network2}.{station2}/'
        '{location1}.{channel1}-{location2}.{channel2}/'
        '{starttime.datetime:%Y-%m-%dT%H:%M:%S}_{duration:.1f}s')}

_INDEX = _INDEXES['standard']

_NOT_SERIALIZABLE = '<not serializable>'

# Cache for datasets to avoid redundant lookups
_DATASET_CACHE = {}
# Cache limit (number of datasets)
_CACHE_LIMIT = 1000

def _is_utc(utc: str) -> bool:
    """Check if string is a UTC timestamp."""
    utc = str(utc)
    return len(utc) == 27 and utc.endswith('Z')


class _FlexibleEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, AttribDict):
            return dict(obj)
        elif isinstance(obj, np.ndarray) and len(np.shape(obj)) == 1:
            return list(obj)
        else:
            warn(f'{obj!r:.80} is not serializable')
        return _NOT_SERIALIZABLE


def set_index(index: str = 'standard') -> None:
    """
    Set index for newly created files.

    Some indexes are hold by the module variable _INDEXES ('standard' and
    'xcorr'). The index can also be set to a custom value, e.g.

    >>> set_index('waveforms/{network}.{station}/{otherstrangeheader}')

    This string gets evaluated by a call to its format method,
    with the stats of each trace as kwargs, e.g.

    >>> 'waveforms/{network}.{station}/{otherstrangeheader}'.format(**stats)

    This means, that headers used in the index must exist for a trace to write.
    The index is stored inside the HDF5 file.

    :param index: 'standard' (default), 'xcorr' or other string.
    """
    global _INDEX
    if index in _INDEXES:
        _INDEX = _INDEXES[index]
    else:
        _INDEX = index


def is_obspyh5(fname: str) -> bool:
    """
    Check if file is a HDF5 file and if it was written by obspyh5.
    """
    if not HAS_H5PY:
        return False
        
    try:
        if not h5py.is_hdf5(fname):
            return False
        with h5py.File(fname, 'r') as f:
            return f.attrs['file_format'].lower() == 'obspyh5'
    except Exception:
        return False


def clear_cache() -> None:
    """Clear the dataset cache to free memory."""
    global _DATASET_CACHE
    _DATASET_CACHE = {}


def get_dataset_paths(fname: str, group: str = '/', readonly: Optional[Dict] = None) -> List[str]:
    """
    Get all dataset paths in an HDF5 file matching criteria.
    
    This is a faster alternative to recursive traversal.
    
    :param fname: Name of file to read
    :param group: Group or subgroup to read, defaults to '/'
    :param readonly: Read only traces restricted by given dict
    :return: List of dataset paths
    """
    paths = []
    
    with h5py.File(fname, 'r', swmr=True) as f:
        # If readonly is specified, construct a partial path
        if readonly is not None:
            try:
                index = f.attrs.get('index', _INDEX)
            except KeyError:
                index = _INDEX
                
            # Split the index into path components
            index_parts = index.split('/')
            partial_path = []
            
            # Try to format each part with the readonly dict
            for part in index_parts:
                try:
                    partial_path.append(part.format(**readonly))
                except KeyError:
                    # Stop when we can't format anymore
                    break
            
            # Create the partial path to search in
            if partial_path:
                search_group = '/'.join([group.rstrip('/')] + partial_path)
                if search_group in f:
                    group = search_group

        # Use visititems for efficient traversal without recursion
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths.append(name)
                
        # Start from the specified group
        if group in f:
            f[group].visititems(collect_datasets)
            
    return paths


def dataset2trace_fast(dataset, headonly: bool = False) -> Trace:
    """
    Load trace from dataset with optimized metadata handling.
    
    :param dataset: HDF5 dataset
    :param headonly: Read only the headers of the trace
    :return: ObsPy Trace
    """
    # Get all attributes at once to minimize HDF5 calls
    attrs = dict(dataset.attrs)
    stats = AttribDict()
    
    # Process attributes
    for key, val in attrs.items():
        # Skip _json for now - we'll process it separately
        if key == '_json':
            continue
            
        # Decode bytes to utf-8 string for py3
        if isinstance(val, bytes):
            val = val.decode('utf-8')
            
        # Convert UTC timestamps
        if _is_utc(val):
            val = UTC(val)
        elif key == 'processing' and isinstance(val, (bytes, str)):
            # Handle legacy processing info (< 0.5.0)
            val = json.loads(val)
            
        stats[key] = val
    
    # Process JSON metadata if present
    json_data = attrs.get('_json')
    if json_data is not None:
        if isinstance(json_data, bytes):
            json_data = json_data.decode('utf-8')
            
        # Parse JSON data
        try:
            for k, v in json.loads(json_data).items():
                stats[k] = v
        except json.JSONDecodeError:
            warn(f"Could not decode JSON metadata in {dataset.name}")
    
    # Create trace with or without data
    if headonly:
        stats['npts'] = len(dataset)
        trace = Trace(header=stats)
    else:
        # Direct array access is faster than dataset[...]
        data = np.array(dataset)
        trace = Trace(data=data, header=stats)
        
    return trace


def process_dataset(file_obj, dataset_path: str, headonly: bool = False) -> Trace:
    """
    Process a single dataset from an HDF5 file.
    
    :param file_obj: Open HDF5 file object
    :param dataset_path: Path to dataset in the file
    :param headonly: Read only the headers of the trace
    :return: ObsPy Trace
    """
    dataset = file_obj[dataset_path]
    return dataset2trace_fast(dataset, headonly=headonly)


def readh5(fname: str, group: str = '/', headonly: bool = False, 
           readonly: Optional[Dict] = None, parallel: bool = True,
           max_workers: Optional[int] = None, batch_size: int = 100,
           use_cache: bool = True, **kwargs) -> Stream:
    """
    Read HDF5 file and return Stream object with optimized performance.

    :param fname: Name of file to read
    :param group: Group or subgroup to read, defaults to '/'
    :param headonly: Read only the headers of the traces
    :param readonly: Read only traces restricted by given dict
    :param parallel: Use parallel processing for loading traces
    :param max_workers: Maximum number of worker threads (default: CPU count)
    :param batch_size: Number of traces to process in each batch
    :param use_cache: Use cache for repeated access to the same file
    :param kwargs: Other kwargs are ignored
    :return: ObsPy Stream
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for readh5_fast")
    
    # Get cache key
    cache_key = f"{fname}:{group}:{headonly}:{readonly}"
    
    # Check cache
    if use_cache and cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]
    
    # Get all dataset paths first
    dataset_paths = get_dataset_paths(fname, group, readonly)
    
    # If no datasets found, return empty stream
    if not dataset_paths:
        return Stream()
    
    traces = []
    
    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Choose between parallel and sequential processing
    if parallel and len(dataset_paths) > 10:  # Only parallelize for enough traces
        with h5py.File(fname, 'r', swmr=True) as f:
            # Process in batches to avoid excessive thread creation
            for i in range(0, len(dataset_paths), batch_size):
                batch = dataset_paths[i:i+batch_size]
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Create partial function with fixed file object
                    process_func = functools.partial(process_dataset, f, headonly=headonly)
                    # Map over dataset paths
                    batch_traces = list(executor.map(process_func, batch))
                    traces.extend(batch_traces)
    else:
        # Sequential processing
        with h5py.File(fname, 'r') as f:
            for path in dataset_paths:
                trace = process_dataset(f, path, headonly=headonly)
                traces.append(trace)
    
    # Create stream from traces
    stream = Stream(traces=traces)
    
    # Cache result if cache is enabled and not too large
    if use_cache and len(_DATASET_CACHE) < _CACHE_LIMIT:
        _DATASET_CACHE[cache_key] = stream
    
    return stream


def writeh5(stream: Stream, fname: str, mode: str = 'w', 
            override: str = 'warn', ignore: tuple = (), group: str = '/',
            libver: str = 'earliest', compression: Optional[str] = 'gzip',
            compression_opts: int = 4, chunks: bool = True,
            parallel: bool = True, max_workers: Optional[int] = None,
            **kwargs) -> None:
    """
    Write stream to HDF5 file with optimized performance.

    :param stream: Stream to write
    :param fname: Filename
    :param mode: 'w' (write, default), 'a' (append) or other
    :param override: 'warn', 'raise', 'ignore', or 'dont'
    :param ignore: Iterable of headers to ignore
    :param group: Group to write to, defaults to '/'
    :param libver: HDF5 version bounding
    :param compression: Compression filter (e.g. 'gzip', 'lzf')
    :param compression_opts: Compression options (for gzip: 0-9)
    :param chunks: Enable chunking for datasets
    :param parallel: Use parallel processing for writing
    :param max_workers: Maximum number of worker threads
    :param kwargs: Additional kwargs passed to create_dataset
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for writeh5_fast")
    
    # Ensure filename has .h5 extension
    if not Path(fname).suffix:
        fname = f"{fname}.h5"
    
    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Open file
    with h5py.File(fname, mode, libver=libver) as f:
        # Set file attributes
        f.attrs['file_format'] = 'obspyh5'
        f.attrs['version'] = __version__
        
        if 'index' not in f.attrs:
            f.attrs['index'] = _INDEX
            
        if 'offset_trc_num' not in f.attrs:
            f.attrs['offset_trc_num'] = 0
            
        trc_num = f.attrs['offset_trc_num']
        group_obj = f.require_group(group)
        
        # Format all trace paths first to reduce string formatting overhead
        trace_data = []
        for tr in stream:
            duration = tr.stats.endtime - tr.stats.starttime
            index = _INDEX.format(trc_num=trc_num, id=tr.id, duration=duration, **tr.stats)
            trace_data.append((tr, index))
            trc_num += 1
        
        # Update trc_num in file attributes
        f.attrs['offset_trc_num'] = trc_num
        
        # Choose between parallel and sequential processing
        if parallel and len(stream) > 20:  # Only parallelize for enough traces
            def write_trace(data):
                tr, index = data
                return                 trace2group(tr, group_obj, index, override, ignore, 
                                compression=compression, 
                                compression_opts=compression_opts,
                                chunks=chunks, **kwargs)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Process in batches
                for i in range(0, len(trace_data), 100):
                    batch = trace_data[i:i+100]
                    list(executor.map(write_trace, batch))
        else:
            # Sequential processing
            for tr, index in trace_data:
                trace2group(tr, group_obj, index, override, ignore,
                             compression=compression,
                             compression_opts=compression_opts,
                             chunks=chunks, **kwargs)


def trace2group(trace: Trace, group, index: str, override: str = 'warn', 
                    ignore: tuple = (), **kwargs) -> None:
    """
    Write trace into group with optimized performance.
    
    :param trace: Trace to write
    :param group: HDF5 group
    :param index: Preformatted index string
    :param override: How to handle existing datasets
    :param ignore: Headers to ignore
    :param kwargs: Additional kwargs passed to create_dataset
    """
    if override not in ('warn', 'raise', 'ignore', 'dont'):
        msg = "Override has to be one of ('warn', 'raise', 'ignore', 'dont')."
        raise ValueError(msg)
    
    if index in group:
        msg = f"Index '{index}' already exists."
        if override == 'warn':
            warn(msg + ' Will override trace.')
        elif override == 'raise':
            raise KeyError(msg)
        elif override == 'dont':
            return
        del group[index]
    
    # Set default dtype if not specified
    kwargs.setdefault('dtype', trace.data.dtype)
    
    # Create dataset and write data
    dataset = group.create_dataset(index, trace.data.shape, **kwargs)
    dataset[:] = trace.data
    
    # Add ignored headers
    ignore = tuple(ignore) + _IGNORE
    if '_format' in trace.stats and '_format' in _IGNORE:
        format_name = trace.stats._format.lower()
        ignore = ignore + (format_name,)
    
    # Separate simple attributes and complex attributes
    simple_attrs = {}
    complex_attrs = {}
    
    for key, val in trace.stats.items():
        if key not in ignore:
            if _is_utc(val):
                val = str(val)
                
            if isinstance(val, (tuple, list, AttribDict, dict)) or np.ndarray:
                # Complex attributes for JSON
                complex_attrs[key] = val
            else:
                try:
                    # Test if attribute can be stored directly
                    simple_attrs[key] = val
                except (KeyError, TypeError):
                    # Fall back to JSON for problematic types
                    complex_attrs[key] = val
    
    # Write simple attributes directly
    for key, val in simple_attrs.items():
        dataset.attrs[key] = val
        
    # Write complex attributes as JSON
    if complex_attrs:
        json_str = json.dumps(complex_attrs, cls=_FlexibleEncoder)
        dataset.attrs['_json'] = json_str


# Define iterh5 as generator from readh5
def iterh5(fname, group='/', readonly=None, headonly=False, mode='r', **kwargs):
    """
    Iterate over traces in HDF5 file. See readh5 for doc of kwargs.
    """
    for tr in readh5(fname, group=group, readonly=readonly, headonly=headonly, **kwargs):
        yield tr


# Functions for memory-efficient batch processing
def batch_process_h5(fname: str, processor_func, batch_size: int = 100, 
                   headonly: bool = False, **kwargs) -> Any:
    """
    Process an HDF5 file in batches to minimize memory usage.
    
    :param fname: Name of file to read
    :param processor_func: Function to process each batch of traces
    :param batch_size: Number of traces to process in each batch
    :param headonly: Read only the headers of the traces
    :param kwargs: Additional kwargs passed to readh5_fast
    :return: Result from processor_func
    """
    # Get all dataset paths
    dataset_paths = get_dataset_paths(fname, **kwargs)
    results = []
    
    # Process in batches
    with h5py.File(fname, 'r', swmr=True) as f:
        for i in range(0, len(dataset_paths), batch_size):
            batch_paths = dataset_paths[i:i+batch_size]
            batch_traces = []
            
            for path in batch_paths:
                trace = process_dataset(f, path, headonly=headonly)
                batch_traces.append(trace)
                
            # Process this batch
            batch_stream = Stream(traces=batch_traces)
            result = processor_func(batch_stream)
            results.append(result)
            
            # Clear memory
            del batch_traces
            del batch_stream
    
    return results


# Memory profile function to help diagnose memory issues
def memory_profile(func):
    """
    Decorator to profile memory usage of a function.
    Requires the psutil package.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            result = func(*args, **kwargs)
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Memory usage: {func.__name__}: {mem_after - mem_before:.2f} MB")
            return result
        except ImportError:
            return func(*args, **kwargs)
    return wrapper
