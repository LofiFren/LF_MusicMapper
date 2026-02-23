"""
coreaudio_aggregate.py - macOS CoreAudio aggregate device helper.

Creates temporary private aggregate devices to combine split USB audio devices
(e.g. when macOS enumerates a single USB interface as separate input-only and
output-only CoreAudio devices). The aggregate properly co-activates both USB
data paths, fixing the all-zeros-on-input problem.

Only imported on macOS (sys.platform == 'darwin'). Will fail to load on other
platforms due to missing frameworks.
"""

import ctypes
import struct
import time

# -- Load macOS frameworks --------------------------------------------
CF = ctypes.cdll.LoadLibrary(
    '/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation'
)
CA = ctypes.cdll.LoadLibrary(
    '/System/Library/Frameworks/CoreAudio.framework/CoreAudio'
)


# -- Helpers --------------------------

def _fourcc(s):
    """Convert a 4-char ASCII string to a big-endian UInt32."""
    return struct.unpack('>I', s.encode('ascii'))[0]


# -- CoreFoundation setup --------------------------------------------

kCFStringEncodingUTF8 = 0x08000100
kCFNumberSInt32Type = 3

CF.CFStringCreateWithCString.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
]
CF.CFStringCreateWithCString.restype = ctypes.c_void_p

CF.CFDictionaryCreate.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_long,
    ctypes.c_void_p, ctypes.c_void_p,
]
CF.CFDictionaryCreate.restype = ctypes.c_void_p

CF.CFArrayCreate.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_long, ctypes.c_void_p,
]
CF.CFArrayCreate.restype = ctypes.c_void_p

CF.CFNumberCreate.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
]
CF.CFNumberCreate.restype = ctypes.c_void_p

CF.CFRelease.argtypes = [ctypes.c_void_p]
CF.CFRelease.restype = None

CF.CFStringGetCString.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
]
CF.CFStringGetCString.restype = ctypes.c_bool

# Callback-struct addresses (global structs, not pointers)
_kDictKeyCB = ctypes.addressof(
    ctypes.c_void_p.in_dll(CF, 'kCFTypeDictionaryKeyCallBacks')
)
_kDictValCB = ctypes.addressof(
    ctypes.c_void_p.in_dll(CF, 'kCFTypeDictionaryValueCallBacks')
)
_kArrayCB = ctypes.addressof(
    ctypes.c_void_p.in_dll(CF, 'kCFTypeArrayCallBacks')
)


# -- CoreAudio setup -------------------------------------------------

class AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ('mSelector', ctypes.c_uint32),
        ('mScope',    ctypes.c_uint32),
        ('mElement',  ctypes.c_uint32),
    ]

kAudioObjectSystemObject       = 1
kAudioObjectPropertyScopeGlobal = _fourcc('glob')
kAudioObjectPropertyScopeInput  = _fourcc('inpt')
kAudioObjectPropertyScopeOutput = _fourcc('outp')
kAudioObjectPropertyElementMain = 0

kAudioHardwarePropertyDevices  = _fourcc('dev#')
kAudioDevicePropertyDeviceUID  = _fourcc('uid ')
kAudioObjectPropertyName       = _fourcc('lnam')
kAudioDevicePropertyStreams    = _fourcc('stm#')

CA.AudioObjectGetPropertyDataSize.argtypes = [
    ctypes.c_uint32, ctypes.POINTER(AudioObjectPropertyAddress),
    ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
]
CA.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32

CA.AudioObjectGetPropertyData.argtypes = [
    ctypes.c_uint32, ctypes.POINTER(AudioObjectPropertyAddress),
    ctypes.c_uint32, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
]
CA.AudioObjectGetPropertyData.restype = ctypes.c_int32

CA.AudioHardwareCreateAggregateDevice.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
]
CA.AudioHardwareCreateAggregateDevice.restype = ctypes.c_int32

CA.AudioHardwareDestroyAggregateDevice.argtypes = [ctypes.c_uint32]
CA.AudioHardwareDestroyAggregateDevice.restype = ctypes.c_int32


# -- CF convenience helpers -------------------------------------------

def _cfstr(s):
    return CF.CFStringCreateWithCString(None, s.encode('utf-8'),
                                        kCFStringEncodingUTF8)

def _cfnum(value):
    val = ctypes.c_int32(value)
    return CF.CFNumberCreate(None, kCFNumberSInt32Type, ctypes.byref(val))

def _cfarray(items):
    arr = (ctypes.c_void_p * len(items))(*items)
    return CF.CFArrayCreate(None, arr, len(items), _kArrayCB)

def _cfdict(pairs):
    """Create CFDictionary from list of (key, value) tuples."""
    n = len(pairs)
    keys = (ctypes.c_void_p * n)()
    vals = (ctypes.c_void_p * n)()
    for i, (k, v) in enumerate(pairs):
        keys[i] = k
        vals[i] = v
    return CF.CFDictionaryCreate(None, keys, vals, n, _kDictKeyCB, _kDictValCB)

def _cfstr_to_python(cf_str):
    """Convert a CFStringRef to a Python str.  Returns None on failure."""
    if not cf_str:
        return None
    buf = ctypes.create_string_buffer(512)
    if CF.CFStringGetCString(cf_str, buf, 512, kCFStringEncodingUTF8):
        return buf.value.decode('utf-8')
    return None


# -- CoreAudio device helpers -----------------------------------------

def _get_all_device_ids():
    """Return every CoreAudio AudioDeviceID on the system."""
    addr = AudioObjectPropertyAddress(
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    )
    size = ctypes.c_uint32(0)
    status = CA.AudioObjectGetPropertyDataSize(
        kAudioObjectSystemObject, ctypes.byref(addr), 0, None,
        ctypes.byref(size),
    )
    if status != 0:
        return []
    count = size.value // ctypes.sizeof(ctypes.c_uint32)
    ids = (ctypes.c_uint32 * count)()
    status = CA.AudioObjectGetPropertyData(
        kAudioObjectSystemObject, ctypes.byref(addr), 0, None,
        ctypes.byref(size), ids,
    )
    if status != 0:
        return []
    return list(ids)


def _get_device_name(device_id):
    """Get the human-readable name of a CoreAudio device."""
    addr = AudioObjectPropertyAddress(
        kAudioObjectPropertyName,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    )
    cf_str = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
    status = CA.AudioObjectGetPropertyData(
        device_id, ctypes.byref(addr), 0, None,
        ctypes.byref(size), ctypes.byref(cf_str),
    )
    if status != 0 or not cf_str.value:
        return None
    name = _cfstr_to_python(cf_str)
    CF.CFRelease(cf_str)
    return name


def _get_device_uid(device_id):
    """Get the UID string of a CoreAudio device."""
    addr = AudioObjectPropertyAddress(
        kAudioDevicePropertyDeviceUID,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    )
    cf_str = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
    status = CA.AudioObjectGetPropertyData(
        device_id, ctypes.byref(addr), 0, None,
        ctypes.byref(size), ctypes.byref(cf_str),
    )
    if status != 0 or not cf_str.value:
        return None
    uid = _cfstr_to_python(cf_str)
    CF.CFRelease(cf_str)
    return uid


def _has_streams(device_id, scope):
    """True if the device has audio streams in the given scope."""
    addr = AudioObjectPropertyAddress(
        kAudioDevicePropertyStreams, scope, kAudioObjectPropertyElementMain,
    )
    size = ctypes.c_uint32(0)
    status = CA.AudioObjectGetPropertyDataSize(
        device_id, ctypes.byref(addr), 0, None, ctypes.byref(size),
    )
    return status == 0 and size.value > 0


# -- Public API -----------------------

AGG_DEVICE_NAME = "LF MusicMapper Aggregate"


def get_uids_for_device_name(device_name):
    """Find all CoreAudio devices matching *device_name*.

    Returns a list of ``(uid, has_input, has_output)`` tuples.
    """
    results = []
    for dev_id in _get_all_device_ids():
        name = _get_device_name(dev_id)
        if name != device_name:
            continue
        uid = _get_device_uid(dev_id)
        if not uid:
            continue
        has_in  = _has_streams(dev_id, kAudioObjectPropertyScopeInput)
        has_out = _has_streams(dev_id, kAudioObjectPropertyScopeOutput)
        results.append((uid, has_in, has_out))
    return results


def create_aggregate_device(sub_uids, master_uid):
    """Create a private CoreAudio aggregate device.

    Args:
        sub_uids:   List of sub-device UID strings to combine.
        master_uid: UID of the clock-master sub-device (typically the output).

    Returns:
        CoreAudio AudioDeviceID of the new aggregate, or ``None`` on failure.
    """
    agg_uid = f"com.lfmusicmapper.agg.{int(time.time() * 1000)}"

    # Build sub-device dictionaries — drift compensation for non-master
    sub_dicts = []
    for uid in sub_uids:
        pairs = [(_cfstr("uid"), _cfstr(uid))]
        if uid != master_uid:
            pairs.append((_cfstr("drift"), _cfnum(1)))
        sub_dicts.append(_cfdict(pairs))

    desc = _cfdict([
        (_cfstr("uid"),        _cfstr(agg_uid)),
        (_cfstr("name"),       _cfstr(AGG_DEVICE_NAME)),
        (_cfstr("subdevices"), _cfarray(sub_dicts)),
        (_cfstr("master"),     _cfstr(master_uid)),
        (_cfstr("private"),    _cfnum(1)),
        (_cfstr("stacked"),    _cfnum(0)),
    ])

    new_id = ctypes.c_uint32(0)
    status = CA.AudioHardwareCreateAggregateDevice(desc, ctypes.byref(new_id))
    CF.CFRelease(desc)

    if status != 0:
        return None

    # CoreAudio sets up the aggregate asynchronously — brief settle time
    time.sleep(0.2)
    return new_id.value


def destroy_aggregate_device(device_id):
    """Destroy a previously created aggregate device."""
    status = CA.AudioHardwareDestroyAggregateDevice(device_id)
    if status != 0:
        raise RuntimeError(
            f"AudioHardwareDestroyAggregateDevice failed (status={status})"
        )
