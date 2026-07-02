PIC_EXPORT_CONTRACT_VERSION = 2
PIC_GRAPH_ROLE_PREFILL = 'prefill'
PIC_GRAPH_ROLE_DECODE = 'decode'
PIC_EXPORT_GRAPH_ROLES = (PIC_GRAPH_ROLE_PREFILL, PIC_GRAPH_ROLE_DECODE)

_DEVICE_ALIASES = {
    'jetson': 'jetson',
    'rhinopi': 'rhinopi',
    'orangepi': 'orangepi',
}

_BACKEND_ALIASES = {
    'generic': 'generic',
    'jetson': 'jetson',
    'rhinopi': 'rhinopi',
}

_DEVICE_FAMILY = {
    'generic': 'generic',
    'jetson': 'cuda',
    'rhinopi': 'adreno',
    'orangepi': 'generic',
}

_BACKEND_FAMILY = {
    'generic': 'generic',
    'jetson': 'cuda',
    'rhinopi': 'adreno',
}

_BACKEND_FROM_DEVICE = {
    'generic': 'generic',
    'jetson': 'jetson',
    'rhinopi': 'rhinopi',
    'orangepi': 'generic',
}

PIC_DECODE_REWRITE_FLAGS = (
    'pic_decode_tiny_fusion',
    'pic_decode_tiny_mlp_fusion',
    'pic_decode_gateup_fusion',
    'pic_decode_gateup_direct_fusion',
    'pic_decode_gateup_split_fusion',
    'pic_decode_silu_nhwc_down_fusion',
    'pic_decode_nhwc_linear_fusion',
)

PIC_GRAPH_ROLE_CONFIG_KEYS = (
    'pic_export_graph_role',
    'pic_decode_rewrites_enabled',
    'pic_decode_fusion_family',
    'pic_decode_nhwc_linear_scope',
) + PIC_DECODE_REWRITE_FLAGS

_NATIVE_DECODE_FLAGS = (
    'pic_decode_tiny_fusion',
    'pic_decode_gateup_fusion',
    'pic_decode_gateup_direct_fusion',
    'pic_decode_gateup_split_fusion',
    'pic_decode_nhwc_linear_fusion',
)

_ADRENO_ONLY_DECODE_FLAGS = (
    'pic_decode_tiny_mlp_fusion',
    'pic_decode_silu_nhwc_down_fusion',
)

_GATEUP_DEPENDENT_FLAGS = (
    'pic_decode_gateup_direct_fusion',
    'pic_decode_gateup_split_fusion',
    'pic_decode_silu_nhwc_down_fusion',
)


def _normalize_key(value):
    return str(value or '').strip().lower().replace('-', '_')


def _enabled(obj, name):
    return bool(getattr(obj, name, False))


def normalize_pic_export_device(value):
    key = _normalize_key(value)
    if not key:
        return None
    if key not in _DEVICE_ALIASES:
        raise ValueError(
            f'Unsupported --pic_export_device={value!r}. '
            'Use jetson, rhinopi, or orangepi.')
    return _DEVICE_ALIASES[key]


def normalize_pic_decode_fusion_backend(value):
    key = _normalize_key(value)
    if not key:
        return None
    if key not in _BACKEND_ALIASES:
        raise ValueError(
            f'Unsupported --pic_decode_fusion_backend={value!r}. '
            'Use generic, jetson, or rhinopi.')
    return _BACKEND_ALIASES[key]


def normalize_pic_export_graph_role(value):
    key = _normalize_key(value or PIC_GRAPH_ROLE_DECODE)
    if key not in PIC_EXPORT_GRAPH_ROLES:
        raise ValueError(
            f'Unsupported PIC export graph role {value!r}. '
            'Use prefill or decode.')
    return key


def _family_from_device(device):
    return _DEVICE_FAMILY.get(device or 'generic', 'generic')


def _family_from_backend(backend):
    return _BACKEND_FAMILY.get(backend or 'generic', 'generic')


def _device_from_backend(backend):
    family = _family_from_backend(backend)
    if family == 'cuda':
        return 'jetson'
    if family == 'adreno':
        return 'rhinopi'
    return 'generic'


def _bad_enabled(obj, names):
    return [name for name in names if _enabled(obj, name)]


class PicExportContract:
    def __init__(self, device, backend, family):
        self.device = device
        self.backend = backend
        self.family = family
        self.version = PIC_EXPORT_CONTRACT_VERSION

    def as_dict(self):
        return {
            'pic_export_contract_version': self.version,
            'pic_export_device': self.device,
            'pic_export_device_family': self.family,
            'pic_decode_fusion_backend': self.backend,
            'pic_decode_fusion_family': self.family,
        }


def resolve_pic_export_contract(obj, validate=False):
    requested_device = normalize_pic_export_device(getattr(obj, 'pic_export_device', None))
    requested_backend = normalize_pic_decode_fusion_backend(
        getattr(obj, 'pic_decode_fusion_backend', None))

    device = requested_device
    backend = requested_backend
    if device is None and backend is None:
        device = 'generic'
        backend = 'generic'
    elif device is None:
        device = _device_from_backend(backend)
    elif backend is None:
        backend = _BACKEND_FROM_DEVICE[device]

    device_family = _family_from_device(device)
    backend_family = _family_from_backend(backend)
    if device_family == 'generic' and backend_family != 'generic':
        raise ValueError(
            f'--pic_export_device={device} cannot use '
            f'--pic_decode_fusion_backend={backend}; export a device-specific model instead.')
    if backend_family != 'generic' and backend_family != device_family:
        raise ValueError(
            f'Conflicting PIC export target: --pic_export_device={device} '
            f'is {device_family}, but --pic_decode_fusion_backend={backend} is {backend_family}.')

    contract = PicExportContract(device, backend, backend_family)
    if validate:
        validate_pic_export_flags(obj, contract)
    return contract


def validate_pic_export_flags(obj, contract=None):
    contract = contract or resolve_pic_export_contract(obj, validate=False)
    if contract.family == 'generic':
        bad = _bad_enabled(obj, _NATIVE_DECODE_FLAGS + _ADRENO_ONLY_DECODE_FLAGS)
        if bad:
            raise ValueError(
                f'--pic_export_device={contract.device} uses the generic/Mali export family. '
                f'Disable backend-specific PIC decode fusion flags: {", ".join(bad)}.')
    elif contract.family == 'cuda':
        bad = _bad_enabled(obj, _ADRENO_ONLY_DECODE_FLAGS)
        if bad:
            raise ValueError(
                f'--pic_export_device={contract.device} uses the CUDA/Jetson export family. '
                f'These RhinoPi/Adreno-only flags are not allowed: {", ".join(bad)}.')
    elif contract.family != 'adreno':
        raise ValueError(f'Unsupported PIC decode fusion family: {contract.family}')

    bad_deps = _bad_enabled(obj, _GATEUP_DEPENDENT_FLAGS)
    if bad_deps and not _enabled(obj, 'pic_decode_gateup_fusion'):
        raise ValueError(
            f'{", ".join(bad_deps)} require --pic_decode_gateup_fusion for a well-defined export graph.')
    return contract


def normalize_pic_export_args(args):
    contract = resolve_pic_export_contract(args, validate=True)
    args.pic_export_contract_version = contract.version
    args.pic_export_device = contract.device
    args.pic_export_device_family = contract.family
    args.pic_decode_fusion_backend = contract.backend
    role = normalize_pic_export_graph_role(getattr(args, 'pic_export_graph_role', None))
    args.pic_export_graph_role = role
    args.pic_decode_fusion_family = 'generic' if role == PIC_GRAPH_ROLE_PREFILL else contract.family
    return contract


def pic_decode_fusion_backend(obj):
    return resolve_pic_export_contract(obj, validate=False).backend


def pic_decode_fusion_family(obj):
    role = normalize_pic_export_graph_role(getattr(obj, 'pic_export_graph_role', None))
    if role == PIC_GRAPH_ROLE_PREFILL:
        return 'generic'
    return resolve_pic_export_contract(obj, validate=False).family


def pic_graph_role_config_values(obj, graph_role):
    role = normalize_pic_export_graph_role(graph_role)
    contract = resolve_pic_export_contract(obj, validate=True)
    rewrite_enabled = role == PIC_GRAPH_ROLE_DECODE
    active_family = contract.family if rewrite_enabled else 'generic'
    native_decode = rewrite_enabled and contract.family in ('adreno', 'cuda')
    adreno_decode = rewrite_enabled and contract.family == 'adreno'
    gateup_fusion = native_decode and _enabled(obj, 'pic_decode_gateup_fusion')
    return {
        **contract.as_dict(),
        'pic_export_graph_role': role,
        'pic_decode_rewrites_enabled': rewrite_enabled,
        'pic_decode_fusion_family': active_family,
        'pic_decode_tiny_fusion': native_decode and _enabled(obj, 'pic_decode_tiny_fusion'),
        'pic_decode_tiny_mlp_fusion': adreno_decode and _enabled(obj, 'pic_decode_tiny_mlp_fusion'),
        'pic_decode_gateup_fusion': gateup_fusion,
        'pic_decode_gateup_direct_fusion': gateup_fusion and _enabled(obj, 'pic_decode_gateup_direct_fusion'),
        'pic_decode_gateup_split_fusion': gateup_fusion and _enabled(obj, 'pic_decode_gateup_split_fusion'),
        'pic_decode_silu_nhwc_down_fusion': (
            adreno_decode and gateup_fusion and _enabled(obj, 'pic_decode_silu_nhwc_down_fusion')),
        'pic_decode_nhwc_linear_fusion': native_decode and _enabled(obj, 'pic_decode_nhwc_linear_fusion'),
        'pic_decode_nhwc_linear_scope': getattr(obj, 'pic_decode_nhwc_linear_scope', 'all'),
    }


def pic_prefill_config_values(obj):
    return pic_graph_role_config_values(obj, PIC_GRAPH_ROLE_PREFILL)


def pic_decode_config_values(obj):
    return pic_graph_role_config_values(obj, PIC_GRAPH_ROLE_DECODE)
