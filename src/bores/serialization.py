import base64
import sys
import threading
import typing
import warnings
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TypeGuard

import attrs
import cattrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.errors import DeserializationError, SerializationError, ValidationError

__all__ = ["Serializable", "converter", "dump", "load"]


_TYPE_SERIALIZERS: typing.Dict[
    typing.Type[typing.Any],
    typing.Callable[[typing.Any, bool], typing.Dict[str, typing.Any]],
] = {}
"""Registry of type serializers."""
_TYPE_DESERIALIZERS: typing.Dict[
    typing.Type[typing.Any],
    typing.Callable[[typing.Mapping[str, typing.Any]], typing.Any],
] = {}
"""Registry of type deserializers."""
_type_serializers_lock = threading.Lock()
_type_deserializers_lock = threading.Lock()


converter = cattrs.Converter()


def fallback_unstructure(value):
    return value


def fallback_structure(value, typ):
    return value


def _is_generic_alias(typ: typing.Any) -> bool:
    """Check if a type is a generic alias (e.g., List[int], Dict[str, float])"""
    return hasattr(typ, "__origin__") and typ.__origin__ is not None


converter.register_unstructure_hook_func(
    check_func=lambda t: _is_generic_alias(t) or not attrs.has(t),
    func=fallback_unstructure,
)
converter.register_structure_hook_func(
    check_func=lambda t: _is_generic_alias(t) or not attrs.has(t),
    func=fallback_structure,
)


def _get_origin_class(typ: typing.Any) -> typing.Optional[type]:
    """
    Extract the origin class from a generic type.

    Examples:
        RockPermeability[NDimension] -> RockPermeability
        List[int] -> list
        int -> int
    """
    if _is_generic_alias(typ):
        origin = typing.get_origin(typ)
        # For custom generic classes, get_origin returns the base class
        return origin
    elif isinstance(typ, type):
        return typ
    return None


def _is_serializable_type(typ: typing.Any) -> bool:
    """
    Check if a type (including generics) is a Serializable subclass.
    """
    origin = _get_origin_class(typ)
    if origin is None:
        return False

    try:
        return isinstance(origin, type) and issubclass(origin, Serializable)
    except TypeError:
        return False


def _is_optional_type(typ: typing.Any) -> bool:
    """Check if a type is Optional[T] (i.e., Union[T, None])."""
    if not _is_generic_alias(typ):
        return False

    origin = typing.get_origin(typ)
    if origin is not typing.Union:
        return False

    args = typing.get_args(typ)
    return type(None) in args


def _is_typed_dict_type(typ: typing.Any) -> TypeGuard[typing.Type[typing.Dict]]:
    return (
        isinstance(typ, type)
        and issubclass(typ, dict)
        and hasattr(typ, "__annotations__")
        and hasattr(typ, "__total__")
    )


def _is_namedtuple_type(typ: typing.Any) -> bool:
    return (
        isinstance(typ, type)
        and issubclass(typ, tuple)
        and hasattr(typ, "_fields")
        and isinstance(typ._fields, tuple)  # type: ignore[attr-defined]
    )


def _is_enum_type(typ: typing.Any) -> TypeGuard[typing.Type[Enum]]:
    return isinstance(typ, type) and issubclass(typ, Enum)


def _is_ndarray_type(typ: typing.Any) -> TypeGuard[typing.Type[npt.NDArray]]:
    return isinstance(typ, type) and issubclass(typ, np.ndarray)


def _unwrap_type(typ: typing.Any) -> typing.List[typing.Type[typing.Any]]:
    """
    Recursively unwrap a type to get all constituent non-None types.

    Examples:
        Optional[Foo] -> [Foo]
        Union[Foo, Bar, None] -> [Foo, Bar]
        List[Optional[Foo]] -> [List[Foo], Foo]
        Dict[str, Optional[Foo]] -> [Dict[str, Foo], Foo]
        Optional[List[Foo]] -> [List[Foo], Foo]
        Union[List[Foo], Dict[str, Bar]] -> [List[Foo], Dict[str, Bar], Foo, Bar]
    """
    result = []

    if typ is type(None):
        return []

    origin = typing.get_origin(typ)
    args = typing.get_args(typ)

    # Handle Union types (including Optional)
    if origin is typing.Union:
        for arg in args:
            if arg is not type(None):
                result.extend(_unwrap_type(arg))
        return result

    # For generic types, add the container first
    if origin is not None and args:
        result.append(typ)

        # Only unwrap nested custom types, not primitives
        for arg in args:
            arg_origin = _get_origin_class(arg)
            # Skip primitives like str, int, float
            if arg_origin and not _is_primitive_type(arg_origin):
                result.extend(_unwrap_type(arg))

        return result

    # Base case: non-generic type
    result.append(typ)
    return result


def _is_primitive_type(typ: type) -> bool:
    """Check if a type is a primitive built-in type."""
    return typ in (str, int, float, bool, bytes, type(None))


def _sort_types_by_specificity(
    types: typing.Iterable[typing.Type[typing.Any]],
) -> typing.List[typing.Type[typing.Any]]:
    """
    Sort types by specificity: more specific types (subclasses) come before general ones.

    This helps in prioritizing serializers/deserializers for more specific types first.
    """

    def specificity_key(typ: typing.Type[typing.Any]) -> int:
        if not isinstance(typ, type):
            return 0
        return len(typ.__mro__)  # More base classes = more general

    return sorted(types, key=specificity_key, reverse=True)


def _sort_types_by_primitivity(
    types: typing.Iterable[typing.Type[typing.Any]],
) -> typing.List[typing.Type[typing.Any]]:
    """
    Sort types by primitivity: non-primitive types come before primitive ones.

    This helps in prioritizing serializers/deserializers for complex types first.
    """

    def primitivity_key(typ: typing.Type[typing.Any]) -> int:
        return 0 if not _is_primitive_type(typ) else 1

    return sorted(types, key=primitivity_key)


def _get_primary_types(typ: typing.Any) -> typing.List[typing.Type[typing.Any]]:
    """
    Get the primary types to check for serializers/deserializers.

    This extracts the outermost meaningful types after unwrapping Unions/Optional,
    plus any deeply nested types that might have custom serializers.

    Examples:
        Optional[Foo] -> [Foo]
        List[Foo] -> [List[Foo], Foo]
        Dict[str, Foo] -> [Dict[str, Foo], Foo]
        Optional[List[Foo]] -> [List[Foo], Foo]
        Union[Foo, Bar] -> [Foo, Bar]
    """
    all_types = _unwrap_type(typ)

    # Remove duplicates while preserving order
    seen = set()
    unique_types = []
    for t in all_types:
        # Create a hashable representation
        type_id = id(t)
        if type_id not in seen:
            seen.add(type_id)
            unique_types.append(t)

    return unique_types


def _discover_type_serializers(
    fields: typing.Mapping[str, typing.Type[typing.Any]],
) -> typing.Dict[str, typing.Callable[[typing.Any, bool], typing.Any]]:
    """
    Auto-discover serializers for fields based on their types.

    Unwraps Optional, Union, and other generic containers to find
    all types that might need custom serializers.
    """
    discovered = {}

    with _type_serializers_lock:
        for field_name, field_type in fields.items():
            # Get all types to check (unwrapping Optional/Union and extracting generics)
            types_to_check = _get_primary_types(field_type)

            # Check each type in order of specificity (most specific first)
            for typ in types_to_check:
                origin = _get_origin_class(typ)
                if origin is None:
                    continue

                # Handle typing special forms (list, dict, etc.)
                if not isinstance(origin, type):
                    if origin in _TYPE_SERIALIZERS:
                        discovered[typ] = _TYPE_SERIALIZERS[origin]
                        break
                    continue

                # Walk MRO for class-based types
                for base in origin.__mro__:
                    if base in _TYPE_SERIALIZERS:
                        discovered[typ] = _TYPE_SERIALIZERS[base]
                        break
                else:
                    # Continue to next type if no serializer found
                    continue

                # Break outer loop if we found a serializer
                break

    return discovered  # type: ignore[return-value]


def _discover_type_deserializers(
    fields: typing.Mapping[str, typing.Type[typing.Any]],
) -> typing.Dict[str, typing.Callable[[typing.Any], typing.Any]]:
    """
    Auto-discover deserializers for fields based on their types.

    Unwraps Optional, Union, and other generic containers to find
    all types that might need custom deserializers.
    """
    discovered = {}

    with _type_deserializers_lock:
        for field_name, field_type in fields.items():
            # Get all types to check (unwrapping Optional/Union and extracting generics)
            types_to_check = _get_primary_types(field_type)

            # Check each type in order of specificity (most specific first)
            for typ in types_to_check:
                origin = _get_origin_class(typ)
                if origin is None:
                    continue

                # Handle typing special forms (list, dict, etc.)
                if not isinstance(origin, type):
                    if origin in _TYPE_DESERIALIZERS:
                        discovered[typ] = _TYPE_DESERIALIZERS[origin]
                        break
                    continue

                # Walk MRO to find deserializer
                for base in origin.__mro__:
                    if base in _TYPE_DESERIALIZERS:
                        discovered[typ] = _TYPE_DESERIALIZERS[base]
                        break
                else:
                    # Continue to next type if no deserializer found
                    continue

                # Break outer loop if we found a deserializer
                break

    return discovered  # type: ignore[return-value]


def _serialize(
    value: typing.Any,
    recurse: bool,
    serializers: typing.Optional[
        typing.Mapping[
            typing.Union[str, typing.Type],
            typing.Callable[[typing.Any, bool], typing.Any],
        ]
    ] = None,
    typ: typing.Optional[typing.Type[typing.Any]] = None,
    *,
    check_serializers: bool = True,
):
    """Dump a value using cattrs, handling nested `Serializable` objects."""
    # If no type provided, infer from value
    typ = typ if typ is not None else type(value)

    # Check for custom serializer first
    if check_serializers and serializers and typ in serializers:
        return serializers[typ](value, recurse)

    if _is_optional_type(typ):
        if value is None:
            return None

        args = typing.get_args(typ)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _serialize(value, recurse, serializers, non_none_args[0])
        else:
            # Try each non-None type in the Union
            for arg in _sort_types_by_primitivity(non_none_args):
                try:
                    return _serialize(value, recurse, serializers, arg)
                except Exception:
                    continue
            raise SerializationError(
                f"Value {value!r} does not match any type in {typ}"
            )

    if _is_serializable_type(typ):
        origin_cls = _get_origin_class(typ)
        return origin_cls.dump(value, recurse)  # type: ignore

    if _is_enum_type(typ):
        return typ(value).value

    # Handle generic types (List, Dict, etc.)
    if _is_generic_alias(typ):
        origin = typing.get_origin(typ)
        args = typing.get_args(typ)

        # Handle typing.Union and other special forms
        if origin is typing.Union:
            for arg in _sort_types_by_primitivity(args):
                try:
                    return _serialize(value, recurse, serializers, arg)
                except Exception:
                    continue
            raise SerializationError(
                f"Value {value!r} does not match any type in {typ}"
            )

        if (
            origin in (list, tuple, Sequence)
            or (origin and isinstance(origin, type) and issubclass(origin, Sequence))
        ) and (not isinstance(value, (str, bytes)) and isinstance(value, Sequence)):
            element_type = args[0] if args else type(None)
            return [_serialize(v, recurse, serializers, element_type) for v in value]

        if (
            origin in (dict, Mapping)
            or (origin and isinstance(origin, type) and issubclass(origin, Mapping))
        ) and isinstance(value, Mapping):
            key_type = args[0] if len(args) > 0 else type(None)
            value_type = args[1] if len(args) > 1 else type(None)
            return {
                _serialize(k, recurse, serializers, key_type): _serialize(
                    v, recurse, serializers, value_type
                )
                for k, v in value.items()
            }

        return _serialize(
            value, recurse, serializers, origin, check_serializers=check_serializers
        )

    if _is_typed_dict_type(typ) and isinstance(value, Mapping):
        annotations = typing.get_type_hints(typ, include_extras=False)
        return {
            k: _serialize(v, recurse, serializers, annotations.get(k, type(v)))
            for k, v in value.items()
        }

    if (
        _is_namedtuple_type(typ)
        and isinstance(value, tuple)
        and hasattr(value, "_fields")
    ):
        annotations = typing.get_type_hints(typ, include_extras=False)
        return {
            field: _serialize(
                getattr(value, field),
                recurse,
                serializers,
                annotations.get(field, type(getattr(value, field))),
            )
            for field in value._fields  # type: ignore[attr-defined]
        }

    # Fallback check for Mapping/Sequence at runtime
    # (for cases where type annotation isn't available or is too generic)
    if isinstance(value, Mapping):
        return {k: _serialize(v, recurse, serializers) for k, v in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_serialize(v, recurse, serializers) for v in value]

    return converter.unstructure(value)


def _deserialize(
    value: typing.Any,
    typ: typing.Type[typing.Any],
    deserializers: typing.Optional[
        typing.Mapping[
            typing.Union[str, typing.Type],
            typing.Callable[[typing.Any], typing.Any],
        ]
    ] = None,
    *,
    check_deserializers: bool = True,
) -> typing.Any:
    """Load a value using cattrs, handling nested `Serializable` objects."""
    # Check for custom deserializer first
    if check_deserializers and deserializers and typ in deserializers:
        return deserializers[typ](value)

    if _is_optional_type(typ):
        if value is None:
            return None

        args = typing.get_args(typ)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _deserialize(value, non_none_args[0], deserializers)
        else:
            for arg in _sort_types_by_primitivity(non_none_args):
                try:
                    return _deserialize(value, arg, deserializers)
                except Exception:
                    continue
            raise DeserializationError(
                f"Value {value!r} does not match any type in {typ}"
            )

    if _is_serializable_type(typ):
        origin_cls = _get_origin_class(typ)
        return origin_cls.load(value)  # type: ignore[union-attr]

    if _is_enum_type(typ):
        return typ(value)

    if _is_generic_alias(typ):
        origin = typing.get_origin(typ)
        args = typing.get_args(typ)

        if origin is typing.Union:
            for arg in _sort_types_by_primitivity(args):
                try:
                    return _deserialize(value, arg, deserializers)
                except Exception:
                    continue
            raise DeserializationError(
                f"Value {value!r} does not match any type in {typ}"
            )

        if origin in (list, tuple, Sequence) or (
            origin and isinstance(origin, type) and issubclass(origin, Sequence)
        ):
            return [_deserialize(v, args[0], deserializers) for v in value]

        if origin in (dict, Mapping) or (
            origin and isinstance(origin, type) and issubclass(origin, Mapping)
        ):
            return {
                k: _deserialize(v, args[1], deserializers) for k, v in value.items()
            }

        if origin is not None:
            return _deserialize(
                value, origin, deserializers, check_deserializers=check_deserializers
            )

    if _is_typed_dict_type(typ):
        annotations = typing.get_type_hints(typ, include_extras=False)
        return typ(
            {
                k: _deserialize(v, annotations[k], deserializers)
                for k, v in value.items()
                # Ignore keys not found in existing annotations incase typed-dict
                # structure changed for backwards compatibility
                if k in annotations
            }
        )

    if _is_namedtuple_type(typ):
        annotations = typing.get_type_hints(typ, include_extras=False)
        return typ(
            **{
                k: _deserialize(v, annotations[k], deserializers)
                for k, v in value.items()
                # Ignore keys not found in existing annotations incase namedtuple
                # structure changed for backwards compatibility
                if k in annotations
            }
        )

    return converter.structure(value, typ)


def _build_serializer(
    fields: typing.Mapping[str, typing.Type],
    exclude: typing.Optional[typing.Iterable[str]] = None,
    serializers: typing.Optional[
        typing.Mapping[
            typing.Union[str, typing.Type],
            typing.Callable[[typing.Any, bool], typing.Any],
        ]
    ] = None,
) -> typing.Callable:
    """
    Build a serializer function for the class.

    :param fields: Mapping of field names to types.
    :param exclude: Optional iterable of field names to exclude from serializing.
    :param serializers: Optional mapping of field names or types to custom serializer callables.
    :return: A serializer function.
    """
    # Cache for lazily discovered serializers
    _lazy_serializers_cache: typing.Dict[
        typing.Union[str, typing.Type[typing.Any]], typing.Any
    ] = {}
    _serializers_discovered = False

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        nonlocal _lazy_serializers_cache, _serializers_discovered

        # Lazy discovery: discover type serializers on first call
        if not _serializers_discovered:
            discovered = _discover_type_serializers(fields)
            # Merge: explicit serializers take precedence over discovered
            _lazy_serializers_cache = {**discovered, **(serializers or {})}  # type: ignore[dict-item]
            _serializers_discovered = True

        # Use the cached/discovered serializers
        active_serializers = _lazy_serializers_cache
        result = {}
        for field, typ in fields.items():
            if exclude and field in exclude:
                continue

            value = getattr(self, field)

            # Custom serializer by field name (highest priority)
            if active_serializers and (field in active_serializers):
                serializer = active_serializers[field]
                try:
                    result[field] = serializer(value, recurse)
                except Exception as exc:
                    raise SerializationError(
                        f"Failed to serialize field '{field}' using custom serializer"
                    ) from exc
                continue

            # Custom serializer by type (handle generics)
            if active_serializers:
                # Try exact type match first
                if typ in active_serializers:
                    serializer = active_serializers[typ]
                    try:
                        result[field] = serializer(value, recurse)
                    except Exception as exc:
                        raise SerializationError(
                            f"Failed to serialize field '{field}' of type {typ} using custom serializer"
                        ) from exc
                    continue

                # Then try origin class for generics
                origin = _get_origin_class(typ)
                if origin and origin != typ and origin in active_serializers:
                    serializer = active_serializers[origin]
                    try:
                        result[field] = serializer(value, recurse)
                    except Exception as exc:
                        raise SerializationError(
                            f"Failed to serialize field '{field}' of type {origin} using custom serializer"
                        ) from exc
                    continue

            if _is_serializable_type(typ):
                origin_cls = _get_origin_class(typ)
                try:
                    result[field] = origin_cls.dump(value, recurse)  # type: ignore
                except Exception as exc:
                    raise SerializationError(
                        f"Failed to serialize nested `Serializable` field '{field}'"
                    ) from exc
            else:
                try:
                    # No need to check serializers again here, as we've already done so above
                    result[field] = _serialize(
                        value=value,
                        recurse=recurse,
                        serializers=active_serializers,
                        typ=typ,
                        check_serializers=False,
                    )
                except Exception as exc:
                    raise SerializationError(
                        f"Failed to unstructure field '{field}' of type {typ}"
                    ) from exc

        return result

    return __dump__


def _build_deserializer(
    fields: typing.Mapping[str, typing.Type],
    exclude: typing.Optional[typing.Iterable[str]] = None,
    deserializers: typing.Optional[
        typing.Mapping[typing.Union[str, typing.Type], typing.Callable]
    ] = None,
) -> typing.Callable:
    """
    Build a deserializer function for the class.

    :param fields: Mapping of field names to types.
    :param exclude: Optional iterable of field names to exclude from deserializing.
    :param deserializers: Optional mapping of field names or types to custom deserializer callables.
    :return: A deserializer function.
    """
    # Cache for lazily discovered deserializers
    _lazy_deserializers_cache: typing.Dict[
        typing.Union[str, typing.Type[typing.Any]], typing.Any
    ] = {}
    _deserializers_discovered = False

    @classmethod  # type: ignore[misc]
    def __load__(cls, data: typing.Mapping[str, typing.Any]):
        nonlocal _lazy_deserializers_cache, _deserializers_discovered

        # Lazy discovery: discover type deserializers on first call
        if not _deserializers_discovered:
            discovered = _discover_type_deserializers(fields)
            # Merge: explicit deserializers take precedence over discovered
            _lazy_deserializers_cache = {**discovered, **(deserializers or {})}  # type: ignore[dict-item]
            _deserializers_discovered = True

        # Use the cached/discovered deserializers
        active_deserializers = _lazy_deserializers_cache

        init_kwargs = {}
        for field, typ in fields.items():
            if exclude and field in exclude:
                continue

            # Field must exist in data (let __init__ handle defaults)
            if field not in data:
                continue

            value = data[field]

            # Custom deserializer by field name (highest priority)
            if active_deserializers and (field in active_deserializers):
                deserializer = active_deserializers[field]
                try:
                    init_kwargs[field] = deserializer(value)
                except Exception as exc:
                    raise DeserializationError(
                        f"Failed to deserialize field '{field}' using custom deserializer"
                    ) from exc
                continue

            # Custom deserializer by type (handle generics)
            if active_deserializers:
                # Check exact type match first
                if typ in active_deserializers:
                    deserializer = active_deserializers[typ]
                    try:
                        init_kwargs[field] = deserializer(value)
                    except Exception as exc:
                        raise DeserializationError(
                            f"Failed to deserialize field '{field}' of type {typ} using custom deserializer"
                        ) from exc
                    continue

                # Then check origin class for generics
                origin = _get_origin_class(typ)
                if origin and origin != typ and origin in active_deserializers:
                    deserializer = active_deserializers[origin]
                    try:
                        init_kwargs[field] = deserializer(value)
                    except Exception as exc:
                        raise DeserializationError(
                            f"Failed to deserialize field '{field}' of type {origin} using custom deserializer"
                        ) from exc
                    continue

            # Check if it's a `Serializable` (including generics)
            if _is_serializable_type(typ):
                origin_cls = _get_origin_class(typ)
                try:
                    init_kwargs[field] = origin_cls.load(value)  # type: ignore[union-attr]
                except Exception as exc:
                    raise DeserializationError(
                        f"Failed to deserialize nested `Serializable` field '{field}' of type {typ}"
                    ) from exc
            else:
                try:
                    # No need to check deserializers again here, as we've already done so above
                    init_kwargs[field] = _deserialize(
                        value=value,
                        typ=typ,
                        deserializers=active_deserializers,
                        check_deserializers=False,
                    )
                except Exception as exc:
                    raise DeserializationError(
                        f"Failed to structure field '{field}' of type {typ}"
                    ) from exc

        return cls(**init_kwargs)

    return __load__


class SerializableMeta(type):
    """Metaclass for `Serializable` classes"""

    def __init__(
        cls,
        name: str,
        bases: typing.Tuple,
        namespace: typing.Dict[str, typing.Any],
        fields: typing.Optional[typing.Mapping[str, typing.Type]] = None,
        dump_exclude: typing.Optional[typing.Iterable[str]] = None,
        load_exclude: typing.Optional[typing.Iterable[str]] = None,
        serializers: typing.Optional[
            typing.Mapping[
                typing.Union[str, typing.Type],
                typing.Callable[[typing.Any, bool], typing.Any],
            ]
        ] = None,
        deserializers: typing.Optional[
            typing.Mapping[
                typing.Union[str, typing.Type],
                typing.Callable[[typing.Any], typing.Any],
            ]
        ] = None,
    ):
        super().__init__(name, bases, namespace)
        parent_serializers = {}
        parent_deserializers = {}
        parent_fields = {}
        for cl in reversed(cls.__mro__[1:-1]):  # Exclude `object`
            if serializable_fields := getattr(cl, "__serializable_fields__", None):
                parent_fields.update(serializable_fields)

            if serializable_serializers := getattr(
                cl, "__serializable_serializers__", None
            ):
                parent_serializers.update(serializable_serializers)

            if serializable_deserializers := getattr(
                cl, "__serializable_deserializers__", None
            ):
                parent_deserializers.update(serializable_deserializers)

        try:
            module = sys.modules.get(cls.__module__)
            annotations = typing.get_type_hints(
                cls,
                globalns=vars(module),
                localns=dict(vars(cls)),
                include_extras=False,
            )
        except NameError as exc:
            annotations = namespace.get("__annotations__", {})
            warnings.warn(
                f"Could not resolve type hints for {cls.__name__}: {exc}. "
                f"Using raw annotations which may not work well with forward references.",
                RuntimeWarning,
            )

        cls_fields = fields or annotations
        all_fields = {**parent_fields, **cls_fields}
        # Clean fields: remove any with value `None` or starting with dunder
        all_fields = {
            k: v
            for k, v in all_fields.items()
            if v is not None and not k.startswith("__")
        }

        # NOTE: We don't discover type serializers/deserializers here.
        # Discovery happens lazily on first dump/load call.
        # We only store explicitly provided serializers/deserializers.

        # Build final serializers/deserializers with proper precedence
        # - Parent class (medium priority)
        # - Explicit on this class (highest priority)
        # - Auto-discovered (lowest priority, added lazily at runtime)
        all_serializers = {
            **parent_serializers,
            **(serializers or {}),
        }
        all_deserializers = {
            **parent_deserializers,
            **(deserializers or {}),
        }

        is_abstract_cls = (
            "__abstract_serializable__" in namespace
            and namespace["__abstract_serializable__"]
        )
        if is_abstract_cls is False and not all_fields:
            raise ValidationError(
                "Serializable classes must have fields defined. If the class is an abstract base class, "
                "set `__abstract_serializable__` to True"
            )

        if all_fields:
            if "__dump__" not in namespace or getattr(
                namespace["__dump__"], "_is_placeholder", False
            ):
                cls.__dump__ = _build_serializer(
                    fields=all_fields,
                    exclude=dump_exclude,
                    serializers=all_serializers,
                )

            if "__load__" not in namespace or getattr(
                namespace["__load__"], "_is_placeholder", False
            ):
                cls.__load__ = _build_deserializer(
                    fields=all_fields,
                    exclude=load_exclude,
                    deserializers=all_deserializers,
                )

        cls.__serializable_fields__ = all_fields
        cls.__serializable_serializers__ = all_serializers
        cls.__serializable_deserializers__ = all_deserializers
        if "__abstract_serializable__" not in namespace:
            cls.__abstract_serializable__ = False


class Serializable(metaclass=SerializableMeta):
    """Base class for serializable objects."""

    __abstract_serializable__ = True

    def __init_subclass__(cls, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init_subclass__()

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Dump the object to a dictionary."""
        raise NotImplementedError

    @classmethod
    def __load__(cls: typing.Type[Self], data: typing.Mapping[str, typing.Any]) -> Self:
        """Load an object from a mapping."""
        raise NotImplementedError

    __dump__._is_placeholder = True  # type: ignore
    __load__._is_placeholder = True  # type: ignore

    def dump(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        try:
            return self.__dump__(recurse)
        except Exception as exc:
            raise SerializationError("Failed to dump serializable object") from exc

    @classmethod
    def load(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        try:
            return cls.__load__(data)
        except Exception as exc:
            raise DeserializationError(
                f"Failed to load serializable object of type {cls.__name__!r}"
            ) from exc


# Register `Serializable` with cattrs converter
def _structure_serializable(
    data: typing.Mapping[str, typing.Any], cls: typing.Type[Serializable]
) -> Serializable:
    return cls.load(data)


def _unstructure_serializable(obj: Serializable) -> typing.Mapping[str, typing.Any]:
    return obj.dump(recurse=True)


converter.register_structure_hook_func(
    lambda typ: _is_serializable_type(typ), _structure_serializable
)
converter.register_unstructure_hook_func(
    lambda typ: _is_serializable_type(typ), _unstructure_serializable
)


SerializableT = typing.TypeVar("SerializableT", bound=Serializable)
_SerializableT = typing.TypeVar("_SerializableT", bound=Serializable)


def dump(o: Serializable, /, recurse: bool = True) -> typing.Dict[str, typing.Any]:
    """Dump a `Serializable` object to a dictionary."""
    return o.__dump__(recurse)


def load(
    cls: typing.Type[SerializableT], data: typing.Mapping[str, typing.Any]
) -> SerializableT:
    """Load a `Serializable` object from a dictionary."""
    return cls.__load__(data)


def make_serializable_type_registrar(
    base_cls: typing.Type[SerializableT],
    registry: typing.Dict[str, typing.Type[SerializableT]],
    key_attr: str = "__type__",
    lock: typing.Optional[threading.Lock] = None,
    key_factory: typing.Optional[
        typing.Callable[[typing.Type[SerializableT]], str]
    ] = None,
    override: bool = False,
    auto_register_serializer: bool = True,
    auto_register_deserializer: bool = True,
) -> typing.Callable[[typing.Type[_SerializableT]], typing.Type[_SerializableT]]:
    """
    Decorator factory to create a registrar for `Serializable` subclasses.

    Creates a decorator to register `Serializable` subclasses in a registry.

    :param base_cls: The base class for the `Serializable` subclasses.
    :param registry: The registry to store the subclasses.
    :param key_attr: The attribute to use as the registry key.
    :param lock: An optional lock for thread safety.
    :param key_factory: An optional factory function to generate the registry key.
    :param override: Whether to allow overriding existing registrations.
    :return: A decorator to register `Serializable` subclasses.
    """
    if not key_attr:
        raise ValidationError("`key_attr` must be a non-empty string.")

    lock = lock or threading.Lock()

    def registrar(cls: typing.Type[_SerializableT]) -> typing.Type[_SerializableT]:
        """Decorator to register a `Serializable` subclass."""
        if not issubclass(cls, base_cls):
            raise ValidationError(
                f"Class {cls.__name__} is not a subclass of {base_cls.__name__}"
            )

        key = getattr(cls, key_attr, None)
        if not key:
            key = key_factory(cls) if key_factory else cls.__name__.lower()
            setattr(cls, key_attr, key)

        if not override and key in registry and not issubclass(cls, registry[key]):
            raise ValidationError(
                f"Class {cls.__name__} is already registered under key '{key}'. Rename the class or set a unique '{key_attr}'."
            )
        with lock:
            registry[key] = cls

        return cls  # type: ignore[return-value]

    if auto_register_serializer:
        serializer = make_registry_serializer(
            base_cls=base_cls,
            registry=registry,
            key_attr=key_attr,
        )
        register_type_serializer(
            typ=base_cls,
            serializer=serializer,
        )

    if auto_register_deserializer:
        deserializer = make_registry_deserializer(
            base_cls=base_cls,
            registry=registry,
        )
        register_type_deserializer(
            typ=base_cls,
            deserializer=deserializer,
        )

    registrar.__name__ = f"register_{base_cls.__name__.lower()}_type"
    return registrar


def make_registry_serializer(
    base_cls: typing.Type[SerializableT],
    registry: typing.Dict[str, typing.Type[SerializableT]],
    key_attr: str = "__type__",
) -> typing.Callable[[SerializableT, bool], typing.Dict[str, typing.Any]]:
    """
    Create a serializer function for a registry of `Serializable` subclasses.

    :param registry: The registry of `Serializable` subclasses.
    :param key_attr: The attribute used as the registry key.
    :return: A serializer function.
    """

    def serializer(
        obj: SerializableT, recurse: bool = True
    ) -> typing.Dict[str, typing.Any]:
        key = getattr(obj, key_attr, None)
        if not key or key not in registry:
            raise ValidationError(
                f"Unsupported {base_cls.__name__} type: {type(obj)!r}"
            )

        dump = {key: obj.dump(recurse)}
        return dump

    serializer.__name__ = f"serialize_{base_cls.__name__.lower()}"
    return serializer


def make_registry_deserializer(
    base_cls: typing.Type[SerializableT],
    registry: typing.Dict[str, typing.Type[SerializableT]],
) -> typing.Callable[[typing.Mapping[str, typing.Any]], SerializableT]:
    """
    Create a deserializer function for a registry of `Serializable` subclasses.

    :param registry: The registry of `Serializable` subclasses.
    :param key_attr: The attribute used as the registry key.
    :return: A deserializer function.
    """

    def deserializer(data: typing.Mapping[str, typing.Any]) -> SerializableT:
        if not isinstance(data, Mapping) or len(data) != 1:
            raise DeserializationError("Invalid data format for deserialization.")

        key, value = next(iter(data.items()))
        if key not in registry:
            raise DeserializationError(f"Unsupported {base_cls.__name__} type: {key!r}")

        cls = registry[key]
        return cls.load(value)

    deserializer.__name__ = f"deserialize_{base_cls.__name__.lower()}"
    return deserializer


T = typing.TypeVar("T")


def register_type_serializer(
    typ: typing.Type[T],
    serializer: typing.Callable[[T, bool], typing.Dict[str, typing.Any]],
) -> None:
    """Register a global type serializer for a specific type."""
    with _type_serializers_lock:
        _TYPE_SERIALIZERS[typ] = serializer


def register_type_deserializer(
    typ: typing.Type[T],
    deserializer: typing.Callable[[typing.Mapping[str, typing.Any]], T],
) -> None:
    """Register a global type deserializer for a specific type."""
    with _type_deserializers_lock:
        _TYPE_DESERIALIZERS[typ] = deserializer


def serialize_ndarray(arr: npt.ArrayLike) -> typing.Dict[str, typing.Any]:
    """
    Encode a numpy array to a JSON-safe dictionary.

    Preserves dtype (including endianness), shape, and exact byte content.
    Works for any numeric dtype: float16/32/64, int8-64, uint8-64, bool,
    complex64/128.

    :param arr: Array-like to encode. Non-ndarray inputs are first converted
        with `np.asarray`.
    :returns: Dict with keys `__ndarray__`, `dtype`, `shape`, `data`.

    Example:
    ```python
    wire = serialize_ndarray(np.array([1.0, 2.0], dtype=np.float32))
    # {'__ndarray__': True, 'dtype': '<f4', 'shape': [2], 'data': 'AACAQAAA...'}
    ```
    """
    a = np.asarray(arr)
    return {
        "__ndarray__": True,
        "dtype": a.dtype.str,  # e.g. "<f4", ">f8", "|b1"
        "shape": list(a.shape),
        "data": base64.b64encode(
            np.ascontiguousarray(a).tobytes()  # C-order bytes
        ).decode("ascii"),
    }


def deserialize_ndarray(
    data: typing.Union[
        typing.Mapping[str, typing.Any], typing.Sequence[typing.Any], npt.NDArray
    ],
    *,
    dtype: typing.Optional[typing.Union[str, npt.DTypeLike]] = None,
) -> npt.NDArray:
    """
    Decode a numpy array from its wire-format dict (produced by
    ``serialize_ndarray``).

    Also handles two convenience cases so callers never need to pre-check
    the input type:

    - **Plain list / nested list**: converted with ``np.asarray`` (dtype
      inferred; pass ``dtype=`` to override).
    - **Existing ndarray passthrough**: returned as-is, or cast if ``dtype=``
      is given.

    :param data:  Wire-format dict, plain Python list, or existing ndarray.
    :param dtype: Optional dtype override.  Ignored when ``data`` is a
                  wire-format dict (the stored dtype is always authoritative).
    :returns: Reconstructed ``np.ndarray``.
    :raises TypeError:  If ``data`` is an unrecognised type.
    :raises ValueError: If the wire dict is malformed or has a byte-length
                        mismatch.

    Example:
    ```python
    arr = deserialize_ndarray(
        {'__ndarray__': True, 'dtype': '<f4', 'shape': [2], 'data': 'AACAPwAAAEA='}
    )
    # array([1., 2.], dtype=float32)
    ```
    """
    if isinstance(data, np.ndarray):
        return data.astype(dtype, copy=False) if dtype is not None else data  # type: ignore[return-value]

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
        return np.asarray(data, dtype=dtype)

    if not isinstance(data, Mapping):
        raise TypeError(
            f"`deserialize_ndarray`: expected dict, list, or ndarray; "
            f"got {type(data).__name__!r}"
        )

    data = dict(data)
    if not data.get("__ndarray__"):
        raise ValueError(
            "`deserialize_ndarray`: dict is missing the '__ndarray__' sentinel. "
            "Did you pass a plain dict by mistake?"
        )

    missing = {"dtype", "shape", "data"} - data.keys()
    if missing:
        raise ValueError(
            f"`deserialize_ndarray`: wire dict is missing required keys: {missing}"
        )

    stored_dtype = np.dtype(data["dtype"])
    shape = tuple(int(s) for s in data["shape"])
    raw_bytes = base64.b64decode(data["data"])

    n_elements = int(np.prod(shape)) if shape else 1
    expected_bytes = stored_dtype.itemsize * n_elements
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"`deserialize_ndarray`: byte-length mismatch — "
            f"expected {expected_bytes} bytes "
            f"(dtype={stored_dtype}, shape={shape}), "
            f"got {len(raw_bytes)}"
        )

    # .copy() detaches from the read-only buffer returned by frombuffer
    return np.frombuffer(raw_bytes, dtype=stored_dtype).reshape(shape).copy()


def _structure_ndarray(data: typing.Any, cls: typing.Type[npt.NDArray]) -> npt.NDArray:
    return deserialize_ndarray(data)


def _unstructure_ndarray(obj: npt.NDArray) -> typing.Mapping[str, typing.Any]:
    return serialize_ndarray(obj)


def register_ndarray_serializers() -> None:
    """
    Register global ndarray serializer / deserializer with the bores
    serialization system.

    Call this **once** at application startup — ideally in the first marimo
    cell, before any `Serializable` objects are dumped or loaded. After
    registration, `np.ndarray` fields on *any* `Serializable` subclass
    are handled automatically without per-field `serializers=` arguments.

    Safe to call multiple times (subsequent calls silently overwrite with the
    same functions).

    Example:

    ```python
    import bores
    from ndarray_serialization import register_ndarray_serializers
    register_ndarray_serializers()
    ```
    """
    register_type_serializer(
        typ=np.ndarray,
        serializer=lambda arr, recurse: serialize_ndarray(arr),
    )
    register_type_deserializer(
        typ=np.ndarray,
        deserializer=deserialize_ndarray,
    )

    # Register `np.ndarray` with cattrs converter
    converter.register_structure_hook_func(
        lambda typ: _is_ndarray_type(typ), _structure_ndarray
    )
    converter.register_unstructure_hook_func(
        lambda typ: _is_ndarray_type(typ), _unstructure_ndarray
    )


def ndarray_serializer(
    arr: npt.NDArray, recurse: bool = True
) -> typing.Dict[str, typing.Any]:
    """
    Adapter matching the bores ``serializers=`` dict signature.

    Use this when you want to opt-in on a specific field rather than
    registering globally:

    ```python

    @attrs.frozen
    class MyGrid(
        Serializable,
        serializers={"kx": ndarray_serializer, "ky": ndarray_serializer},
        deserializers={"kx": ndarray_deserializer, "ky": ndarray_deserializer}
    ):
        kx: np.ndarray
        ky: np.ndarray
    ```
    """
    return serialize_ndarray(arr)


def ndarray_deserializer(data: typing.Any) -> np.ndarray:
    """
    Adapter matching the bores ``deserializers=`` dict signature.

    Pair with ``ndarray_serializer`` for explicit per-field registration.
    """
    return deserialize_ndarray(data)
