from __future__ import annotations

import inspect
from dataclasses import dataclass, field, Field, fields
from typing import Any, Callable, TypeVar, overload
from logging import getLogger as get_logger

logger = get_logger("conditional_fields")
T = TypeVar("T")

_inputs_key: str = "inputs"
_default_factory_key: str = "default_factory"


# A sentinel object to detect if a parameter is waiting for its dependencies or not.  Use
# a class to give it a better repr.
class _MISSING_CONDITIONAL_TYPE:
    pass


MISSING_CONDITIONAL = _MISSING_CONDITIONAL_TYPE()

# IDEA: Could maybe create a ConditionalField object, that inherits from `dataclasses.Field`?


@overload
def conditional_field(
    default_factory: Callable[..., T],
    inputs: None = None,
) -> T:
    ...


# The number of specified 'inputs' must match exactly the number of arguments to the function.
@overload
def conditional_field(default_factory: Callable[[Any], T], inputs: str) -> T:
    ...


@overload
def conditional_field(
    default_factory: Callable[[Any, Any], T], inputs: tuple[str, str]
) -> T:
    ...


@overload
def conditional_field(
    default_factory: Callable[[Any, Any, Any], T], inputs: tuple[str, str, str]
) -> T:
    ...


def conditional_field(
    default_factory: Callable[..., T], inputs: str | tuple[str, ...] | None = None
) -> T:
    if inputs is None:
        signature = inspect.signature(default_factory)
        input_names = tuple(signature.parameters)
    elif isinstance(inputs, str):
        input_names = (inputs,)
    else:
        input_names = tuple(inputs)

    return field(
        default=MISSING_CONDITIONAL,  # type: ignore
        metadata={
            _inputs_key: input_names,
            _default_factory_key: default_factory,
        },
    )


def is_conditional(field: Field) -> bool:
    return _inputs_key in field.metadata and _default_factory_key in field.metadata


def _get_input_names(f: Field) -> list[str]:
    assert is_conditional(f)
    return list(f.metadata[_inputs_key])


def _get_conditional_default_factory(f: Field) -> Callable:
    assert is_conditional(f)
    return f.metadata[_default_factory_key]


def set_conditionals(obj) -> None:
    """Sets the conditional fields on `obj` by resolving the dependencies and calling the factories."""
    t_fields = fields(obj)
    if not any(is_conditional(f) for f in t_fields):
        return

    def _is_set(f: Field) -> bool:
        return getattr(obj, f.name) is not MISSING_CONDITIONAL

    def _get_input_fields(f: Field) -> list[Field]:
        assert is_conditional(f)
        input_names = _get_input_names(f)
        for input_name in input_names:
            if not hasattr(obj, input_name):
                raise RuntimeError(
                    f"Field {f.name} is conditioned on the value of '{input_name}', but there is "
                    f"no field with that name on type {type(obj)}!"
                )
        return [f for f in fields(obj) if f.name in input_names]

    def _get_conditional_fields_left() -> list[Field]:
        return sorted(
            [f for f in t_fields if is_conditional(f) and f.init and not _is_set(f)],
            key=lambda f: f.name,
        )

    conditional_fields_left = _get_conditional_fields_left()

    while conditional_fields_left:
        # Find all the fields whose dependencies are all set.
        leaves = [
            f
            for f in conditional_fields_left
            if all(_is_set(input_field) for input_field in _get_input_fields(f))
        ]
        if not leaves:
            raise RuntimeError(
                f"There are conditional fields left, but no leaves, so there must be a "
                f"dependency cycle between fields {[f.name for f in conditional_fields_left]}!"
            )

        for field in leaves:
            input_names = _get_input_names(field)
            default_factory = _get_conditional_default_factory(field)
            logger.debug(
                f"Instantiating field {field.name} using its dependencies {input_names}"
            )

            factory_fn_inputs = {name: getattr(obj, name) for name in input_names}
            value = default_factory(**factory_fn_inputs)
            setattr(obj, field.name, value)

        conditional_fields_left = _get_conditional_fields_left()


@dataclass
class WithConditionalFields:
    def __post_init__(self):
        set_conditionals(self)
