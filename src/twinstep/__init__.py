"""twinstep: seeded-error twin generator for generic liquid-handling protocols."""

from twinstep.baseline import predict_record
from twinstep.evaluate import evaluate_records
from twinstep.injectors import ERROR_TYPES, generate_dataset, inject_error
from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    TransferStep,
    WashStep,
    load_protocol,
)

__version__ = "0.1.0"

__all__ = [
    "ERROR_TYPES",
    "IncubateStep",
    "MixStep",
    "Protocol",
    "ReadStep",
    "TransferStep",
    "WashStep",
    "__version__",
    "evaluate_records",
    "generate_dataset",
    "inject_error",
    "load_protocol",
    "predict_record",
]
