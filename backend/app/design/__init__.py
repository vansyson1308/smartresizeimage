"""Design representation: the editable, versioned source of truth.

A ``DesignDocument`` describes a master design as a typed program: elements
with stable ids, native text with style runs, asset references with content
hashes, semantic roles with confidence, relationships expressed as constraints,
allowed transformations and provenance. Preview, adaptation, verification,
editing and export all read and write this representation.
"""

from .document import (
    SCHEMA_VERSION,
    AllowedTransforms,
    AssetRef,
    Constraint,
    DesignDocument,
    Element,
    FontRef,
    Geometry,
    Provenance,
    TextContent,
    TextRun,
    TextStyle,
)
from .serialize import document_from_dict, document_to_dict, migrate_document_dict

__all__ = [
    "SCHEMA_VERSION",
    "AllowedTransforms",
    "AssetRef",
    "Constraint",
    "DesignDocument",
    "Element",
    "FontRef",
    "Geometry",
    "Provenance",
    "TextContent",
    "TextRun",
    "TextStyle",
    "document_from_dict",
    "document_to_dict",
    "migrate_document_dict",
]
