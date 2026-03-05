"""Content-hash generation for provenance verification.

Mirrors Week 1's spatial-hashing pattern: each LDU gets a SHA-256
digest so that downstream provenance chains can verify content
integrity even when the source document is re-paginated.
"""

from __future__ import annotations

import hashlib


def generate_content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*.

    The input is normalised (stripped of leading/trailing whitespace)
    before hashing so that cosmetic whitespace changes do not
    invalidate the provenance link.

    Parameters
    ----------
    text : str
        Raw chunk content.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    normalised = text.strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
