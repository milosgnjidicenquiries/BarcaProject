"""Writing the one file the competition actually reads.

SUBMISSION.md is the contract, so the writer is deliberately dumb: a header line, then one row per
video frame in order, no index column, no blanks, no columns that are not asked for. Everything else
a run produces -- the report, the manifest, the transition table -- goes somewhere else, because a
submission that also carries a confidence column is a submission that fails validation.

The file is written with explicit LF newlines on every platform. Windows participants otherwise get
CRLF, which pandas reads happily and a byte-for-byte reproducibility check does not.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from .. import official

LABELS = official.pm.LABELS  # ("H", "A", "D", "X") -- taken from the scorer, never restated
_HEADER = "frame,label\n"


def sha256_file(path):
    """Hash of the artifact, so a reproducibility test can compare bytes rather than opinions."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def codes_to_names(codes):
    names = np.asarray(LABELS, dtype=object)
    return names[np.asarray(codes, dtype=np.intp)]


def write_submission(codes, path):
    """Write `codes` (0=H, 1=A, 2=D, 3=X) as frame,label rows and return (path, sha256, rows)."""
    codes = np.asarray(codes, dtype=np.int8).ravel()
    rows = len(codes)
    if not rows:
        raise ValueError("refusing to write an empty submission")
    if (codes < 0).any():
        # A -1 here is an unset frame, not a label. Silently naming index -1 "X" would turn a bug
        # into an abstention, which is the one mistake this writer cannot be allowed to make.
        raise ValueError("refusing to write {} frames with no label".format(int((codes < 0).sum())))

    names = codes_to_names(codes)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    digest = hashlib.sha256()
    with open(path, "w", newline="\n", encoding="ascii") as handle:
        handle.write(_HEADER)
        digest.update(_HEADER.encode("ascii"))
        # One join per chunk rather than one write per frame: 199319 separate writes is the slowest
        # possible way to produce this file and buys nothing.
        for start in range(0, rows, 65536):
            stop = min(rows, start + 65536)
            block = "\n".join("{},{}".format(frame, name)
                              for frame, name in enumerate(names[start:stop], start=start)) + "\n"
            handle.write(block)
            digest.update(block.encode("ascii"))

    return {"path": path, "sha256": digest.hexdigest(), "rows": rows,
            "label_counts": {name: int((codes == index).sum()) for index, name in enumerate(LABELS)}}


def read_submission(path):
    """Read a submission back through the official loader, so a writer bug and a reader bug cannot
    agree with each other and hide."""
    return official.pm.read_label_csv(path)


def verify_round_trip(codes, path):
    """True when the file on disk re-reads as exactly the codes that were written."""
    table = read_submission(path)
    reason = official.pm.validate_submission(table, len(codes))
    if reason:
        return False, reason
    read = official.pm.label_codes(table)
    if not np.array_equal(read, np.asarray(codes, dtype=np.int8)):
        first = int(np.flatnonzero(read != np.asarray(codes, dtype=np.int8))[0])
        return False, "differs from frame {}".format(first)
    return True, "identical"
