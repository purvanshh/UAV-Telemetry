"""Upload helpers: individual CSVs and ZIP archives → named sensor streams."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.schema import STREAM_FILENAME_HINTS, STREAMS
from src.validation import FlightValidationResult, read_csv_bytes, validate_flight


BytesLike = Union[bytes, bytearray, BinaryIO]


@dataclass
class UploadResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    filenames: Dict[str, str] = field(default_factory=dict)
    file_sizes: Dict[str, int] = field(default_factory=dict)
    validation: Optional[FlightValidationResult] = None
    source: str = ""  # "individual" | "zip"

    @property
    def all_errors(self) -> List[str]:
        errs = list(self.errors)
        if self.validation is not None:
            errs.extend(self.validation.errors)
        return errs

    @property
    def all_warnings(self) -> List[str]:
        warns = list(self.warnings)
        if self.validation is not None:
            warns.extend(self.validation.warnings)
        return warns


def _read_bytes(data: BytesLike) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    data.seek(0)
    return data.read()


def _basename_hint(name: str) -> str:
    return Path(name).name.lower()


def classify_stream_from_name(name: str) -> Optional[str]:
    """Map a filename or ZIP member path to IMU / ATT / BARO.

    Checks the file stem, basename tokens, and parent directory names
    (e.g. ``flight/IMU/00000001.csv``). Returns None if ambiguous or unrecognized.
    """
    path = Path(name)
    base = path.name.lower()
    stem = path.stem.lower()
    parts_lower = [p.lower() for p in path.parts]

    exact = {"imu": "IMU", "att": "ATT", "baro": "BARO"}
    if stem in exact:
        return exact[stem]

    # Parent directory named IMU / ATT / BARO
    for part in parts_lower[:-1]:
        if part in exact:
            return exact[part]

    hits: List[str] = []
    # Search the full relative path so folder+filename layouts work
    haystack = name.replace("\\", "/").lower()
    for stream, hints in STREAM_FILENAME_HINTS.items():
        for h in hints:
            if re.search(rf"(^|[_\-\s/]){re.escape(h)}($|[_\-\s./])", haystack):
                hits.append(stream)
                break
            if stem == h or base == f"{h}.csv":
                hits.append(stream)
                break

    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0]
    return None


def _parse_csv_member(
    raw: bytes,
    filename: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = read_csv_bytes(raw, filename=filename)
        return df, None
    except ValueError as exc:
        return None, str(exc)


def load_individual_csvs(
    files: Dict[str, BytesLike],
    filenames: Optional[Dict[str, str]] = None,
) -> UploadResult:
    """Load explicitly labeled IMU / ATT / BARO uploads.

    `files` keys must be stream names (IMU, ATT, BARO).
    """
    result = UploadResult(ok=True, source="individual")
    filenames = filenames or {}

    for stream in STREAMS:
        if stream not in files or files[stream] is None:
            continue
        raw = _read_bytes(files[stream])
        name = filenames.get(stream) or f"{stream}.csv"
        result.file_sizes[stream] = len(raw)
        result.filenames[stream] = name

        # Guard against wrong stream assignment by filename when provided
        guessed = classify_stream_from_name(name)
        if guessed is not None and guessed != stream:
            result.ok = False
            result.errors.append(
                f"File '{name}' looks like {guessed} data but was uploaded as "
                f"{stream}. Re-assign the file or rename it to match the stream."
            )
            continue

        df, err = _parse_csv_member(raw, name)
        if err:
            result.ok = False
            result.errors.append(err)
            continue
        result.frames[stream] = df

    missing = [s for s in STREAMS if s not in result.frames]
    if missing:
        result.ok = False
        result.errors.append(
            f"Missing required upload(s): {', '.join(missing)}."
        )

    if result.frames:
        result.validation = validate_flight(
            result.frames,
            filenames=result.filenames,
            file_sizes=result.file_sizes,
        )
        if not result.validation.ok:
            result.ok = False

    return result


def load_zip_archive(
    data: BytesLike,
    archive_name: str = "upload.zip",
) -> UploadResult:
    """Extract and map IMU / ATT / BARO CSVs from a ZIP archive."""
    result = UploadResult(ok=True, source="zip")
    raw = _read_bytes(data)

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        result.ok = False
        result.errors.append(
            f"'{archive_name}' is not a valid ZIP archive."
        )
        return result

    # Collect CSV members (skip macOS junk / directories)
    members: List[str] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        base = Path(name).name
        if base.startswith(".") or base.startswith("__MACOSX"):
            continue
        if ".__macosx" in name.lower():
            continue
        if not base.lower().endswith(".csv"):
            continue
        members.append(name)

    if not members:
        result.ok = False
        result.errors.append(
            f"'{archive_name}' contains no CSV files."
        )
        return result

    assigned: Dict[str, str] = {}
    ambiguous: List[str] = []
    unrecognized: List[str] = []

    for member in members:
        stream = classify_stream_from_name(member)
        if stream is None:
            unrecognized.append(member)
            continue
        if stream in assigned:
            ambiguous.append(
                f"{stream} matched both '{assigned[stream]}' and '{member}'"
            )
            continue
        assigned[stream] = member

    if ambiguous:
        result.ok = False
        result.errors.append(
            "Ambiguous ZIP contents — multiple files mapped to the same stream: "
            + "; ".join(ambiguous)
            + ". Use clear names such as IMU.csv, ATT.csv, BARO.csv."
        )
        return result

    missing = [s for s in STREAMS if s not in assigned]
    if missing:
        result.ok = False
        hint = ""
        if unrecognized:
            hint = f" Unrecognized CSV(s): {', '.join(unrecognized)}."
        result.errors.append(
            f"ZIP is missing stream(s): {', '.join(missing)}."
            f" Expected filenames containing imu, att, and baro.{hint}"
        )
        return result

    if unrecognized:
        result.warnings.append(
            "Ignored unrecognized CSV(s) in ZIP: "
            + ", ".join(unrecognized)
        )

    for stream, member in assigned.items():
        member_bytes = zf.read(member)
        name = Path(member).name
        result.filenames[stream] = name
        result.file_sizes[stream] = len(member_bytes)
        df, err = _parse_csv_member(member_bytes, name)
        if err:
            result.ok = False
            result.errors.append(err)
            continue
        result.frames[stream] = df

    if len(result.frames) == len(STREAMS):
        result.validation = validate_flight(
            result.frames,
            filenames=result.filenames,
            file_sizes=result.file_sizes,
        )
        if not result.validation.ok:
            result.ok = False
    elif result.ok:
        result.ok = False
        result.errors.append("Failed to load all three streams from ZIP.")

    return result
