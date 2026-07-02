"""Canonical table schemas and storage helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence, Union
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

INTERACTION_COLUMNS = (
    "play_id",
    "play_title",
    "scene_id",
    "speech1_id",
    "speech2_id",
    "speaker1",
    "speaker2",
    "text1",
    "text2",
    "cosine_similarity",
    "model_id",
)
PLAY_FEATURE_COLUMNS = (
    "play_id",
    "title",
    "genre",
    "year",
    "representation",
    "feature",
    "value",
    "transform",
)
SPEECH_COLUMNS = ("speech_id", "play_id", "scene_id", "position", "speaker_id", "text")

SCHEMAS: Mapping[str, Sequence[str]] = {
    "speech_interactions": INTERACTION_COLUMNS,
    "play_features": PLAY_FEATURE_COLUMNS,
    "speeches": SPEECH_COLUMNS,
}


def validate_table(frame: pd.DataFrame, schema: str) -> None:
    expected = SCHEMAS[schema]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{schema} is missing columns: {', '.join(missing)}")


def write_table(frame: pd.DataFrame, path: Union[str, Path], *, schema: str = "") -> Path:
    target = Path(path)
    if schema:
        validate_table(frame, schema)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.casefold()
    if suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif suffix == ".csv":
        frame.to_csv(target, index=False)
    elif suffix == ".json":
        frame.to_json(target, orient="records", indent=2)
    elif suffix == ".xlsx":
        _write_xlsx(frame, target)
    else:
        raise ValueError(f"Unsupported table format: {target.suffix}")
    return target


def read_table(path: Union[str, Path], *, schema: str = "") -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        frame = pd.read_parquet(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix == ".json":
        frame = pd.read_json(source)
    else:
        raise ValueError(f"Unsupported table format: {source.suffix}")
    if schema:
        validate_table(frame, schema)
    return frame


def _excel_column(index: int) -> str:
    name = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(value: object, row: int, column: int) -> str:
    reference = f"{_excel_column(column)}{row}"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f'<c r="{reference}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = escape(str(value), {'"': "&quot;"})
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _write_xlsx(frame: pd.DataFrame, path: Path) -> None:
    rows = [list(frame.columns), *frame.astype(object).where(pd.notna(frame), None).values.tolist()]
    sheet_rows = []
    for row_number, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(value, row_number, column) for column, value in enumerate(row))
        sheet_rows.append(f'<row r="{row_number}">{cells}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml)
        workbook.writestr("_rels/.rels", rels_xml)
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)
