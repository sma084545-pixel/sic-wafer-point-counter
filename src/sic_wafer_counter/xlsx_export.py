"""Small dependency-free XLSX writer for auditable result tables.

The project needs Excel-compatible output in local/offline installations and in
WPS Office.  This module writes a conservative OOXML subset using only the
standard library; values remain typed and worksheets are streamed into ZIP so a
large candidate table is not duplicated in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Callable, Iterable, Iterator, Sequence
from xml.sax.saxutils import escape, quoteattr
import zipfile


CellValue = str | int | float | bool | None
RowsFactory = Callable[[], Iterable[Sequence[CellValue]]]


@dataclass(frozen=True, slots=True)
class SheetSpec:
    """Declarative data and lightweight formatting for one worksheet."""

    name: str
    rows: RowsFactory
    column_widths: Sequence[float] = field(default_factory=tuple)
    title_rows: frozenset[int] = frozenset()
    header_rows: frozenset[int] = frozenset()
    freeze_rows: int = 0
    auto_filter_ref: str | None = None


def _column_name(index: int) -> str:
    value = index + 1
    name = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _clean_text(value: object) -> str:
    text = str(value)
    return "".join(
        character
        for character in text
        if character in "\t\n\r" or ord(character) >= 32
    )


def _sheet_name(value: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", value).strip("'")[:31] or "Sheet"
    candidate = cleaned
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{cleaned[:31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _style_for(value: CellValue, row_number: int, spec: SheetSpec) -> int:
    if row_number in spec.title_rows:
        return 1
    if row_number in spec.header_rows:
        return 2
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 4
    if isinstance(value, float):
        return 3
    return 0


def _cell_xml(value: CellValue, row_number: int, column_index: int, spec: SheetSpec) -> str:
    reference = f"{_column_name(column_index)}{row_number}"
    style = _style_for(value, row_number, spec)
    style_attr = f' s="{style}"' if style else ""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f'<c r="{reference}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{reference}"{style_attr}><v>{value}</v></c>'
    text = escape(_clean_text(value))
    return (
        f'<c r="{reference}" t="inlineStr"{style_attr}>'
        f'<is><t xml:space="preserve">{text}</t></is></c>'
    )


def _worksheet_chunks(spec: SheetSpec) -> Iterator[bytes]:
    yield (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    ).encode()
    pane = ""
    if spec.freeze_rows > 0:
        top = spec.freeze_rows + 1
        pane = (
            f'<pane ySplit="{spec.freeze_rows}" topLeftCell="A{top}" '
            'activePane="bottomLeft" state="frozen"/>'
        )
    yield (
        '<sheetViews><sheetView workbookViewId="0" showGridLines="0">'
        f'{pane}</sheetView></sheetViews><sheetFormatPr defaultRowHeight="16"/>'
    ).encode()
    if spec.column_widths:
        columns = "".join(
            f'<col min="{index + 1}" max="{index + 1}" width="{max(4.0, min(float(width), 48.0)):.1f}" customWidth="1"/>'
            for index, width in enumerate(spec.column_widths)
        )
        yield f"<cols>{columns}</cols>".encode()
    yield b"<sheetData>"
    for row_number, row in enumerate(spec.rows(), start=1):
        cells = "".join(
            _cell_xml(value, row_number, column_index, spec)
            for column_index, value in enumerate(row)
        )
        height = ' ht="26" customHeight="1"' if row_number in spec.title_rows else ""
        yield f'<row r="{row_number}"{height}>{cells}</row>'.encode("utf-8")
    yield b"</sheetData>"
    if spec.auto_filter_ref:
        yield f"<autoFilter ref={quoteattr(spec.auto_filter_ref)}/>".encode("utf-8")
    yield b"<pageMargins left=\"0.5\" right=\"0.5\" top=\"0.6\" bottom=\"0.6\" header=\"0.2\" footer=\"0.2\"/></worksheet>"


_STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts>
  <fonts count="3">
    <font><sz val="10"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FF234047"/><sz val="10"/><name val="Calibri"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF0F766E"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EEEC"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFCBD8D6"/></left><right style="thin"><color rgb="FFCBD8D6"/></right><top style="thin"><color rgb="FFCBD8D6"/></top><bottom style="thin"><color rgb="FFCBD8D6"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="1" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def write_xlsx(path: str | Path, sheets: Sequence[SheetSpec]) -> Path:
    """Write a valid XLSX workbook and return its path."""

    if not sheets:
        raise ValueError("At least one worksheet is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    names = [_sheet_name(spec.name, used) for spec in sheets]
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    content_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
{content_overrides}</Types>"""
    workbook_sheets = "".join(
        f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(names, start=1)
    )
    workbook_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><bookViews><workbookView/></bookViews><sheets>{workbook_sheets}</sheets><calcPr calcId="191029"/></workbook>"""
    sheet_rels = "".join(
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    workbook_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{sheet_rels}<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>"""
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>SiC wafer point counter</dc:creator><cp:lastModifiedBy>SiC wafer point counter</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified></cp:coreProperties>"""
    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>SiC wafer point counter</Application></Properties>"""

    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", root_rels)
            archive.writestr("docProps/core.xml", core_xml)
            archive.writestr("docProps/app.xml", app_xml)
            archive.writestr("xl/workbook.xml", workbook_xml)
            archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
            archive.writestr("xl/styles.xml", _STYLES_XML)
            for index, spec in enumerate(sheets, start=1):
                with archive.open(f"xl/worksheets/sheet{index}.xml", "w") as handle:
                    for chunk in _worksheet_chunks(spec):
                        handle.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = ["SheetSpec", "write_xlsx"]
