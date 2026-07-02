"""Pure Folger TEI parsing with no model inference or plotting."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

from crelat.domain.play import Play, PlaySpec, Scene
from crelat.domain.speech import Speech

TEI = "http://www.tei-c.org/ns/1.0"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"tei": TEI, "xml": XML}
SKIP_TEXT_TAGS = {"speaker", "stage", "sound", "fw", "head"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_speech_text(element: ET.Element) -> str:
    """Extract spoken text while excluding headings and stage apparatus."""
    parts: list[str] = []

    def visit(node: ET.Element) -> None:
        tag = _local_name(node.tag)
        if tag in SKIP_TEXT_TAGS:
            return
        if tag == "lb":
            parts.append("\n")
        elif node.text and tag in {"w", "c", "pc"}:
            parts.append(node.text)
        for child in node:
            visit(child)

    visit(element)
    return "".join(parts).strip()


def _speaker_id(raw: Optional[str], folger_code: str) -> str:
    if not raw:
        return "[UNKNOWN]"
    first = raw.split()[0].lstrip("#")
    if not first:
        return "[UNKNOWN]"
    base = first.rsplit("_", 1)[0] if "_" in first else first
    return f"{base}_{folger_code}"


def _title(root: ET.Element, fallback: str) -> str:
    candidates = root.findall(".//tei:titleStmt/tei:title", NS)
    for candidate in candidates:
        value = "".join(candidate.itertext()).strip()
        if value:
            return value
    return fallback


def parse_play(spec: PlaySpec, xml_path: Optional[Union[str, Path]] = None) -> Play:
    """Parse one Folger TEI play into stable domain records."""
    path = Path(xml_path) if xml_path is not None else spec.xml
    root = ET.parse(path).getroot()
    people = root.findall(".//tei:person", NS) + root.findall(".//tei:personGrp", NS)
    character_ids = []
    for person in people:
        character_id = _speaker_id(person.get(f"{{{XML}}}id"), spec.folger_code)
        if character_id != "[UNKNOWN]" and character_id not in character_ids:
            character_ids.append(character_id)

    scenes: list[Scene] = []
    acts = root.findall(".//tei:div1", NS)
    if acts:
        scene_elements = [
            (act_index, scene_index, scene)
            for act_index, act in enumerate(acts, start=1)
            for scene_index, scene in enumerate(act.findall(".//tei:div2", NS), start=1)
        ]
    else:
        scene_elements = [
            (None, scene_index, scene)
            for scene_index, scene in enumerate(root.findall(".//tei:div2", NS), start=1)
        ]

    for global_index, (act_number, scene_number, element) in enumerate(scene_elements, start=1):
        scene_id = (
            f"A{act_number}.S{scene_number}" if act_number is not None else f"S{scene_number}"
        )
        speeches = []
        for position, speech_element in enumerate(element.findall("tei:sp", NS), start=1):
            speaker = _speaker_id(speech_element.get("who"), spec.folger_code)
            if speaker not in character_ids:
                character_ids.append(speaker)
            speeches.append(
                Speech(
                    id=f"{spec.id}:{global_index}:{position}",
                    play_id=spec.id,
                    scene_id=scene_id,
                    position=position,
                    speaker_id=speaker,
                    text=extract_speech_text(speech_element),
                )
            )
        scenes.append(Scene(scene_id, act_number, scene_number, speeches))

    return Play(
        id=spec.id,
        title=_title(root, spec.title),
        characters=tuple(character_ids),
        scenes=scenes,
    )
