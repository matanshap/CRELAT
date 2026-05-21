import xml.etree.ElementTree as ET
import os

xml_path = 'Data/XML/folger_corpus/Tmp.xml'
root = ET.parse(xml_path).getroot()
namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

def get_text(elem):
    res = ""
    for child in elem:
        if child.tag.endswith('speaker') or child.tag.endswith('stage') or child.tag.endswith('sound'):
            continue
        if child.tag.endswith('lb'):
            res += '\n'
        if child.text:
            res += child.text
        res += get_text(child)
        if child.tail:
            res += child.tail
    return res

scenes = root.findall('.//tei:div2', namespaces)
if scenes:
    for speech in scenes[0].findall('tei:sp', namespaces)[:5]:
        who = speech.get('who')
        speaker = who.split()[0][1:] if who is not None else '[UNKNOWN]'
        text = get_text(speech).strip()
        print(f"[{speaker}]:\n{text}\n" + "-"*40)

    # Let's also find a speech that might have been empty previously
    # We'll print one speech from scene 2 to check regular text
    for speech in scenes[1].findall('tei:sp', namespaces)[:2]:
        who = speech.get('who')
        speaker = who.split()[0][1:] if who is not None else '[UNKNOWN]'
        text = get_text(speech).strip()
        print(f"[{speaker}]:\n{text}\n" + "-"*40)
