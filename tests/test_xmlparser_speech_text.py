import tempfile
import unittest
from pathlib import Path

from crelat.domain.play import PlaySpec
from crelat.io.folger import parse_play


class SpeechTextExtractionTests(unittest.TestCase):
    def test_running_page_header_is_not_part_of_speech(self):
        xml = """\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body><div1><div2><sp who="#Pistol_Test">
    <speaker><w>PISTOL</w></speaker><ab>
      <w>Before</w><c> </c><w>header</w><pc>,</pc><lb/>
      <fw type="header">ACT 1. SC. 4</fw>
      <w>after</w><c> </c><w>header</w><pc>.</pc>
    </ab>
  </sp></div2></div1></body></text>
</TEI>
"""
        with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8") as fh:
            fh.write(xml)
            fh.flush()
            spec = PlaySpec("test", "Test", "Test Play", "tragedy", 1600, Path(fh.name))
            play = parse_play(spec)

        speech = play.scenes[0].speeches[0].text
        self.assertEqual(speech, "Before header,\nafter header.")
        self.assertNotIn("ACT 1. SC. 4", speech)


if __name__ == "__main__":
    unittest.main()
