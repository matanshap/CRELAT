import unittest

import numpy as np
import pandas as pd

from crelat.analysis.statistics import benjamini_hochberg
from crelat.features.speech_length import add_length_metrics, add_word_count_metrics, speech_pair_export


class _WhitespaceTokenizer:
    model_max_length = 512

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": text.split()}


class SpeechLengthSimilarityTests(unittest.TestCase):
    def test_pair_length_metrics(self):
        rows = pd.DataFrame(
            [
                {
                    "text1": "one two three",
                    "text2": "one two three four five",
                }
            ]
        )
        result = add_length_metrics(rows, _WhitespaceTokenizer()).iloc[0]

        self.assertEqual(result.length_difference, 2)
        self.assertEqual(result.length_sum, 8)
        self.assertEqual(result.length_min, 3)
        self.assertEqual(result.length_max, 5)
        self.assertAlmostEqual(result.length_ratio, 0.6)

    def test_word_count_metrics_match_requested_example(self):
        rows = pd.DataFrame(
            [
                {
                    "text1": "one two three four five six seven eight nine ten",
                    "text2": "one",
                }
            ]
        )
        result = add_word_count_metrics(rows).iloc[0]

        self.assertEqual(result.word_count1, 10)
        self.assertEqual(result.word_count2, 1)
        self.assertEqual(result.word_count_difference_nominal, 9)
        self.assertAlmostEqual(result.word_count_change_percent, -90.0)

    def test_word_count_percent_handles_empty_first_text(self):
        rows = pd.DataFrame([{"text1": "", "text2": "one two"}])
        result = add_word_count_metrics(rows).iloc[0]

        self.assertEqual(result.word_count1, 0)
        self.assertEqual(result.word_count2, 2)
        self.assertTrue(np.isnan(result.word_count_change_percent))

    def test_speech_pair_export_uses_requested_columns(self):
        rows = pd.DataFrame(
            [
                {
                    "play_title": "Antony and Cleopatra",
                    "scene_id": "A1.S1",
                    "speaker1": "Antony_Ant",
                    "speaker2": "Cleopatra_Ant",
                    "text1": "one two",
                    "text2": "three",
                    "cosine_similarity": 0.5,
                    "tfidf_cosine_similarity": 0.25,
                    "word_count_change_percent": -50.0,
                    "word_count_difference_nominal": 1,
                    "model_id": "bert",
                }
            ]
        )
        result = speech_pair_export(rows)

        self.assertEqual(
            list(result.columns),
            [
                "play_name",
                "act_scene",
                "speaker1",
                "speaker2",
                "text1",
                "text2",
                "cosine_similarity",
                "tfidf_cosine_similarity",
                "word_count_change_percent",
                "word_count_difference_nominal",
            ],
        )
        self.assertNotIn("model_id", result.columns)

    def test_benjamini_hochberg_is_monotone_by_rank(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
        expected = np.array([0.02, 0.04, 0.04, 0.008])
        np.testing.assert_allclose(adjusted, expected)


if __name__ == "__main__":
    unittest.main()
