import unittest

import pandas as pd

from crelat.features.speech_length import add_tfidf_similarity


class TfidfSimilarityTests(unittest.TestCase):
    def test_identical_text_scores_above_unrelated_text(self):
        rows = pd.DataFrame(
            [
                {
                    "text1": "I'll follow Caesar",
                    "text2": "I’ll follow Caesar",
                    "cosine_similarity": 0.4,
                },
                {
                    "text1": "I'll follow Caesar",
                    "text2": "winter ships depart",
                    "cosine_similarity": 0.8,
                },
            ]
        )

        result = add_tfidf_similarity(rows)

        self.assertAlmostEqual(result.loc[0, "tfidf_cosine_similarity"], 1.0)
        self.assertLess(
            result.loc[1, "tfidf_cosine_similarity"],
            result.loc[0, "tfidf_cosine_similarity"],
        )
        self.assertNotIn("’", result.loc[0, "text2"])


if __name__ == "__main__":
    unittest.main()
