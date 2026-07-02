import numpy as np

from crelat.domain.play import Play, Scene
from crelat.domain.speech import Speech
from crelat.features.interactions import build_speech_interactions


def test_consecutive_interactions_skip_same_speaker():
    speeches = [
        Speech("x:1:1", "x", "A1.S1", 1, "a", "one"),
        Speech("x:1:2", "x", "A1.S1", 2, "a", "two"),
        Speech("x:1:3", "x", "A1.S1", 3, "b", "three"),
    ]
    play = Play("x", "Example", ("a", "b"), [Scene("A1.S1", 1, 1, speeches)])
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = build_speech_interactions(play, embeddings, "test")
    assert len(result) == 1
    assert result.iloc[0].speech1_id == "x:1:2"
    assert result.iloc[0].cosine_similarity == 0.0
