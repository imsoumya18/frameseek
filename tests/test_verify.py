"""Tests for evidence checks, on real strings from the test video."""

from frameseek.models.vision import Timing
from frameseek.verify import check_evidence, normalise, quote_score

# Whisper's transcript of 77-101 s (real, misspellings included)
CHUNK = ("আমি তো প্রেমিক তালে জুরেছি হাত খো পালে ভিরেরে রো শমলে ঠাকা তো মার দেখা পাই খুশ্মি জাহাজে বতটি "
         "বাহাজে এশেছে আজি দুকা মাজে ধুলো চি এরো সাহাজে দুশো বুজাশে দুকা মাজে")


def test_normalise_keeps_bengali_vowel_signs() -> None:
    assert normalise("তোমাকে, দেখে!") == "তোমাকেদেখে"     # re's \\w would give "তমকদখ"
    assert normalise("EMPEROR vs Sarat") == "emperorvssarat"


def test_quote_scores_on_the_real_transcript() -> None:
    assert quote_score("আমি তো প্রেমিক", CHUNK) == 1.0
    assert quote_score("খুশির জাহাজে", CHUNK) > 0.7            # real line through Whisper's misspelling
    assert quote_score("শিউলি ফুলের মালা", CHUNK) < 0.6        # made up
    assert quote_score("", CHUNK) == 0.0 and quote_score("x", "") == 0.0


class Judge:
    def __init__(self, supports):
        self.supports, self.calls = supports, []

    def structured(self, image, prompt, schema, max_side=None, num_predict=400):
        self.calls.append((image, prompt))
        return {"reason": "r", "supports": self.supports}, Timing(0, 1.0, 180, 1.0, 20, 2.0)


def test_check_order_and_outcomes() -> None:
    j = Judge(True)
    assert check_evidence(j, "q", "", CHUNK, "speech", 80, 0.7).status == "unverified"   # visual claim
    assert check_evidence(j, "q", "আমি তো", "", "on-screen text", 80, 0.7).status == "unverified"
    assert check_evidence(j, "q", "শিউলি ফুলের মালা", CHUNK, "speech", 80, 0.7).status == "failed"
    assert j.calls == []                                           # none of those needed the model
    c = check_evidence(j, "q", "আমি তো প্রেমিক", CHUNK, "speech", 80, 0.7)
    assert c.status == "verified" and j.calls[0][0] is None       # check 2 is text only
    assert check_evidence(Judge(False), "q", "আমি তো প্রেমিক", CHUNK, "speech", 80, 0.7).status == "failed"
