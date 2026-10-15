from oncora.rag.bm25 import BM25Index
from oncora.rag.text import tokenize

DOCS = [
    "trastuzumab improves survival in HER2 positive breast cancer",
    "aromatase inhibitors versus tamoxifen in postmenopausal women",
    "exercise reduces fatigue during chemotherapy",
]


def test_bm25_ranks_the_matching_document_first():
    idx = BM25Index().fit([tokenize(d) for d in DOCS])
    scores = idx.score(tokenize("letrozole or tamoxifen for postmenopausal women"))
    assert scores.argmax() == 1


def test_bm25_state_round_trip():
    idx = BM25Index().fit([tokenize(d) for d in DOCS])
    clone = BM25Index.from_state(idx.state())
    q = tokenize("fatigue exercise")
    assert (clone.score(q) == idx.score(q)).all()


def test_unknown_terms_score_zero():
    idx = BM25Index().fit([tokenize(d) for d in DOCS])
    assert idx.score(tokenize("zzzz qqqq")).sum() == 0
