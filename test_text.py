from oncora.rag.text import normalise, numbers, split_sentences, stem, tokenize


def test_uk_spelling_and_synonyms_fold_to_index_terms():
    toks = tokenize("Oestrogen receptor positive tumour treated with T-DM1")
    assert "estrogen" in toks and "tumor" in toks
    assert "trastuzumab" in toks and "emtansine" in toks


def test_lay_language_maps_to_clinical_terms():
    assert "lymphedema" in normalise("arm swelling after surgery")
    assert "breast conserving surgery" in normalise("she had a lumpectomy")


def test_stopwords_and_short_tokens_removed_but_clinical_short_tokens_kept():
    toks = tokenize("the ER and PR status of the patients")
    assert "the" not in toks and "patient" not in toks
    assert "er" in toks and "pr" in toks


def test_stemmer_is_conservative():
    assert stem("flashes") == "flash"
    assert stem("therapies") == "therapy"
    assert stem("her2") == "her2"


def test_numbers_canonicalise_percent():
    assert "18%" in numbers("a reduction of 18 percent")
    assert "18%" in numbers("reduced by 18%")
    assert "0.82" in numbers("hazard ratio 0.82")


def test_sentence_split_keeps_citations_attached():
    parts = split_sentences("First claim [S1]. Second claim [S2].")
    assert parts == ["First claim [S1].", "Second claim [S2]."]
