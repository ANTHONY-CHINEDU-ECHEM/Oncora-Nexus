"""Reproducible acquisition of every public source used by Oncora Nexus.

Each source is declared once in SOURCES with its origin, licence and the local
file it produces. Downloads are idempotent: an existing file is never fetched
twice unless force=True. The METABRIC expression matrix is 690 MB, so it is
streamed and filtered to the PAM50 panel on the fly instead of being stored.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from oncora.config import settings

log = logging.getLogger(__name__)

CBIO = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
RAW_GH = "https://raw.githubusercontent.com"

PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1",
    "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
]


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    target: str
    description: str
    licence: str
    citation: str
    approx_mb: float


SOURCES: list[Source] = [
    Source(
        key="pubmed_rct_train",
        url=f"{RAW_GH}/Franck-Dernoncourt/pubmed-rct/master/PubMed_200k_RCT/train.7z",
        target="pubmed_rct/train.7z",
        description="PubMed 200k RCT training split: about 190,000 randomised controlled trial abstracts with sentence role labels and PMIDs.",
        licence="Abstract text from PubMed (NLM terms of use); dataset released for research by Dernoncourt and Lee.",
        citation="Dernoncourt F, Lee JY. PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. IJCNLP 2017.",
        approx_mb=67.0,
    ),
    Source(
        key="pubmed_rct_dev",
        url=f"{RAW_GH}/Franck-Dernoncourt/pubmed-rct/master/PubMed_200k_RCT/dev.txt",
        target="pubmed_rct/dev.txt",
        description="PubMed 200k RCT development split (2,500 abstracts).",
        licence="As above.",
        citation="Dernoncourt and Lee, 2017.",
        approx_mb=4.7,
    ),
    Source(
        key="pubmed_rct_test",
        url=f"{RAW_GH}/Franck-Dernoncourt/pubmed-rct/master/PubMed_200k_RCT/test.txt",
        target="pubmed_rct/test.txt",
        description="PubMed 200k RCT test split (2,500 abstracts).",
        licence="As above.",
        citation="Dernoncourt and Lee, 2017.",
        approx_mb=4.7,
    ),
    Source(
        key="nli4ct",
        url=f"{RAW_GH}/ai-systems/nli4ct/main/Complete_dataset.zip",
        target="nli4ct/Complete_dataset.zip",
        description="NLI4CT: 999 breast cancer clinical trial reports from ClinicalTrials.gov (eligibility, interventions, results, adverse events) plus 2,400 expert annotated entailment statements.",
        licence="Released for SemEval 2023 Task 7; trial text originates from ClinicalTrials.gov (public domain, US NLM).",
        citation="Jullien M, Valentino M, Frost H, et al. SemEval-2023 Task 7: Multi-Evidence Natural Language Inference for Clinical Trial Data. SemEval 2023.",
        approx_mb=2.5,
    ),
    Source(
        key="metabric_patient",
        url=f"{CBIO}/brca_metabric/data_clinical_patient.txt",
        target="metabric/data_clinical_patient.txt",
        description="METABRIC patient level clinical data for 2,509 breast cancer patients (survival, relapse, treatment, receptor status, PAM50 subtype).",
        licence="cBioPortal datahub, ODbL; original study Curtis et al. Nature 2012, Pereira et al. Nat Commun 2016.",
        citation="Pereira B, Chin SF, Rueda OM, et al. The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. Nat Commun 2016.",
        approx_mb=0.4,
    ),
    Source(
        key="metabric_sample",
        url=f"{CBIO}/brca_metabric/data_clinical_sample.txt",
        target="metabric/data_clinical_sample.txt",
        description="METABRIC sample level data (tumour size, grade, stage, histology, ER/PR/HER2, TMB).",
        licence="cBioPortal datahub, ODbL.",
        citation="Pereira et al., 2016.",
        approx_mb=0.3,
    ),
    Source(
        key="metabric_mutations",
        url=f"{CBIO}/brca_metabric/data_mutations.txt",
        target="metabric/data_mutations.txt",
        description="METABRIC targeted sequencing mutations across 173 driver genes.",
        licence="cBioPortal datahub, ODbL.",
        citation="Pereira et al., 2016.",
        approx_mb=4.0,
    ),
    Source(
        key="tcga_brca_patient",
        url=f"{CBIO}/brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        target="tcga_brca/data_clinical_patient.txt",
        description="TCGA BRCA PanCancer Atlas patient clinical data (1,084 patients) with curated survival endpoints.",
        licence="cBioPortal datahub, ODbL; TCGA data policies.",
        citation="Liu J, Lichtenberg T, Hoadley KA, et al. An Integrated TCGA Pan-Cancer Clinical Data Resource. Cell 2018.",
        approx_mb=0.35,
    ),
    Source(
        key="tcga_brca_sample",
        url=f"{CBIO}/brca_tcga_pan_can_atlas_2018/data_clinical_sample.txt",
        target="tcga_brca/data_clinical_sample.txt",
        description="TCGA BRCA PanCancer Atlas sample data including PAM50 subtype calls.",
        licence="cBioPortal datahub, ODbL.",
        citation="Liu et al., 2018.",
        approx_mb=0.3,
    ),
]

EXPRESSION_SOURCE = Source(
    key="metabric_expression_pam50",
    url=f"{CBIO}/brca_metabric/data_mrna_illumina_microarray.txt",
    target="metabric/expression_pam50.tsv",
    description="METABRIC Illumina HT 12 expression (log2 intensity), streamed and reduced to the 50 PAM50 genes.",
    licence="cBioPortal datahub, ODbL.",
    citation="Curtis C, Shah SP, Chin SF, et al. The genomic and transcriptomic architecture of 2,000 breast tumours. Nature 2012.",
    approx_mb=690.0,
)


def _download(url: str, dest: Path, retries: int = 8) -> None:
    """Resumable download. Interrupted transfers continue with HTTP Range
    requests, and the final size is checked against Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    expected = 0
    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 206:
                    total = resp.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                    expected = int(total) if total.isdigit() else expected
                    mode = "ab"
                else:
                    expected = int(resp.headers.get("Content-Length") or 0)
                    mode = "wb"
                with open(tmp, mode) as fh:
                    shutil.copyfileobj(resp, fh, length=1 << 20)
        except Exception as exc:  # pragma: no cover, network dependent
            log.warning("Attempt %s for %s interrupted: %s", attempt, url, exc)
        got = tmp.stat().st_size if tmp.exists() else 0
        if got and (not expected or got == expected):
            tmp.replace(dest)
            return
        log.info("Resuming %s at %s of %s bytes", dest.name, got, expected)
        time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Could not download {url}")


def stream_pam50_expression(dest: Path, genes: list[str] = PAM50_GENES) -> int:
    """Stream the 690 MB expression matrix and keep only PAM50 rows."""
    wanted = set(genes) | {"ORC6L"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with urllib.request.urlopen(EXPRESSION_SOURCE.url, timeout=300) as resp:
        reader = io.TextIOWrapper(resp, encoding="utf-8", newline="")
        with open(dest, "w", encoding="utf-8") as out:
            header = reader.readline()
            out.write(header)
            for line in reader:
                symbol = line.split("\t", 1)[0]
                if symbol in wanted:
                    out.write(line)
                    kept += 1
    return kept


def extract_archives(raw: Path) -> None:
    seven = raw / "pubmed_rct" / "train.7z"
    if seven.exists() and not (raw / "pubmed_rct" / "train.txt").exists():
        import py7zr

        with py7zr.SevenZipFile(seven, mode="r") as z:
            z.extractall(path=raw / "pubmed_rct")
    nli_zip = raw / "nli4ct" / "Complete_dataset.zip"
    if nli_zip.exists() and not (raw / "nli4ct" / "Complete_dataset").exists():
        with zipfile.ZipFile(nli_zip) as z:
            members = [m for m in z.namelist() if not m.startswith("__MACOSX") and not m.endswith(".DS_Store")]
            z.extractall(raw / "nli4ct", members=members)


def acquire_all(force: bool = False, with_expression: bool = True) -> dict:
    paths = settings.paths.ensure()
    manifest = {"retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": []}
    for src in SOURCES:
        dest = paths.raw / src.target
        if force or not dest.exists():
            log.info("Downloading %s (about %.1f MB)", src.key, src.approx_mb)
            _download(src.url, dest)
        entry = asdict(src)
        entry["bytes"] = dest.stat().st_size
        manifest["sources"].append(entry)
    if with_expression:
        dest = paths.raw / EXPRESSION_SOURCE.target
        if force or not dest.exists():
            log.info("Streaming METABRIC expression and filtering to PAM50 genes")
            kept = stream_pam50_expression(dest)
            log.info("Kept %s gene rows", kept)
        entry = asdict(EXPRESSION_SOURCE)
        entry["bytes"] = dest.stat().st_size
        manifest["sources"].append(entry)
    extract_archives(paths.raw)
    (paths.raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


__all__ = ["SOURCES", "PAM50_GENES", "acquire_all", "stream_pam50_expression", "extract_archives"]
