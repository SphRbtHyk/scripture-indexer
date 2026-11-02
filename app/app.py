"""POC Streamlit: upload a manuscript PDF, index it with scripture_indexer,
and navigate between the manuscript page images and the associated index.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from pipeline import (
    IndexingResult,
    load_existing_result,
    missal_lines_for_reading,
    rendered_page_image,
    run_indexing,
    transcription_path,
    vulgata_verses,
)

st.set_page_config(
    page_title="Scripture Indexer",
    page_icon=":book:",
    layout="wide",
)


def page_label(source_id: str) -> str:
    """Human-readable label for a source id like ``ms10__page_0003``."""
    match = re.search(r"page[_-]?(\d+)$", source_id)
    return f"Page {int(match.group(1))}" if match else source_id


def readings_for(index: dict, source_id: str) -> list[dict]:
    """Return the index readings attached to one source id."""
    for entry in index.get("pages", []):
        if entry.get("page") == source_id:
            return entry.get("readings", [])
    return []


BOOK_REFERENCE_RE = re.compile(r"^(.*?)\s+\d+:\d+")


def book_of(reference: str) -> str:
    """Extract the biblical book from a reference like ``John 10:1-10``."""
    match = BOOK_REFERENCE_RE.match(reference)
    return match.group(1).strip() if match else reference


def collect_readings(result: IndexingResult) -> list[dict]:
    """Flatten the index into one row per reading, with page positions."""
    positions = {
        source_id: position
        for position, source_id in enumerate(result.source_ids)
    }
    rows: list[dict] = []
    for entry in result.index.get("pages", []):
        source_id = entry.get("page")
        for reading in entry.get("readings", []):
            rows.append(
                {
                    "book": book_of(reading["reference"]),
                    "reference": reading["reference"],
                    "source_id": source_id,
                    "position": positions.get(source_id, 0),
                    "flags": reading_flags(reading),
                }
            )
    return rows


def open_page_button(source_id: str, position: int, key: str) -> None:
    """Render a button that opens the manuscript page of one reading."""
    if st.button("Ouvrir la page", key=key):
        st.session_state.view_mode = "page"
        st.session_state.current_page = position
        st.rerun()


def show_books_view(result: IndexingResult) -> None:
    """Group the readings by biblical book, each occurrence clickable."""
    st.subheader("Vue par livre biblique")
    st.caption(
        "Pour chaque livre, les pages du manuscrit où ses lectures "
        "apparaissent ; « Ouvrir la page » affiche l'image du manuscrit."
    )
    rows = collect_readings(result)
    if not rows:
        st.info("Aucune lecture biblique détectée dans le document.")
        return

    books = sorted({row["book"] for row in rows})
    for book in books:
        book_rows = [row for row in rows if row["book"] == book]
        pages = sorted(
            {page_label(row["source_id"]) for row in book_rows},
            key=lambda label: int(re.search(r"\d+", label).group()) if re.search(r"\d+", label) else 0,
        )
        with st.expander(
            f"{book} — {len(book_rows)} lecture(s) sur "
            f"{len(pages)} page(s)"
        ):
            for row in book_rows:
                col_ref, col_button = st.columns([4, 1])
                with col_ref:
                    st.markdown(
                        f"{row['reference']} · {page_label(row['source_id'])}"
                        f"{row['flags']}"
                    )
                with col_button:
                    open_page_button(
                        row["source_id"],
                        row["position"],
                        key=f"book_{row['source_id']}_{row['reference']}",
                    )


def show_stats_view(result: IndexingResult) -> None:
    """Distribution of the detected readings across biblical books."""
    st.subheader("Répartition des lectures par livre")
    rows = collect_readings(result)
    if not rows:
        st.info("Aucune lecture biblique détectée dans le document.")
        return

    counts = Counter(row["book"] for row in rows)
    frame = (
        pd.DataFrame(
            {"Livre": list(counts), "Lectures": list(counts.values())}
        )
        .sort_values("Lectures", ascending=False)
        .reset_index(drop=True)
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Lectures détectées", len(rows))
    col2.metric("Livres bibliques", len(counts))
    col3.metric(
        "Pages avec lectures",
        len({row["source_id"] for row in rows}),
    )
    st.bar_chart(frame, x="Livre", y="Lectures")
    with st.expander("Données"):
        st.dataframe(frame, width="stretch", hide_index=True)


def reading_flags(reading: dict) -> str:
    """Suffix describing page-continuity of one reading."""
    flags = []
    if reading.get("continues_from_previous_page"):
        flags.append("suite de la page précédente")
    if reading.get("continues_on_next_page"):
        flags.append("continue page suivante")
    return f" · {', '.join(flags)}" if flags else ""


def show_index_view(result: IndexingResult) -> None:
    """Primary post-run view: the page-by-page index, each page clickable."""
    st.subheader("Index des lectures bibliques")
    st.caption(
        "Cliquez sur « Ouvrir la page » pour afficher l'image du manuscrit "
        "correspondante."
    )

    total = sum(
        len(readings_for(result.index, source_id))
        for source_id in result.source_ids
    )
    if total == 0:
        st.info("Aucune lecture biblique détectée dans le document.")

    for position, source_id in enumerate(result.source_ids):
        readings = readings_for(result.index, source_id)
        col_head, col_button = st.columns([4, 1])
        with col_head:
            st.markdown(
                f"**{page_label(source_id)}** — "
                f"{len(readings)} lecture(s) détectée(s)"
            )
        with col_button:
            if st.button("Ouvrir la page", key=f"open_{source_id}"):
                st.session_state.view_mode = "page"
                st.session_state.current_page = position
                st.rerun()

        if readings:
            for reading in readings:
                st.markdown(f"- **{reading['reference']}**{reading_flags(reading)}")
        else:
            st.markdown("- _Aucune lecture détectée sur cette page_")
        st.divider()


def vulgata_running_text(verses: list[dict]) -> str:
    """Vulgate verses as running text.

    The book and chapter are spelled out inline where they change, and
    verse numbers are inline markers, so the text reads continuously.
    """
    parts: list[str] = []
    book: str | None = None
    chapter: int | None = None
    for entry in verses:
        if entry["book"] != book or entry["chapter"] != chapter:
            book, chapter = entry["book"], entry["chapter"]
            parts.append(f"**{book} {chapter}**")
        parts.append(f"{entry['verse']} {entry['text']}")
    return " ".join(parts)


def show_reading(
    result: IndexingResult,
    source_id: str,
    reading: dict,
    transcription_lines: list[str] | None,
) -> None:
    """One detected reading, as a toggle: Vulgate text and its HTR lines."""
    label = f"{reading['reference']}{reading_flags(reading)}"
    verses = vulgata_verses(reading["reference"])
    missal_lines = (
        missal_lines_for_reading(
            result.json_path,
            source_id,
            reading["reference"],
            transcription_lines,
        )
        if transcription_lines is not None
        else None
    )

    with st.expander(label):
        col_vulgata, col_missal = st.columns(2)
        with col_vulgata:
            st.caption("Vulgate sixto-clémentine")
            if verses:
                st.markdown(vulgata_running_text(verses))
            else:
                st.info("Référence absente de la Vulgate.")
        with col_missal:
            st.caption("Texte du missel (HTR)")
            if missal_lines:
                st.code("\n".join(missal_lines), language=None)
            elif transcription_lines is not None:
                st.code("\n".join(transcription_lines), language=None)
            else:
                st.info("Transcription indisponible.")


def show_manuscript_page(result: IndexingResult) -> None:
    """Manuscript page opened from the index, with its readings and nav."""
    if st.button("← Retour à l'index"):
        st.session_state.view_mode = "index"
        st.rerun()

    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    current = st.session_state.get("current_page", 0)
    current = min(current, len(result.source_ids) - 1)
    source_id = result.source_ids[current]

    with col_nav1:
        if st.button("◀ Page précédente", disabled=current == 0):
            st.session_state.current_page = current - 1
            st.rerun()
    with col_nav3:
        if st.button("Page suivante ▶", disabled=current >= len(result.source_ids) - 1):
            st.session_state.current_page = current + 1
            st.rerun()
    with col_nav2:
        choice = st.selectbox(
            "Aller à la page",
            range(len(result.source_ids)),
            index=current,
            format_func=lambda i: page_label(result.source_ids[i]),
        )
        if choice != current:
            st.session_state.current_page = choice
            st.rerun()

    st.caption(f"Fichier : {result.pdf_name} — {source_id}")

    col_image, col_index = st.columns([3, 2])

    with col_image:
        st.subheader("Manuscrit")
        image = rendered_page_image(result.htr_dir, source_id)
        if image is not None:
            st.image(str(image), use_container_width=True)
        else:
            st.info("Image de la page indisponible.")

    with col_index:
        st.subheader("Index des lectures")
        st.caption(
            "Dépliez une lecture pour voir le texte de la Vulgate et la "
            "transcription HTR qui lui correspond."
        )
        readings = readings_for(result.index, source_id)
        transcription = transcription_path(result.htr_dir, source_id)
        transcription_lines = (
            transcription.read_text(encoding="utf-8").splitlines()
            if transcription.is_file()
            else None
        )
        if not readings:
            st.info("Aucune lecture biblique détectée sur cette page.")
        for reading in readings:
            show_reading(result, source_id, reading, transcription_lines)

        if transcription.is_file():
            with st.expander("Transcription HTR complète de la page"):
                st.code(transcription.read_text(encoding="utf-8"), language=None)


def show_downloads(result: IndexingResult) -> None:
    """Download buttons for the pipeline artifacts."""
    st.subheader("Téléchargements")
    st.download_button(
        "Télécharger l'index (YAML)",
        data=result.yaml_path.read_bytes(),
        file_name="index.yaml",
        mime="text/yaml",
    )
    if result.json_path.is_file():
        st.download_button(
            "Télécharger le résultat détaillé (JSON)",
            data=result.json_path.read_bytes(),
            file_name="index.json",
            mime="application/json",
        )
    with st.expander("Aperçu de la sortie YAML"):
        st.code(
            yaml.safe_dump(result.index, allow_unicode=True, sort_keys=False),
            language="yaml",
        )


def show_calibration(result: IndexingResult) -> None:
    """Show the auto-calibrated retrieval parameters, when available."""
    if result.calibration_path is None or not result.calibration_path.is_file():
        return
    calibration = json.loads(result.calibration_path.read_text(encoding="utf-8"))
    status = calibration.get("status")
    parameters = calibration.get("parameters", {})
    with st.expander("Hyperparamètres calibrés automatiquement"):
        if status != "ok":
            st.info(
                f"Calibration abandonnée ({status}); "
                "paramètres de repli utilisés."
            )
        st.json(parameters)


def list_sessions(upload_root: Path) -> list[Path]:
    """Upload sessions that still hold at least one PDF, newest first."""
    sessions = [
        item
        for item in upload_root.glob("session_*")
        if item.is_dir() and any(item.glob("*.pdf"))
    ]
    return sorted(sessions, key=lambda item: item.stat().st_mtime, reverse=True)


def session_label(session: Path) -> str:
    """Sidebar label of one upload session, with its last-modified time."""
    mtime = datetime.fromtimestamp(session.stat().st_mtime)
    return f"{session.name} · {mtime:%d/%m/%Y %H:%M}"


def main() -> None:
    st.title("Scripture Indexer — index biblique d'un manuscrit PDF")
    st.markdown(
        "Chargez un PDF de manuscrit : les pages sont transcrites par "
        "Kraken (HTR), les citations bibliques identifiées par "
        "[Scripture Indexer]("
        "https://github.com/sophrob/missal_indexing), et l'index "
        "page par page s'affiche ; cliquez sur une page de l'index pour "
        "ouvrir l'image du manuscrit correspondante."
    )

    with st.sidebar:
        st.header("Entrée")
        upload_root = Path(__file__).resolve().parent / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        labels = {session_label(item): item for item in list_sessions(upload_root)}
        session_choice = st.selectbox(
            "Session",
            ["Nouvelle session", *labels],
            help=(
                "Reprendre une session existante évite de re-transcrire : "
                "les pages déjà passées dans Kraken sont réutilisées, seul "
                "l'index est recalculé."
            ),
        )
        resumed_dir = labels.get(session_choice)
        pdf_file = None
        resumed_pdf = None
        if resumed_dir is None:
            pdf_file = st.file_uploader("PDF du manuscrit", type=["pdf"])
        else:
            session_pdfs = sorted(
                resumed_dir.glob("*.pdf"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if not session_pdfs:
                st.warning("Cette session ne contient plus aucun PDF.")
            else:
                resumed_pdf = st.selectbox(
                    "PDF de la session",
                    session_pdfs,
                    format_func=lambda item: item.name,
                )
                if (resumed_dir / f"{resumed_pdf.stem}_run").is_dir():
                    st.caption(
                        "Ce PDF a déjà été analysé : le HTR existant sera "
                        "réutilisé si vous relancez l'analyse."
                    )

        st.header("Paramètres")
        limit = st.number_input(
            "Nombre de pages à traiter (0 = toutes)",
            min_value=0,
            value=0,
            step=1,
        )
        pdf_dpi = st.number_input(
            "Résolution de rastérisation (DPI)", min_value=72, value=300, step=10
        )
        device = st.selectbox("Device Kraken", ("auto", "cpu"), index=0)
        auto_hyperparam = st.toggle(
            "Calibration automatique des paramètres de retrieval "
            "(ngram, bruit)",
            value=True,
            help=(
                "Estime ngram_size et noise_level sur ce manuscrit, "
                "sans données annotées (module auto_hyperparam du "
                "package), à la fenêtre fixée ci-dessous. Sinon, le "
                "profil optimisé MS10 est utilisé."
            ),
        )
        with st.expander("Paramètres de l'index"):
            min_reference_words = st.number_input(
                "min_reference_words", min_value=1, value=40, step=1
            )
            missing_verse_tolerance = st.number_input(
                "missing_verse_tolerance", min_value=0, value=3, step=1
            )
            exclude_psalms = st.checkbox("Exclure les Psaumes", value=True)

        run_clicked = st.button(
            "Lancer l'analyse",
            type="primary",
            disabled=pdf_file is None and resumed_pdf is None,
        )

    if pdf_file is None and resumed_pdf is None:
        st.stop()

    if pdf_file is not None:
        if "upload_dir" not in st.session_state:
            st.session_state.upload_dir = Path(
                tempfile.mkdtemp(prefix="session_", dir=upload_root)
            )
        upload_dir = Path(st.session_state.upload_dir)
        pdf_path = upload_dir / pdf_file.name
        if (
            "pdf_path" not in st.session_state
            or st.session_state.pdf_path != pdf_path
            or st.session_state.get("pdf_bytes_key") != pdf_file.size
            or not pdf_path.is_file()
        ):
            pdf_path.write_bytes(pdf_file.getvalue())
            st.session_state.pdf_path = pdf_path
            st.session_state.pdf_bytes_key = pdf_file.size
            st.session_state.pop("indexing_result", None)
    else:
        upload_dir = resumed_dir
        pdf_path = resumed_pdf
        if st.session_state.get("pdf_path") != pdf_path:
            st.session_state.pop("indexing_result", None)
            st.session_state.pop("pdf_bytes_key", None)
        st.session_state.upload_dir = upload_dir
        st.session_state.pdf_path = pdf_path

    work_dir = upload_dir / f"{pdf_path.stem}_run"

    if run_clicked:
        try:
            with st.status("Analyse en cours…", expanded=True) as status:
                st.write("Transcription HTR (Kraken) puis indexation…")
                result = run_indexing(
                    pdf_path,
                    work_dir,
                    limit=int(limit),
                    pdf_dpi=int(pdf_dpi),
                    device=device,
                    min_reference_words=int(min_reference_words),
                    missing_verse_tolerance=int(missing_verse_tolerance),
                    exclude_psalms=bool(exclude_psalms),
                    auto_hyperparam=bool(auto_hyperparam),
                )
                st.write(f"{len(result.source_ids)} page(s) traitée(s).")
                status.update(
                    label="Analyse terminée", state="complete", expanded=False
                )
            st.session_state.indexing_result = result
            st.session_state.current_page = 0
            st.session_state.view_mode = "index"
        except Exception as exc:  # noqa: BLE001 - surface any pipeline error
            st.error(f"Échec de l'analyse : {exc}")
    elif (work_dir / "index.yaml").is_file():
        st.caption(
            "Ce PDF a déjà été analysé dans cette session ; vous pouvez "
            "recharger directement son dernier index, ou relancer "
            "l'analyse (le HTR existant sera réutilisé)."
        )
        if st.button("Recharger le dernier index"):
            result = load_existing_result(work_dir, pdf_path.name)
            if result is None:
                st.error("Impossible de recharger les résultats de ce run.")
            else:
                st.session_state.indexing_result = result
                st.session_state.current_page = 0
                st.session_state.view_mode = "index"
                st.rerun()

    result = st.session_state.get("indexing_result")
    if result is None:
        st.info(
            "Chargez un PDF ou reprenez une session existante, puis "
            "cliquez sur « Lancer l'analyse »."
        )
        st.stop()

    show_calibration(result)

    tab_index, tab_books, tab_stats, tab_downloads = st.tabs(
        ["Index", "Par livre", "Statistiques", "Téléchargements"]
    )
    with tab_index:
        if st.session_state.get("view_mode", "index") == "page":
            show_manuscript_page(result)
        else:
            show_index_view(result)
    with tab_books:
        show_books_view(result)
    with tab_stats:
        show_stats_view(result)
    with tab_downloads:
        show_downloads(result)


main()
