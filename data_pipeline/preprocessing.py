"""Convert raw FLUXNET2015-format site data into the canonical schema.

The training pipeline expects every site to have the same columns:

    DateTime, NEE, H, Tau, LE, RH, VPD, Ustar, Ta, Tsoil1, Rg

This module reads a site's raw XLSX/parquet (130+ columns, FLUXNET naming) and
produces a parquet with exactly that schema. Sites where SW components are
unavailable can pass a single SWin column instead.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FLUXNET_COLUMN_MAPPING,
    FLUXNET_TIMESTAMP_FORMAT,
    NAN_SENTINEL,
    RADIATION_COMPONENT_COLUMNS,
)

log = logging.getLogger("wienernet.data_pipeline.preprocessing")


class SitePreprocessor:
    """Stateless converter: raw FLUXNET site dataframe -> canonical schema.

    The class wraps the notebook's `preprocess_site_data` function plus the
    manually-edited overrides for sites that don't have SW components.

    Args:
        column_mapping: canonical -> ordered list of candidate FLUXNET columns.
            Defaults to `FLUXNET_COLUMN_MAPPING`.
        replace_sentinels: if True (default), replace FLUXNET's -9999 with NaN
            before any processing.
    """

    def __init__(
        self,
        column_mapping: dict[str, list[str]] | None = None,
        *,
        replace_sentinels: bool = True,
    ) -> None:
        self.column_mapping = column_mapping or dict(FLUXNET_COLUMN_MAPPING)
        self.replace_sentinels = replace_sentinels

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def process(
        self,
        df: pd.DataFrame,
        *,
        radiation_source: str = "components",
        swin_column: str = "SWin",
        rename_columns: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Apply the full conversion to an in-memory dataframe.

        Args:
            df: raw site dataframe (FLUXNET columns).
            radiation_source: how to obtain Rg.
                - "components": Rg = (SW_IN - SW_OUT) + (LW_IN - LW_OUT)
                - "swin": Rg = `swin_column` directly.
            swin_column: column name for the SWin variant. Ignored if
                radiation_source != "swin".
            rename_columns: optional pre-processing renames, e.g.
                {"ustar": "Ustar"} for sites with non-standard capitalisation.

        Returns:
            A new dataframe with the canonical columns. Order:
            DateTime, NEE, H, Tau, LE, RH, VPD, Ustar, Ta, Tsoil1, Rg
            (or whatever subset of canonical columns was findable).
        """
        out = df.copy()
        if rename_columns:
            out = out.rename(columns=rename_columns)
        if self.replace_sentinels:
            out = out.replace(NAN_SENTINEL, np.nan)

        canonical: dict[str, pd.Series] = {}
        canonical["DateTime"] = self._extract_datetime(out)
        canonical.update(self._map_fluxnet_columns(out))
        canonical["Tsoil1"] = self._derive_tsoil(out)
        canonical["Rg"] = self._derive_rg(out, source=radiation_source, swin_column=swin_column)

        return pd.DataFrame(canonical)

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        radiation_source: str = "components",
        swin_column: str = "SWin",
        rename_columns: dict[str, str] | None = None,
    ) -> Path:
        """Convenience wrapper: read parquet, process, write parquet.

        Returns the output path. Parent directory is created automatically.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Reading %s", input_path)
        df = pd.read_parquet(input_path)
        out = self.process(
            df,
            radiation_source=radiation_source,
            swin_column=swin_column,
            rename_columns=rename_columns,
        )
        out.to_parquet(output_path)
        log.info("Wrote %s (%d rows, %d cols)", output_path, len(out), len(out.columns))
        return output_path

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _extract_datetime(self, df: pd.DataFrame) -> pd.Series:
        """Build a DateTime column from FLUXNET timestamp columns."""
        if "DateTime" in df.columns:
            return pd.to_datetime(df["DateTime"])
        for ts_col in ("TIMESTAMP_START", "TIMESTAMP_END"):
            if ts_col in df.columns:
                return pd.to_datetime(df[ts_col].astype(str), format=FLUXNET_TIMESTAMP_FORMAT)
        raise ValueError(
            "Cannot build DateTime column: none of "
            "['DateTime', 'TIMESTAMP_START', 'TIMESTAMP_END'] found"
        )

    def _map_fluxnet_columns(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Return canonical -> series mapping by taking the first matching candidate."""
        out: dict[str, pd.Series] = {}
        for canonical, candidates in self.column_mapping.items():
            for cand in candidates:
                if cand in df.columns:
                    out[canonical] = df[cand]
                    break
            else:
                log.warning(
                    "Canonical column %r has no available candidate (looked for %s)",
                    canonical, candidates,
                )
        return out

    @staticmethod
    def _derive_tsoil(df: pd.DataFrame) -> pd.Series:
        """Average the shallowest TS_<pos>_<depth>_<channel> column per position.

        Returns NaN series if no TS_* columns exist (e.g. site reports
        soil temp under a different name).
        """
        ts_cols = [c for c in df.columns if c.startswith("TS_")]
        if not ts_cols:
            return pd.Series(np.nan, index=df.index, name="Tsoil1")

        parsed: list[tuple[str, int, int, int]] = []
        for col in ts_cols:
            m = re.match(r"TS_(\d+)_(\d+)_(\d+)", col)
            if m:
                position, depth, channel = map(int, m.groups())
                parsed.append((col, position, depth, channel))

        # Per position, keep the column at the minimum depth
        shallowest_per_position: dict[int, tuple[str, int]] = {}
        for col, position, depth, _channel in parsed:
            current = shallowest_per_position.get(position)
            if current is None or depth < current[1]:
                shallowest_per_position[position] = (col, depth)

        shallowest_cols = [col for col, _ in shallowest_per_position.values()]
        return df[shallowest_cols].mean(axis=1)

    @staticmethod
    def _derive_rg(
        df: pd.DataFrame,
        *,
        source: str = "components",
        swin_column: str = "SWin",
    ) -> pd.Series:
        """Compute Rg either from SW/LW components or directly from a SWin column."""
        if source == "swin":
            if swin_column not in df.columns:
                raise ValueError(f"Column {swin_column!r} not found (radiation_source='swin')")
            return df[swin_column]
        if source == "components":
            missing = [c for c in RADIATION_COMPONENT_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"radiation_source='components' requires {RADIATION_COMPONENT_COLUMNS}; "
                    f"missing: {missing}"
                )
            return (
                (df["SW_IN"] - df["SW_OUT"])
                + (df["LW_IN"] - df["LW_OUT"])
            )
        raise ValueError(f"Unknown radiation_source: {source!r}")
