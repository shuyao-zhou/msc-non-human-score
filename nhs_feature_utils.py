"""
Reusable feature-building utilities for the LANL authentication Non-Human Score project.

Static feature construction is label-free: account naming proxies, anchors, red-team
events and downstream NHS scores are not used here.

Version note
------------
Fisher-type periodicity strength v2 clips the approximate log10(p) to [-300, 0],
so the resulting strength is always in [0, 300]. This fixes the older implementation,
which could return negative values when the approximation exceeded p=1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd


FEATURE_UTILS_VERSION = "2026-08-static-v2"
FISHER_IMPLEMENTATION_VERSION = "fisher_type_v2_clip_logp_minus300_0"


AUTH_COLUMNS = {
    "t": "BIGINT",
    "src_user": "VARCHAR",
    "dst_user": "VARCHAR",
    "src_comp": "VARCHAR",
    "dst_comp": "VARCHAR",
    "auth": "VARCHAR",
    "logon": "VARCHAR",
    "orient": "VARCHAR",
    "outcome": "VARCHAR",
}


def write_parquet(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    path: str | Path,
    temp_name: str = "_tmp_df",
) -> None:
    """Overwrite a parquet file using DuckDB rather than pandas/pyarrow."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        path.unlink()

    con.register(temp_name, df)
    try:
        con.execute(f"COPY {temp_name} TO '{path}' (FORMAT PARQUET)")
    finally:
        try:
            con.unregister(temp_name)
        except Exception:
            pass


def ensure_auth_parquet(
    con: duckdb.DuckDBPyConnection,
    auth_path: str | Path,
    auth_parquet: str | Path,
    force: bool = False,
) -> None:
    """
    Build auth_main.parquet from LANL Cyber1 auth.txt.gz if needed.

    Fixed event rule:
    - keep outcome == 'Success'
    - retain t, src_user, dst_comp, orient, logon
    """
    auth_path = Path(auth_path)
    auth_parquet = Path(auth_parquet)

    if force and auth_parquet.exists():
        auth_parquet.unlink()

    if auth_parquet.exists():
        print(f"Using existing {auth_parquet}")
        return

    if not auth_path.exists():
        raise FileNotFoundError(
            f"Neither {auth_parquet!s} nor raw auth file {auth_path!s} exists."
        )

    cols_sql = ",".join([f"'{k}':'{v}'" for k, v in AUTH_COLUMNS.items()])

    print("Building", auth_parquet, "from", auth_path)
    con.execute(f"""
    COPY (
        SELECT t, src_user, dst_comp, orient, logon
        FROM read_csv(
             '{auth_path}',
             header=false,
             ignore_errors=true,
             columns={{ {cols_sql} }}
        )
        WHERE outcome = 'Success'
    ) TO '{auth_parquet}' (FORMAT PARQUET)
    """)
    print("Built", auth_parquet)


def initialise_duckdb(
    auth_parquet: str | Path,
    auth_path: str | Path | None = None,
    window_days: int = 14,
    force_auth_rebuild: bool = False,
) -> tuple[duckdb.DuckDBPyConnection, int, int]:
    """
    Create a DuckDB connection and recreate session-local views `a` and `e`.

    View `a`
        Successful authentication events, all orientations.

    View `e`
        Main event stream: successful LogOn events, src_user as account `u`,
        no event deduplication, complete windows only.
    """
    con = duckdb.connect()
    auth_parquet = Path(auth_parquet)

    if auth_path is not None:
        ensure_auth_parquet(
            con,
            auth_path,
            auth_parquet,
            force=force_auth_rebuild,
        )
    elif not auth_parquet.exists():
        raise FileNotFoundError(
            f"Missing {auth_parquet!s}; pass auth_path to build it."
        )

    window = window_days * 86400

    con.execute(
        f"CREATE OR REPLACE VIEW a AS SELECT * FROM '{auth_parquet}'"
    )

    max_t = con.execute("SELECT MAX(t) FROM a").fetchone()[0]
    n_win = max(1, int(max_t // window))

    con.execute(f"""
    CREATE OR REPLACE VIEW e AS
    SELECT
        src_user AS u,
        t,
        dst_comp,
        CAST(t // {window} AS INT) AS win
    FROM a
    WHERE orient = 'LogOn'
      AND t < {n_win * window}
    """)

    n_success = con.execute("SELECT COUNT(*) FROM a").fetchone()[0]
    print(f"Success events: {n_success:,}")
    print(f"Complete {window_days}-day windows: {n_win}")
    print(f"Discarded tail: {max(0, max_t - n_win * window)} seconds")
    print("DuckDB views restored: a, e")

    return con, int(max_t), int(n_win)


def compute_quiet_hours(
    con: duckdb.DuckDBPyConnection,
    quiet_hours_path: str | Path,
    n_hours: int = 8,
    force: bool = False,
) -> list[int]:
    """
    Define enterprise-wide quiet hours from aggregate activity in `e`.

    This function remains available for reproducibility, but the cleaned 01 notebook
    should normally LOAD the frozen quiet_hours.json produced by notebook 00 rather
    than redefine it.
    """
    quiet_hours_path = Path(quiet_hours_path)

    if quiet_hours_path.exists() and not force:
        quiet_hours = json.loads(quiet_hours_path.read_text())
        print("Loaded quiet hours:", quiet_hours)
        return [int(x) for x in quiet_hours]

    hourly = con.execute("""
        SELECT
            CAST((t % 86400) // 3600 AS INT) AS hr,
            COUNT(*) AS n
        FROM e
        GROUP BY hr
        ORDER BY n, hr
    """).df()

    quiet_hours = sorted(
        hourly.head(n_hours)["hr"].astype(int).tolist()
    )
    quiet_hours_path.write_text(json.dumps(quiet_hours, indent=2))

    print("Computed enterprise-wide quiet hours:", quiet_hours)
    return quiet_hours


def compute_base_features(
    con: duckdb.DuckDBPyConnection,
    quiet_hours: Sequence[int],
    base_features_path: str | Path,
    window_days: int = 14,
    n_win: int | None = None,
    min_agg: int = 20,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute aggregate account-window features.

    Main behavioural features from `e` = Success + LogOn:
    - volume
    - fanout
    - active_days
    - quiet_frac
    - iat_cv

    Auxiliary protocol features from `a` = Success, all orientations:
    - ticket_frac
    - logon_miss_frac
    """
    base_features_path = Path(base_features_path)

    if base_features_path.exists() and not force:
        base = con.execute(
            f"SELECT * FROM '{base_features_path}'"
        ).df()
        print(
            f"Loaded base features: {base.shape} "
            f"from {base_features_path}"
        )
        return base

    if n_win is None:
        max_t = con.execute("SELECT MAX(t) FROM a").fetchone()[0]
        n_win = max(
            1,
            int(max_t // (window_days * 86400)),
        )

    window = window_days * 86400
    quiet_list = ",".join(map(str, quiet_hours))

    print("Computing aggregate features ...")
    base = con.execute(f"""
    WITH agg AS (
        SELECT
            u,
            win,
            COUNT(*) AS volume,
            COUNT(DISTINCT dst_comp) AS fanout,
            COUNT(DISTINCT t // 86400) AS active_days,
            AVG(
                CASE
                    WHEN CAST((t % 86400) // 3600 AS INT)
                         IN ({quiet_list})
                    THEN 1.0
                    ELSE 0.0
                END
            ) AS quiet_frac
        FROM e
        GROUP BY u, win
    ),
    gaps AS (
        SELECT
            u,
            win,
            t - LAG(t) OVER (
                PARTITION BY u, win
                ORDER BY t
            ) AS gap
        FROM e
    ),
    cv AS (
        SELECT
            u,
            win,
            STDDEV_POP(gap) / NULLIF(AVG(gap), 0) AS iat_cv
        FROM gaps
        WHERE gap IS NOT NULL
        GROUP BY u, win
    ),
    aux AS (
        SELECT
            src_user AS u,
            CAST(t // {window} AS INT) AS win,
            AVG(
                CASE
                    WHEN orient IN ('TGS', 'TGT')
                    THEN 1.0
                    ELSE 0.0
                END
            ) AS ticket_frac,
            AVG(
                CASE
                    WHEN logon = '?'
                    THEN 1.0
                    ELSE 0.0
                END
            ) AS logon_miss_frac
        FROM a
        WHERE t < {n_win * window}
        GROUP BY 1, 2
    )
    SELECT
        agg.*,
        cv.iat_cv,
        aux.ticket_frac,
        aux.logon_miss_frac
    FROM agg
    LEFT JOIN cv USING (u, win)
    LEFT JOIN aux USING (u, win)
    WHERE agg.volume >= {min_agg}
    """).df()

    write_parquet(
        con,
        base,
        base_features_path,
        "_base_features",
    )
    print(
        f"Saved base features: {base.shape} "
        f"-> {base_features_path}"
    )
    return base


def fisher_periodicity_strength(
    times: Sequence[int] | np.ndarray,
    horizon: int,
    bin_s: int = 10,
    max_period_s: int = 6 * 3600,
    min_period: int = 50,
    robust: bool = True,
) -> float:
    """
    Fisher-type periodicity strength in [0, 300].

    Procedure:
    - convert event times to relative bins within the account-window;
    - use 10-second bins by default;
    - optionally apply log1p to counts for burst robustness;
    - compute the periodogram after removing the zero frequency;
    - retain frequencies corresponding to periods <= max_period_s;
    - calculate the historical Fisher-g approximation

          log10(p_approx) =
              log10(n) + (n - 1) * log10(1 - g)

      where g is the maximum retained spectral mass divided by total retained
      spectral mass;
    - clip log10(p_approx) to [-300, 0];
    - return -log10(p_approx).

    The clipping at zero is important: the approximation can otherwise exceed
    p=1 and yield a negative "strength". The result is therefore described as a
    Fisher-type periodicity strength rather than an exact p-value.
    """
    nbins = int(horizon // bin_s)

    idx = np.asarray(times, dtype=np.int64) // bin_s
    idx = idx[(idx >= 0) & (idx < nbins)]

    if len(idx) < min_period:
        return np.nan

    counts = np.bincount(
        idx,
        minlength=nbins,
    ).astype(float)

    if robust:
        counts = np.log1p(counts)

    x = counts - counts.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]  # remove zero frequency

    periods = (
        (nbins * bin_s)
        / np.arange(1, len(spec) + 1)
    )
    band = spec[periods <= max_period_s]

    if len(band) < 10 or band.sum() == 0:
        return np.nan

    n = len(band)
    g = band.max() / band.sum()

    logp_approx = (
        np.log10(n)
        + (n - 1)
        * np.log10(max(1 - g, 1e-15))
    )

    logp_approx = np.clip(
        logp_approx,
        -300.0,
        0.0,
    )

    return float(-logp_approx)


# Backwards-compatible function name used by older notebooks.
def fisher_neglogp(
    times: Sequence[int] | np.ndarray,
    horizon: int,
    bin_s: int = 10,
    max_period_s: int = 6 * 3600,
    min_period: int = 50,
    robust: bool = True,
) -> float:
    """Alias for the corrected Fisher-type periodicity strength."""
    return fisher_periodicity_strength(
        times=times,
        horizon=horizon,
        bin_s=bin_s,
        max_period_s=max_period_s,
        min_period=min_period,
        robust=robust,
    )


def compute_periodicity(
    con: duckdb.DuckDBPyConnection,
    period_parts_dir: str | Path,
    period_features_path: str | Path,
    n_win: int,
    window_days: int = 14,
    min_period: int = 50,
    chunk_size: int = 1000,
    bin_s: int = 10,
    max_period_s: int = 6 * 3600,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute corrected Fisher-type period_strength for each account-window.

    Results are checkpointed by window, allowing interrupted fresh calculations
    to resume without recomputing completed windows.
    """
    period_parts_dir = Path(period_parts_dir)
    period_features_path = Path(period_features_path)

    period_parts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    period_features_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if period_features_path.exists() and not force:
        period = con.execute(
            f"SELECT * FROM '{period_features_path}'"
        ).df()
        print(
            f"Loaded periodicity table: {period.shape} "
            f"from {period_features_path}"
        )
        return period

    if force:
        for part in period_parts_dir.glob(
            "periodicity_win*.parquet"
        ):
            part.unlink()

        if period_features_path.exists():
            period_features_path.unlink()

    window = window_days * 86400
    rows_all: list[pd.DataFrame] = []

    for w in range(n_win):
        part_path = (
            period_parts_dir
            / f"periodicity_win{w}.parquet"
        )

        if part_path.exists() and not force:
            part = con.execute(
                f"SELECT * FROM '{part_path}'"
            ).df()

            # Canonical v2 checkpoints must never contain negative scores.
            if (
                "period_strength" in part.columns
                and part["period_strength"].dropna().lt(0).any()
            ):
                raise ValueError(
                    f"{part_path} contains negative period_strength values. "
                    "It is not a valid v2 checkpoint."
                )

            print(
                f"Window {w}: loaded {len(part):,} rows "
                "from canonical checkpoint"
            )
            rows_all.append(part)
            continue

        users = con.execute(f"""
            SELECT u
            FROM e
            WHERE win = {w}
            GROUP BY u
            HAVING COUNT(*) >= {min_period}
            ORDER BY u
        """).df()["u"].tolist()

        print(
            f"Window {w}: {len(users):,} users "
            "reach periodicity threshold"
        )

        period_rows = []

        for i in range(0, len(users), chunk_size):
            chunk = users[i : i + chunk_size]

            con.register(
                "_chunk_users",
                pd.DataFrame({"u": chunk}),
            )
            try:
                ts = con.execute(f"""
                    SELECT
                        u,
                        array_agg(
                            t - {w * window}
                            ORDER BY t
                        ) AS times
                    FROM e
                    WHERE win = {w}
                      AND u IN (
                          SELECT u
                          FROM _chunk_users
                      )
                    GROUP BY u
                """).df()
            finally:
                try:
                    con.unregister("_chunk_users")
                except Exception:
                    pass

            for row in ts.itertuples(index=False):
                period_rows.append(
                    (
                        row.u,
                        w,
                        fisher_periodicity_strength(
                            row.times,
                            horizon=window,
                            bin_s=bin_s,
                            max_period_s=max_period_s,
                            min_period=min_period,
                            robust=True,
                        ),
                    )
                )

        part = pd.DataFrame(
            period_rows,
            columns=[
                "u",
                "win",
                "period_strength",
            ],
        )

        write_parquet(
            con,
            part,
            part_path,
            f"_period_win_{w}",
        )

        rows_all.append(part)
        print(
            f"Window {w}: saved {len(part):,} rows "
            f"-> {part_path}"
        )

    if rows_all:
        period = pd.concat(
            rows_all,
            ignore_index=True,
        )
    else:
        period = pd.DataFrame(
            columns=[
                "u",
                "win",
                "period_strength",
            ]
        )

    write_parquet(
        con,
        period,
        period_features_path,
        "_period_all",
    )

    print(
        f"Saved periodicity table: {period.shape} "
        f"-> {period_features_path}"
    )
    return period


def build_final_features(
    con: duckdb.DuckDBPyConnection,
    base_features_path: str | Path,
    period_features_path: str | Path,
    final_features_path: str | Path,
    compatibility_path: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Merge aggregate and periodicity features and add log transforms.

    Canonical transformed features:
    - log_volume = log10(volume)
    - log_fanout = log10(fanout + 1)
    """
    final_features_path = Path(
        final_features_path
    )

    if final_features_path.exists() and not force:
        feat = con.execute(
            f"SELECT * FROM '{final_features_path}'"
        ).df()

        print(
            f"Loaded final features: {feat.shape} "
            f"from {final_features_path}"
        )
        return feat

    base = con.execute(
        f"SELECT * FROM '{base_features_path}'"
    ).df()

    period = con.execute(
        f"SELECT * FROM '{period_features_path}'"
    ).df()

    feat = base.merge(
        period,
        on=["u", "win"],
        how="left",
    )

    feat["log_volume"] = np.log10(
        feat["volume"].clip(lower=1)
    )
    feat["log_fanout"] = np.log10(
        feat["fanout"].clip(lower=0) + 1
    )

    order = [
        "u",
        "win",
        "volume",
        "fanout",
        "active_days",
        "quiet_frac",
        "iat_cv",
        "ticket_frac",
        "logon_miss_frac",
        "period_strength",
        "log_volume",
        "log_fanout",
    ]

    feat = feat[
        [c for c in order if c in feat.columns]
    ]

    write_parquet(
        con,
        feat,
        final_features_path,
        "_feat_final",
    )

    print(
        f"Saved final features: {feat.shape} "
        f"-> {final_features_path}"
    )

    if compatibility_path is not None:
        write_parquet(
            con,
            feat,
            compatibility_path,
            "_feat_compat",
        )
        print(
            "Also wrote compatibility checkpoint "
            f"-> {compatibility_path}"
        )

    return feat
