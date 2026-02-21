import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="RPL 6 Predictor Dashboard", layout="wide")

DATA_DIR = Path("data")
DROP_MASTER_PATH = DATA_DIR / "Drop Master with final correct options.xlsx"
PREDICTOR_MASTER_PATH = DATA_DIR / "RPL 6 Predictor Answer Master.xlsx"

PREDICTOR_SHEET = "Scoring master"  # as per your file

# -----------------------------
# Data Loading + Parsing
# -----------------------------
@st.cache_data
def load_drop_master(path: Path, file_mtime: float) -> pd.DataFrame:
    """
    Reads the 'Drop Master with final correct options.xlsx' and returns:
      drop (int), question (str), correct_option (str), status (valid/scrapped/calculated)
    """
    df = pd.read_excel(path, sheet_name=0)

    # The first row contains column names: Drop #, Predictor Question, Correct Option
    header = df.iloc[0].tolist()
    # Expected positions: [nan, 'Drop #', 'Predictor Question', 'Correct Option']
    clean = df.iloc[1:].copy()
    clean.columns = ["_", "drop", "question", "correct_option"]
    clean = clean.drop(columns=["_"])

    clean["drop"] = clean["drop"].astype(int)
    clean["question"] = clean["question"].astype(str).str.strip()
    clean["correct_option"] = clean["correct_option"].astype(str).str.strip()

    def infer_status(x: str) -> str:
        xl = x.strip().lower()
        if "scrap" in xl:
            return "scrapped"
        if "calculated" in xl:
            return "calculated"
        return "valid"

    clean["status"] = clean["correct_option"].apply(infer_status)
    return clean.sort_values("drop").reset_index(drop=True)


@st.cache_data
def load_predictor_long(path: Path, file_mtime: float) -> pd.DataFrame:
    """
    Reads the 'RPL 6 Predictor Answer Master.xlsx' (Scoring master) which is in a
    wide 2-col-per-drop format, and converts it into a long table:

    columns:
      player_name, drop, response, points, attempted, power_play
    """
    raw = pd.read_excel(path, sheet_name=PREDICTOR_SHEET, header=None)

    # Player rows are 50 players: rows 2..51 (0-indexed)
    players_block = raw.iloc[2:52].copy()

    # Row 0 has 'Drop 1', 'Drop 2' ... in every response column
    row0 = raw.iloc[0]
    drop_response_cols = [
        i for i, v in enumerate(row0)
        if isinstance(v, str) and v.strip().lower().startswith("drop")
    ]
    if len(drop_response_cols) != 38:
        raise ValueError(f"Expected 38 drops, but found {len(drop_response_cols)} in header row.")

    records = []
    for _, row in players_block.iterrows():
        player_name = str(row[1]).strip()

        # Power play drops are in columns 2 and 3
        pp_set = set()
        for pp in [row[2], row[3]]:
            if pd.isna(pp):
                continue
            try:
                pp_set.add(int(pp))
            except Exception:
                pass

        # For each drop, response is in col, score in col+1
        for drop_num, resp_col in enumerate(drop_response_cols, start=1):
            resp = row[resp_col]
            score = row[resp_col + 1]

            attempted = not (pd.isna(resp) or (isinstance(resp, str) and resp.strip() == ""))
            points = 0.0 if pd.isna(score) else float(score)

            records.append({
                "player_name": player_name,
                "drop": int(drop_num),
                "response": None if not attempted else resp,
                "points": float(points),     # already 0/1/3 per your scoring
                "attempted": int(attempted),
                "power_play": int(drop_num in pp_set),
            })

    df = pd.DataFrame.from_records(records)
    df["is_correct"] = (df["points"] > 0).astype(int)
    return df


@st.cache_data
def build_models(drop_master: pd.DataFrame, predictor_long: pd.DataFrame):
    """
    Precompute helpful tables used across the app.
    """
    # Merge in question + correct option + status
    merged = predictor_long.merge(
        drop_master[["drop", "question", "correct_option", "status"]],
        on="drop",
        how="left"
    )

    scrapped_drops = set(drop_master.loc[drop_master["status"] == "scrapped", "drop"].tolist())
    valid_drops = drop_master.loc[drop_master["status"] == "valid", "drop"].tolist()
    calculated_drops = drop_master.loc[drop_master["status"] == "calculated", "drop"].tolist()

    scorable_attendance_drops = [d for d in drop_master["drop"].tolist() if d not in scrapped_drops]

    # Active players: anyone with at least 1 attempted in scorable drops
    attempted_scorable = merged[merged["drop"].isin(scorable_attendance_drops)]
    active_players = (
        attempted_scorable.groupby("player_name")["attempted"].sum()
        .reset_index(name="attempts")
    )
    active_set = set(active_players.loc[active_players["attempts"] > 0, "player_name"].tolist())

    # Attendance per player (exclude scrapped)
    attendance = (
        merged[merged["drop"].isin(scorable_attendance_drops)]
        .groupby("player_name")["attempted"].sum()
        .reset_index(name="attempted_count")
    )
    attendance["attendance_den"] = len(scorable_attendance_drops)
    attendance["attendance_pct"] = attendance["attempted_count"] / attendance["attendance_den"]

    # 100% attendance list
    full_attendance = attendance.loc[attendance["attempted_count"] == attendance["attendance_den"], "player_name"].tolist()

    # Drop-level stats (exclude scrapped for correctness/difficulty)
    drop_stats = []
    for d in drop_master["drop"].tolist():
        subset = merged[merged["drop"] == d]
        attempted = subset["attempted"].sum()
        correct = subset["is_correct"].sum()
        pp_used = subset["power_play"].sum()
        pp_correct = subset.loc[subset["power_play"] == 1, "is_correct"].sum()

        drop_stats.append({
            "drop": d,
            "question": drop_master.loc[drop_master["drop"] == d, "question"].iloc[0],
            "correct_option": drop_master.loc[drop_master["drop"] == d, "correct_option"].iloc[0],
            "status": drop_master.loc[drop_master["drop"] == d, "status"].iloc[0],
            "attempted": int(attempted),
            "correct": int(correct),
            "accuracy_pct": (float(correct) / float(attempted) * 100.0) if attempted else 0.0,
            "pp_used": int(pp_used),
            "pp_correct": int(pp_correct),
            "pp_success_pct": (float(pp_correct) / float(pp_used) * 100.0) if pp_used else 0.0,
        })

    drop_stats_df = pd.DataFrame(drop_stats).sort_values("drop")

    # Difficulty only among VALID drops
    valid_stats = drop_stats_df[drop_stats_df["status"] == "valid"].copy()
    # Define difficulty: lower accuracy = harder
    valid_stats["difficulty_score"] = 1 - (valid_stats["accuracy_pct"] / 100.0)
    hardest_drops = valid_stats.sort_values(["accuracy_pct", "attempted"], ascending=[True, False]).head(5)

    # Drops where nobody got it right (VALID only)
    unsolved_drops = valid_stats[(valid_stats["attempted"] > 0) & (valid_stats["correct"] == 0)].copy()

    return merged, drop_stats_df, hardest_drops, unsolved_drops, full_attendance, active_set, scorable_attendance_drops


# -----------------------------
# Leaderboards / Rankings
# -----------------------------
def leaderboard_upto(merged: pd.DataFrame, upto_drop: int) -> pd.DataFrame:
    df = merged[merged["drop"] <= upto_drop].copy()
    lb = (
        df.groupby("player_name", as_index=False)
        .agg(points=("points", "sum"),
             correct=("is_correct", "sum"),
             pp_used=("power_play", "sum"),
             attempted=("attempted", "sum"))
    )
    lb = lb.sort_values(["points", "correct"], ascending=False).reset_index(drop=True)

    # Rank with ties allowed: method='min'
    lb["rank"] = lb["points"].rank(method="min", ascending=False).astype(int)

    # within same points, you may also want correctness ordering for display stability:
    lb = lb.sort_values(["points", "correct", "player_name"], ascending=[False, False, True]).reset_index(drop=True)
    return lb


def add_rank_delta(merged: pd.DataFrame, upto_drop: int) -> pd.DataFrame:
    now = leaderboard_upto(merged, upto_drop)
    if upto_drop <= 1:
        now["delta"] = "—"
        return now

    prev = leaderboard_upto(merged, upto_drop - 1)[["player_name", "rank"]].rename(columns={"rank": "rank_prev"})
    out = now.merge(prev, on="player_name", how="left")
    out["change"] = out["rank_prev"] - out["rank"]

    def fmt(x):
        if pd.isna(x) or x == 0:
            return "—"
        return f"▲{int(x)}" if x > 0 else f"▼{abs(int(x))}"

    out["delta"] = out["change"].apply(fmt)
    return out


def top_n_with_ties(lb: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return top N ranks, including everyone tied at the cutoff rank.
    """
    if lb.empty:
        return lb
    lb_sorted = lb.sort_values(["rank", "player_name"], ascending=[True, True]).copy()
    cutoff_rank = lb_sorted[lb_sorted["rank"] <= n]["rank"].max() if any(lb_sorted["rank"] <= n) else lb_sorted["rank"].min()
    return lb_sorted[lb_sorted["rank"] <= cutoff_rank].copy()


# -----------------------------
# UI Helpers
# -----------------------------
def drop_label(drop_row: pd.Series) -> str:
    # Dropdown label: "Drop X — Question"
    return f"Drop {int(drop_row['drop'])} — {drop_row['question']}"


def safe_pct(x: float) -> str:
    return f"{x:.1f}%"


# -----------------------------
# Load everything
# -----------------------------
if not DROP_MASTER_PATH.exists() or not PREDICTOR_MASTER_PATH.exists():
    st.error(
        "Missing required data files.\n\n"
        "Please make sure these exist in the repo:\n"
        f"- {DROP_MASTER_PATH}\n"
        f"- {PREDICTOR_MASTER_PATH}\n"
    )
    st.stop()

drop_master = load_drop_master(
    DROP_MASTER_PATH,
    DROP_MASTER_PATH.stat().st_mtime
)

predictor_long = load_predictor_long(
    PREDICTOR_MASTER_PATH,
    PREDICTOR_MASTER_PATH.stat().st_mtime
)

merged, drop_stats_df, hardest_drops_df, unsolved_drops_df, full_attendance_list, active_set, scorable_attendance_drops = build_models(drop_master, predictor_long)

max_drop = int(drop_master["drop"].max())
final_lb = add_rank_delta(merged, max_drop)

# -----------------------------
# App Layout
# -----------------------------
st.title("RPL 6 Predictor — Results Dashboard")

tabs = st.tabs(["🏁 Overview", "🎯 Drop Explorer", "👤 Player Explorer", "📈 Leaderboard Race", "📜 Answer Key"])

# =============================
# Overview
# =============================
with tabs[0]:
    st.subheader("Season Summary")

    total_participants = merged["player_name"].nunique()  # should be 50
    active_count = len(active_set)
    total_predictions = int(merged["attempted"].sum())
    total_pp_used = int(merged["power_play"].sum())

    # Overall accuracy on valid drops (exclude scrapped + calculated)
    valid_mask = merged["status"].eq("valid")
    valid_attempted = int(merged.loc[valid_mask, "attempted"].sum())
    valid_correct = int(merged.loc[valid_mask, "is_correct"].sum())
    overall_valid_acc = (valid_correct / valid_attempted * 100.0) if valid_attempted else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participants", total_participants)
    c2.metric("Active players", active_count)
    c3.metric("Total predictions", total_predictions)
    c4.metric("Total Power Plays used", total_pp_used)
    c5.metric("Overall accuracy (valid drops)", safe_pct(overall_valid_acc))

    st.divider()

    st.subheader("Top 5 (+ ties)")
    top5 = top_n_with_ties(final_lb, n=5)
    st.dataframe(
        top5[["rank", "delta", "player_name", "points", "correct", "pp_used", "attempted"]]
        .rename(columns={
            "rank": "Rank", "delta": "Δ", "player_name": "Player",
            "points": "Points", "correct": "Correct", "pp_used": "PP Used", "attempted": "Attempted"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Final Leaderboard (Full)")
    st.dataframe(
        final_lb[["rank", "delta", "player_name", "points", "correct", "pp_used", "attempted"]]
        .rename(columns={
            "rank": "Rank", "delta": "Δ", "player_name": "Player",
            "points": "Points", "correct": "Correct", "pp_used": "PP Used", "attempted": "Attempted"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # 1) 100% Attendance
    st.subheader("100% Attendance")
    st.caption(f"Attendance calculated across {len(scorable_attendance_drops)} drops (excluding scrapped drops).")

    if full_attendance_list:
        st.write(f"**{len(full_attendance_list)} players** have 100% attendance:")
        st.write(", ".join(sorted(full_attendance_list)))
    else:
        st.write("No one has 100% attendance (as per recorded attempts).")

    st.divider()

    # 2) Hardest drops + who got them right
    st.subheader("Hardest Drops (Valid questions only)")
    st.caption("Hardest = lowest % correct among attempted responses.")

    # Display hardest drops table
    hd_view = hardest_drops_df[["drop", "question", "accuracy_pct", "attempted", "correct"]].copy()
    hd_view["accuracy_pct"] = hd_view["accuracy_pct"].map(lambda x: safe_pct(x))
    st.dataframe(
        hd_view.rename(columns={"drop": "Drop", "question": "Question", "accuracy_pct": "% Correct", "attempted": "Attempted", "correct": "Correct"}),
        use_container_width=True,
        hide_index=True
    )

    # Who got hardest drops right
    st.markdown("#### Players who got the hardest drops right")
    for _, r in hardest_drops_df.iterrows():
        d = int(r["drop"])
        subset = merged[(merged["drop"] == d) & (merged["is_correct"] == 1)]
        names = sorted(subset["player_name"].unique().tolist())
        st.markdown(f"**Drop {d}** — {r['question']}")
        st.caption(f"Correct by {len(names)} players ({safe_pct(r['accuracy_pct'])})")
        if names:
            st.write(", ".join(names))
        else:
            st.write("No one got this right.")
        st.write("")

    st.divider()

    # 3) Drops nobody got right
    st.subheader("Unsolved Drops (0 correct)")
    if len(unsolved_drops_df) == 0:
        st.write("None — every valid drop had at least one correct answer.")
    else:
        show_answers = st.toggle("Show correct options for unsolved drops", value=True)
        for _, r in unsolved_drops_df.iterrows():
            st.markdown(f"**Drop {int(r['drop'])}** — {r['question']}")
            st.caption(f"Attempted: {int(r['attempted'])} • Correct: 0 • Status: {r['status']}")
            if show_answers:
                st.write(f"✅ Correct option: **{r['correct_option']}**")
            st.write("")

# =============================
# Drop Explorer
# =============================
with tabs[1]:
    st.subheader("Drop Explorer")

    drop_options = drop_master.copy()
    labels = drop_options.apply(drop_label, axis=1).tolist()
    label_to_drop = {labels[i]: int(drop_options.iloc[i]["drop"]) for i in range(len(labels))}

    selected_label = st.selectbox("Select a question", labels, index=0)
    selected_drop = label_to_drop[selected_label]

    drow = drop_master.loc[drop_master["drop"] == selected_drop].iloc[0]
    status = drow["status"]

    st.markdown(f"### Drop {selected_drop}")
    st.write(drow["question"])

    # Correct option reveal toggle
    reveal = st.toggle("Show correct option", value=True)
    if reveal:
        st.write(f"✅ Correct option: **{drow['correct_option']}**")
    st.caption(f"Status: **{status.upper()}**")

    # Drop stats
    ds = drop_stats_df.loc[drop_stats_df["drop"] == selected_drop].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Attempted", int(ds["attempted"]))
    c2.metric("Correct", int(ds["correct"]))
    c3.metric("Accuracy", safe_pct(float(ds["accuracy_pct"])))
    c4.metric("PP used", int(ds["pp_used"]))
    c5.metric("PP success", safe_pct(float(ds["pp_success_pct"])) if int(ds["pp_used"]) > 0 else "—")

    st.divider()

    # Response distribution (raw)
    st.markdown("#### Response distribution")
    resp_subset = merged[merged["drop"] == selected_drop].copy()
    dist = (
        resp_subset[resp_subset["attempted"] == 1]
        .groupby("response")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if dist.empty:
        st.write("No responses recorded for this drop.")
    else:
        dist["pct"] = dist["count"] / dist["count"].sum() * 100.0
        st.dataframe(
            dist.rename(columns={"response": "Response", "count": "Count", "pct": "%"}).assign(**{"%": dist["pct"].map(safe_pct)}),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.markdown("#### Leaderboard up to this drop (Top 5 + ties)")
    lb = add_rank_delta(merged, selected_drop)
    st.dataframe(
        top_n_with_ties(lb, n=5)[["rank", "delta", "player_name", "points", "correct", "pp_used", "attempted"]]
        .rename(columns={"rank": "Rank", "delta": "Δ", "player_name": "Player", "points": "Points", "correct": "Correct", "pp_used": "PP Used", "attempted": "Attempted"}),
        use_container_width=True,
        hide_index=True
    )

    with st.expander("Show full leaderboard up to this drop"):
        st.dataframe(
            lb[["rank", "delta", "player_name", "points", "correct", "pp_used", "attempted"]]
            .rename(columns={"rank": "Rank", "delta": "Δ", "player_name": "Player", "points": "Points", "correct": "Correct", "pp_used": "PP Used", "attempted": "Attempted"}),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Who got it right / wrong (this drop)"):
        view = resp_subset[["player_name", "response", "attempted", "power_play", "is_correct", "points"]].copy()
        view["Result"] = np.where(view["attempted"] == 0, "— Not attempted",
                                 np.where(view["is_correct"] == 1, "✅ Correct", "❌ Wrong"))
        view = view.sort_values(["is_correct", "power_play", "points", "player_name"], ascending=[False, False, False, True])
        st.dataframe(
            view.rename(columns={
                "player_name": "Player",
                "response": "Response",
                "attempted": "Attempted?",
                "power_play": "Power Play",
                "points": "Points Earned"
            }),
            use_container_width=True,
            hide_index=True
        )

# =============================
# Player Explorer
# =============================
with tabs[2]:
    st.subheader("Player Explorer")

    players = sorted(merged["player_name"].unique().tolist())
    player = st.selectbox("Select a player", players)

    p = merged[merged["player_name"] == player].sort_values("drop").copy()
    total_points = float(p["points"].sum())
    attempted = int(p[p["drop"].isin(scorable_attendance_drops)]["attempted"].sum())
    den = len(scorable_attendance_drops)
    attendance_pct = (attempted / den * 100.0) if den else 0.0

    valid_mask = p["status"].eq("valid")
    valid_attempted = int(p.loc[valid_mask, "attempted"].sum())
    valid_correct = int(p.loc[valid_mask, "is_correct"].sum())
    acc = (valid_correct / valid_attempted * 100.0) if valid_attempted else 0.0

    pp_used = int(p["power_play"].sum())
    pp_hits = int(p.loc[p["power_play"] == 1, "is_correct"].sum())
    pp_hit_rate = (pp_hits / pp_used * 100.0) if pp_used else 0.0

    # Final rank
    final_rank_row = final_lb[final_lb["player_name"] == player]
    final_rank = int(final_rank_row["rank"].iloc[0]) if not final_rank_row.empty else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final Rank", final_rank if final_rank is not None else "—")
    c2.metric("Total Points", int(total_points))
    c3.metric("Attendance", safe_pct(attendance_pct))
    c4.metric("Accuracy (valid drops)", safe_pct(acc))
    c5.metric("PP Hit Rate", safe_pct(pp_hit_rate) if pp_used else "—")

    st.divider()

    st.markdown("#### Drop-by-drop log")
    log = p[["drop", "status", "response", "attempted", "power_play", "is_correct", "points"]].copy()
    log["Attempted"] = np.where(log["attempted"] == 1, "Yes", "No")
    log["Correct"] = np.where(log["attempted"] == 0, "—",
                             np.where(log["is_correct"] == 1, "Yes", "No"))
    log["PP"] = np.where(log["power_play"] == 1, "Yes", "No")
    log = log.drop(columns=["attempted", "is_correct", "power_play"])

    st.dataframe(
        log.rename(columns={"drop": "Drop", "status": "Status", "response": "Response", "points": "Points"}),
        use_container_width=True,
        hide_index=True
    )

# =============================
# Leaderboard Race
# =============================
with tabs[3]:
    st.subheader("Leaderboard Race (Rank over time)")

    # Precompute ranks for each drop
    drops = drop_master["drop"].tolist()

    # Default: top 10 by final points (stable set)
    top_n = st.slider("Show top N players (by final result)", min_value=5, max_value=20, value=10, step=1)
    final_sorted = final_lb.sort_values(["rank", "player_name"])
    focus_players = final_sorted.head(top_n)["player_name"].tolist()

    # Build rank timeline for those players
    timeline = []
    for d in drops:
        lb_d = leaderboard_upto(merged, d)[["player_name", "rank", "points"]]
        lb_d = lb_d[lb_d["player_name"].isin(focus_players)].copy()
        lb_d["drop"] = d
        timeline.append(lb_d)
    tl = pd.concat(timeline, ignore_index=True)

    # Pivot to make line chart data
    pivot = tl.pivot(index="drop", columns="player_name", values="rank").sort_index()

    st.caption("Lower rank number = better. Chart is inverted so #1 appears at the top.")
    st.line_chart(pivot)

    # Optional: show points timeline
    with st.expander("Show points over time (same players)"):
        pivot_pts = tl.pivot(index="drop", columns="player_name", values="points").sort_index()
        st.line_chart(pivot_pts)

# =============================
# Answer Key
# =============================
with tabs[4]:
    st.subheader("Answer Key (All Drops)")
    show_answers = st.toggle("Show answers", value=False)

    for _, r in drop_master.sort_values("drop").iterrows():
        d = int(r["drop"])
        status = r["status"]

        # Small status tag
        tag = "✅ VALID" if status == "valid" else ("🧮 CALCULATED" if status == "calculated" else "🗑️ SCRAPPED")

        st.markdown(f"### Drop {d} — {tag}")
        st.write(r["question"])
        if show_answers:
            st.write(f"**Answer:** {r['correct_option']}")

        st.divider()

