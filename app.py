import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="RPL 6 Predictor: Experience Centre", layout="wide")

# --- Broadcast hero header (gold accent) ---
st.markdown("""
<div style="
  background: radial-gradient(circle at 10% 10%, rgba(245,197,66,0.18), rgba(0,0,0,0) 40%),
              linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(245,197,66,0.06);
  border-radius: 18px;
  padding: 18px 20px;
  margin-bottom: 18px;
">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:34px;font-weight:900;color:#f5c542;line-height:1;">
      🏆 RPL 6 Predictor: Experience Centre
    </div>
    <div style="opacity:0.92;font-size:14px;margin-left:8px;">
      Season review • Drop-by-drop analytics • Leaderboard race • Answer archive
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
# --- end hero ---

st.markdown("""
<style>
/* Broadcast vibe: bold, clean, high-contrast */
.block-container { padding-top: 4.2rem; padding-bottom: 2rem; }
html, body, [class*="css"]  { -webkit-font-smoothing: antialiased; }

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { margin-bottom: 0.6rem; font-weight: 800; }
h2 { margin-top: 1.2rem; font-weight: 800; }
h3 { font-weight: 700; }

/* Metric tiles */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
  padding: 14px 14px 10px 14px;
  border-radius: 18px;
}

/* Dataframes look like TV panels */
[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* Tabs */
button[data-baseweb="tab"] {
  font-weight: 700;
  padding-top: 10px;
  padding-bottom: 10px;
}

/* Reduce dead space between sections */
hr { margin: 1.2rem 0; opacity: 0.25; }

.qa-card{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 14px 16px;
  margin: 12px 0 16px 0;
  box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}
.qa-title{
  font-weight: 800;
  font-size: 15px;
  margin-bottom: 6px;
}
.qa-sub{
  opacity: 0.85;
  font-size: 12.5px;
  margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")
DROP_MASTER_PATH = DATA_DIR / "Drop Master with final correct options.xlsx"
PREDICTOR_MASTER_PATH = DATA_DIR / "RPL 6 Predictor Answer Master.xlsx"
PREDICTOR_SHEET = "Scoring master"

# -----------------------------
# Data Loading (cache keyed by file mtime)
# -----------------------------
@st.cache_data
def load_drop_master(path: Path, file_mtime: float) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)

    clean = df.iloc[1:].copy()
    clean.columns = ["_", "drop", "question", "correct_option", "actuals"]
    clean = clean.drop(columns=["_"])
    clean["drop"] = clean["drop"].astype(int)
    clean["question"] = clean["question"].astype(str).str.strip()
    clean["correct_option"] = clean["correct_option"].astype(str).str.strip()
    clean["actuals"] = clean["actuals"].astype(str).str.strip()

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
    raw = pd.read_excel(path, sheet_name=PREDICTOR_SHEET, header=None)

    players_block = raw.iloc[2:52].copy()  # rows 3..52 => 50 players
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

        pp_set = set()
        for pp in [row[2], row[3]]:
            if pd.isna(pp):
                continue
            try:
                pp_set.add(int(pp))
            except Exception:
                pass

        for drop_num, resp_col in enumerate(drop_response_cols, start=1):
            resp = row[resp_col]
            score = row[resp_col + 1]

            resp_clean = "" if pd.isna(resp) else str(resp).strip()
            attempted = resp_clean != ""
            points = 0.0 if pd.isna(score) else float(score)

            records.append({
                "player_name": player_name,
                "drop": int(drop_num),
                "response": resp_clean,             # always string (prevents Arrow issues)
                "points": float(points),            # uses your score column (0/1/3)
                "attempted": int(attempted),
                "power_play": int(drop_num in pp_set),
            })

    df = pd.DataFrame.from_records(records)
    df["is_correct"] = (df["points"] > 0).astype(int)
    return df


@st.cache_data
def build_models(drop_master: pd.DataFrame, predictor_long: pd.DataFrame):
    merged = predictor_long.merge(
        drop_master[["drop", "question", "correct_option", "actuals", "status"]],
        on="drop",
        how="left"
    )

    scrapped_drops = set(drop_master.loc[drop_master["status"] == "scrapped", "drop"].tolist())
    scorable_attendance_drops = [d for d in drop_master["drop"].tolist() if d not in scrapped_drops]

    # Active = attempted at least once in scorable drops
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

    full_attendance = attendance.loc[
        attendance["attempted_count"] == attendance["attendance_den"],
        "player_name"
    ].tolist()

    # Drop stats
    drop_stats = []
    for d in drop_master["drop"].tolist():
        subset = merged[merged["drop"] == d]
        attempted = int(subset["attempted"].sum())
        correct = int(subset["is_correct"].sum())
        pp_used = int(subset["power_play"].sum())
        pp_correct = int(subset.loc[subset["power_play"] == 1, "is_correct"].sum())

        drop_stats.append({
            "drop": d,
            "question": drop_master.loc[drop_master["drop"] == d, "question"].iloc[0],
            "actuals": drop_master.loc[drop_master["drop"] == d, "actuals"].iloc[0],
            "correct_option": drop_master.loc[drop_master["drop"] == d, "correct_option"].iloc[0],
            "status": drop_master.loc[drop_master["drop"] == d, "status"].iloc[0],
            "attempted": attempted,
            "correct": correct,
            "accuracy_pct": (correct / attempted * 100.0) if attempted else 0.0,
            "pp_used": pp_used,
            "pp_correct": pp_correct,
        })

    drop_stats_df = pd.DataFrame(drop_stats).sort_values("drop")

    valid_stats = drop_stats_df[drop_stats_df["status"] == "valid"].copy()

    hardest = valid_stats.sort_values(["accuracy_pct", "attempted"], ascending=[True, False]).head(5)
    easiest = valid_stats.sort_values(["accuracy_pct", "attempted"], ascending=[False, False]).head(5)

    unsolved = valid_stats[(valid_stats["attempted"] > 0) & (valid_stats["correct"] == 0)].copy()

    return merged, drop_stats_df, hardest, easiest, unsolved, full_attendance, attendance, active_set, scorable_attendance_drops


def safe_pct(x: float) -> str:
    return f"{x:.1f}%"


def drop_label(drop_row: pd.Series) -> str:
    return f"Drop {int(drop_row['drop'])} — {drop_row['question']}"


# -----------------------------
# Leaderboards / Rankings
# -----------------------------
def leaderboard_upto(merged: pd.DataFrame, upto_drop: int) -> pd.DataFrame:
    df = merged[merged["drop"] <= upto_drop].copy()

    lb = (
        df.groupby("player_name", as_index=False)
        .agg(points=("points", "sum"),
             correct=("is_correct", "sum"),
             attempted=("attempted", "sum"),
             pp_used=("power_play", "sum"),
             pp_correct=("power_play", lambda s: int(((df.loc[s.index, "power_play"] == 1) & (df.loc[s.index, "is_correct"] == 1)).sum())))
    )

    # Dense rank (1,2,2,3...) based on points
    lb["rank"] = lb["points"].rank(method="dense", ascending=False).astype(int)

    # stable sorting
    lb = lb.sort_values(["rank", "player_name"], ascending=[True, True]).reset_index(drop=True)
    return lb

def leaderboard_with_movement(merged: pd.DataFrame, upto_drop: int) -> pd.DataFrame:
    # leaderboard now
    now = leaderboard_upto(merged, upto_drop).copy()

    # if first drop, nothing to compare to
    if upto_drop <= 1:
        now["Move"] = "—"
        return now

    # leaderboard previous drop
    prev = leaderboard_upto(merged, upto_drop - 1)[["player_name", "rank"]].rename(columns={"rank": "prev_rank"})

    # join prev rank to current
    out = now.merge(prev, on="player_name", how="left")

    # movement = prev_rank - current_rank (positive means moved UP)
    out["delta"] = out["prev_rank"] - out["rank"]

    def fmt(d):
        if pd.isna(d) or d == 0:
            return "—"
        return f"▲{int(d)}" if d > 0 else f"▼{abs(int(d))}"

    out["Move"] = out["delta"].apply(fmt)
    return out


def top_full_leaderboard_view(lb: pd.DataFrame, attendance: pd.DataFrame) -> pd.DataFrame:
    out = lb.merge(attendance[["player_name", "attendance_pct"]], on="player_name", how="left")
    out["attendance_pct"] = out["attendance_pct"].fillna(0.0) * 100.0

    # show exactly requested columns: Rank, Name, Score, #PP correct, Attendance%
    out = out[["rank", "player_name", "points", "pp_correct", "attendance_pct"]].copy()
    out = out.rename(columns={
        "rank": "Rank",
        "player_name": "Name",
        "points": "Score",
        "pp_correct": "#PP correct",
        "attendance_pct": "Attendance%"
    })
    out["Attendance%"] = out["Attendance%"].map(safe_pct)
    return out


# -----------------------------
# Load everything
# -----------------------------
if not DROP_MASTER_PATH.exists() or not PREDICTOR_MASTER_PATH.exists():
    st.error("Missing required data files in /data folder.")
    st.stop()

drop_master = load_drop_master(DROP_MASTER_PATH, DROP_MASTER_PATH.stat().st_mtime)
predictor_long = load_predictor_long(PREDICTOR_MASTER_PATH, PREDICTOR_MASTER_PATH.stat().st_mtime)
merged, drop_stats_df, hardest_df, easiest_df, unsolved_df, full_attendance_list, attendance_df, active_set, scorable_attendance_drops = build_models(drop_master, predictor_long)

max_drop = int(drop_master["drop"].max())
final_lb_raw = leaderboard_upto(merged, max_drop)
final_lb = top_full_leaderboard_view(final_lb_raw, attendance_df)

# PP both correct / both wrong
pp_summary = (
    merged.groupby("player_name", as_index=False)
    .agg(pp_used=("power_play", "sum"),
         pp_correct=("power_play", lambda s: int(((merged.loc[s.index, "power_play"] == 1) & (merged.loc[s.index, "is_correct"] == 1)).sum())))
)
pp_summary["pp_wrong"] = pp_summary["pp_used"] - pp_summary["pp_correct"]

both_pp_correct_names = sorted(pp_summary[(pp_summary["pp_used"] == 2) & (pp_summary["pp_correct"] == 2)]["player_name"].tolist())
both_pp_wrong_names = sorted(pp_summary[(pp_summary["pp_used"] == 2) & (pp_summary["pp_wrong"] == 2)]["player_name"].tolist())

# -----------------------------
# App Layout
# -----------------------------

tabs = st.tabs(["🏁 Overview", "🎯 Drop Explorer", "👤 Player Explorer", "📈 Leaderboard Race", "📜 Answer Key"])

# =============================
# Overview
# =============================
with tabs[0]:
    st.header("🏁 Season Overview")

    # -----------------------------
    # Season summary tiles
    # -----------------------------
    total_participants = merged["player_name"].nunique()
    active_count = len(active_set)
    total_predictions = int(merged["attempted"].sum())
    total_pp_used = int(merged["power_play"].sum())
    total_drops = int(drop_master["drop"].nunique())  # should be 38

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participants", total_participants)
    c2.metric("Active players", active_count)
    c3.metric("Total predictions", total_predictions)
    c4.metric("Total Power Plays used", total_pp_used)
    c5.metric("Total drops", total_drops)

    st.divider()

    # -----------------------------
    # PP / Attendance tiles row
    # -----------------------------
    a, b, c = st.columns(3)

    with a:
        st.metric("Both PP correct", len(both_pp_correct_names))
        with st.popover("View names"):
            st.write(", ".join(both_pp_correct_names) if both_pp_correct_names else "None")

    with b:
        st.metric("Both PP wrong", len(both_pp_wrong_names))
        with st.popover("View names"):
            st.write(", ".join(both_pp_wrong_names) if both_pp_wrong_names else "None")

    with c:
        st.metric("100% attendance", len(full_attendance_list))
        with st.popover("View names"):
            st.write(", ".join(sorted(full_attendance_list)) if full_attendance_list else "None")

    st.divider()

    # -----------------------------
    # Full Leaderboard + Awards + Chart
    # -----------------------------
    st.subheader("Full Leaderboard")

    # Prepare numeric score + sorted leaderboard
    viz_df = final_lb.copy()
    viz_df["Score"] = pd.to_numeric(viz_df["Score"], errors="coerce").fillna(0)
    viz_df = viz_df.sort_values(["Score", "Name"], ascending=[False, True]).reset_index(drop=True)

    # ===== Awards Podium CSS =====
    st.markdown("""
    <style>
      .podium-row{
        display:grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin: 8px 0 10px 0;
      }
      .pod{
        border-radius: 16px;
        padding: 12px 14px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 12px 26px rgba(0,0,0,0.40);
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
        min-height: 108px;
      }
      .pod.gold{
        border-color: rgba(245,197,66,0.45);
        background: radial-gradient(circle at 18% 0%, rgba(245,197,66,0.25), rgba(0,0,0,0) 55%),
                    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
        box-shadow: 0 0 0 1px rgba(245,197,66,0.20),
                    0 0 26px rgba(245,197,66,0.18),
                    0 16px 34px rgba(0,0,0,0.45);
      }
      .pod.silver{
        border-color: rgba(200,200,200,0.38);
        background: radial-gradient(circle at 18% 0%, rgba(200,200,200,0.18), rgba(0,0,0,0) 55%),
                    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
      }
      .pod.bronze{
        border-color: rgba(205,127,50,0.48);
        background: radial-gradient(circle at 18% 0%, rgba(205,127,50,0.20), rgba(0,0,0,0) 55%),
                    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
      }
      .pod-title{ font-weight: 900; font-size: 15px; margin-bottom: 10px; letter-spacing: 0.06em; }
      .pod-line{ display:flex; justify-content:space-between; font-weight: 850; font-size: 16px; margin: 6px 0; }
      .pod-pts{ font-weight: 950; }
      @media (max-width: 900px){
        .podium-row{ grid-template-columns: 1fr; }
      }
    </style>
    """, unsafe_allow_html=True)

    # Podium groups (ties handled because Rank already exists in final_lb)
    podium_df = viz_df.copy()
    gold_df = podium_df[podium_df["Rank"] == 1][["Name", "Score"]]
    silver_df = podium_df[podium_df["Rank"] == 2][["Name", "Score"]]
    bronze_df = podium_df[podium_df["Rank"] == 3][["Name", "Score"]]

    def render_lines(df):
        if df.empty:
            return '<div style="opacity:0.7">—</div>'
        html = ""
        for _, r in df.iterrows():
            html += f'<div class="pod-line"><span>{r["Name"]}</span><span class="pod-pts">{int(r["Score"])} pts</span></div>'
        return html

    st.markdown(f"""
      <div class="podium-row">
        <div class="pod gold">
          <div class="pod-title">🥇 CHAMPION</div>
          {render_lines(gold_df)}
        </div>
        <div class="pod silver">
          <div class="pod-title">🥈 RUNNERS-UP</div>
          {render_lines(silver_df)}
        </div>
        <div class="pod bronze">
          <div class="pod-title">🥉 THIRD PLACE</div>
          {render_lines(bronze_df)}
        </div>
      </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ===== Leaderboard bar chart (with names at the right end) =====
    st.caption("Leaderboard visual (Top N by score). Use the slider to expand the field.")
    top_n = st.slider("Show Top N", min_value=10, max_value=25, value=15, step=1, key="lb_topn")

    chart_df = viz_df.head(top_n).copy()
    chart_df["right_label"] = (
        chart_df["Name"]
        + "  ("
        + chart_df["Score"].astype(int).astype(str)
        + " pts)"
    )

    import altair as alt

    # Sort Y by Rank so order is stable and intuitive
    y_sort = alt.SortField(field="Rank", order="ascending")

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Name:N", sort=y_sort, title="", axis=alt.Axis(labels=False, ticks=False)),
            x=alt.X("Score:Q", title="Points"),
            color=alt.Color("Name:N", legend=None),
            tooltip=["Rank:Q", "Name:N", "Score:Q"]
        )
    )

    name_labels = (
        alt.Chart(chart_df)
        .mark_text(align="left", dx=8, baseline="middle", fontSize=13)
        .encode(
            y=alt.Y("Name:N", sort=y_sort),
            x=alt.X("Score:Q"),
            text=alt.Text("right_label:N")
        )
    )

    st.altair_chart(bars + name_labels, use_container_width=True)

    scores_all = viz_df["Score"]
    st.caption(
        f"Field: {len(viz_df)} players • "
        f"Leader: {scores_all.max():.0f} • "
        f"Median: {scores_all.median():.0f} • "
        f"Lowest: {scores_all.min():.0f}"
    )

    # Full leaderboard table
    st.dataframe(final_lb, use_container_width=True, hide_index=True)

    st.divider()

    # -------------------------
    # Hardest drops
    # -------------------------
    st.subheader("Hardest Drops (Questions only)")
    st.caption("Hardest = lowest % correct among attempted responses.")

    hd_view = hardest_df[["drop", "question", "accuracy_pct", "attempted", "correct"]].copy()
    hd_view["accuracy_pct"] = hd_view["accuracy_pct"].map(safe_pct)

    st.dataframe(
        hd_view.rename(columns={
            "drop": "Drop",
            "question": "Question",
            "accuracy_pct": "% Correct",
            "attempted": "Attempted",
            "correct": "Correct"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Players who got the hardest drops right (PP used called out)")
    for _, r in hardest_df.iterrows():
        d = int(r["drop"])
        subset = merged[(merged["drop"] == d) & (merged["is_correct"] == 1)]
        names = sorted(subset["player_name"].unique().tolist())

        pp_used_names = sorted(
            merged[(merged["drop"] == d) & (merged["power_play"] == 1)]["player_name"]
            .unique().tolist()
        )

        st.markdown('<div class="qa-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="qa-title">Drop {d} — {r["question"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="qa-sub">Correct by {len(names)} players ({safe_pct(float(r["accuracy_pct"]))})</div>',
            unsafe_allow_html=True
        )

        if pp_used_names:
            st.write(f"🔥 PP used by: {', '.join(pp_used_names)}")
        else:
            st.write("🧊 No players used Power Play for this question.")

        st.write(", ".join(names) if names else "No one got this right.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # Easiest drops
    # -------------------------
    st.subheader("Easiest Drops (Questions only)")
    st.caption("Easiest = highest % correct among attempted responses.")

    ez_view = easiest_df[["drop", "question", "accuracy_pct", "attempted", "correct"]].copy()
    ez_view["accuracy_pct"] = ez_view["accuracy_pct"].map(safe_pct)

    st.dataframe(
        ez_view.rename(columns={
            "drop": "Drop",
            "question": "Question",
            "accuracy_pct": "% Correct",
            "attempted": "Attempted",
            "correct": "Correct"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Players who got the easiest drops right (PP used called out)")
    for _, r in easiest_df.iterrows():
        d = int(r["drop"])
        subset = merged[(merged["drop"] == d) & (merged["is_correct"] == 1)]
        names = sorted(subset["player_name"].unique().tolist())

        pp_used_names = sorted(
            merged[(merged["drop"] == d) & (merged["power_play"] == 1)]["player_name"]
            .unique().tolist()
        )

        st.markdown('<div class="qa-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="qa-title">Drop {d} — {r["question"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="qa-sub">Correct by {len(names)} players ({safe_pct(float(r["accuracy_pct"]))})</div>',
            unsafe_allow_html=True
        )

        if pp_used_names:
            st.write(f"🔥 PP used by: {', '.join(pp_used_names)}")
        else:
            st.write("🧊 No players used Power Play for this question.")

        st.write(", ".join(names) if names else "No responses.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # Unsolved drops
    # -------------------------
    st.subheader("Unsolved Drops (0 correct)")
    if len(unsolved_df) == 0:
        st.write("None — every question had at least one correct answer.")
    else:
        for _, r in unsolved_df.iterrows():
            st.markdown(f"**Drop {int(r['drop'])}** — {r['question']}")
            st.caption(f"Attempted: {int(r['attempted'])} • Correct: 0")
            st.write(f"✅ Correct option: **{r['correct_option']}**")
            st.write("")
          
with tabs[1]:
    st.header("🎯 Drop Explorer")

    labels = drop_master.apply(drop_label, axis=1).tolist()
    label_to_drop = {labels[i]: int(drop_master.iloc[i]["drop"]) for i in range(len(labels))}

    selected_label = st.selectbox("Select a question", labels, index=0)
    selected_drop = label_to_drop[selected_label]

    drow = drop_master.loc[drop_master["drop"] == selected_drop].iloc[0]
    status = drow["status"]

    st.markdown(f"### Drop {selected_drop}")
    st.write(drow["question"])
    st.write(f"✅ Correct option: **{drow['correct_option']}**")
    if status == "scrapped":
        st.caption("Status: SCRAPPED (Drop 12)")

    ds = drop_stats_df.loc[drop_stats_df["drop"] == selected_drop].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attempted", int(ds["attempted"]))
    c2.metric("Correct", int(ds["correct"]))
    c3.metric("Accuracy", safe_pct(float(ds["accuracy_pct"])))
    c4.metric("PP used", int(ds["pp_used"]))

    st.divider()

    st.markdown("#### Response distribution")
    resp_subset = merged[merged["drop"] == selected_drop].copy()

    dist = (
        resp_subset[resp_subset["attempted"] == 1]
        .groupby("response")["player_name"]
        .apply(lambda s: sorted(set(s.tolist())))
        .reset_index(name="names")
    )
    dist["Count"] = dist["names"].apply(len)
    dist["Players"] = dist["names"].apply(lambda arr: ", ".join(arr))
    dist = dist.sort_values("Count", ascending=False)
    dist["%"] = (dist["Count"] / dist["Count"].sum() * 100.0) if dist["Count"].sum() else 0.0
    dist["%"] = dist["%"].map(safe_pct)

    if dist.empty:
        st.write("No responses recorded for this drop.")
    else:
        st.dataframe(
            dist.rename(columns={"response": "Response"})[["Response", "Count", "%", "Players"]],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.markdown("#### Leaderboard up to this drop (Full)")
    lb = leaderboard_with_movement(merged, selected_drop)
    lb_view = top_full_leaderboard_view(lb, attendance_df)
    lb_view.insert(1, "Move", lb["Move"].values)
    st.dataframe(lb_view, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### Results breakdown (Correct / Wrong / No Response)")
    correct_names = []
    wrong_names = []
    noresp_names = []

    for _, r in resp_subset.iterrows():
        nm = r["player_name"]
        suffix = " (PP)" if r["power_play"] == 1 else ""
        if r["attempted"] == 0:
            noresp_names.append(nm + suffix)
        elif r["is_correct"] == 1:
            correct_names.append(nm + suffix)
        else:
            wrong_names.append(nm + suffix)

    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown(f"### ✅ Correct ({len(correct_names)})")
        st.write("\n".join(sorted(correct_names)) if correct_names else "—")
    with cB:
        st.markdown(f"### ❌ Wrong ({len(wrong_names)})")
        st.write("\n".join(sorted(wrong_names)) if wrong_names else "—")
    with cC:
        st.markdown(f"### ⏸️ No response ({len(noresp_names)})")
        st.write("\n".join(sorted(noresp_names)) if noresp_names else "—")

# =============================
# Player Explorer
# =============================
with tabs[2]:
    st.header("👤 Player Explorer")

    players = sorted(merged["player_name"].unique().tolist())
    player = st.selectbox("Select a player", players)

    p = merged[merged["player_name"] == player].sort_values("drop").copy()

    # Basic totals
    total_points = int(p["points"].sum())
    total_correct = int(p["is_correct"].sum())

    # -----------------------------
    # Accuracy = Correct / Attempted
    # IMPORTANT: exclude Drop 12 from accuracy so it doesn't distort anything
    # -----------------------------
    p_acc = p[p["drop"] != 12].copy()
    attempted_for_accuracy = int(p_acc["attempted"].sum())
    correct_for_accuracy = int(p_acc["is_correct"].sum())
    acc = (correct_for_accuracy / attempted_for_accuracy * 100.0) if attempted_for_accuracy else 0.0

    # -----------------------------
    # Power Play hit rate
    # -----------------------------
    pp_used = int(p["power_play"].sum())
    pp_hits = int(p[(p["power_play"] == 1) & (p["is_correct"] == 1)].shape[0])
    pp_hit_rate = (pp_hits / pp_used * 100.0) if pp_used else 0.0

    # -----------------------------
    # Final Rank
    # -----------------------------
    final_rank_row = final_lb_raw[final_lb_raw["player_name"] == player]
    final_rank = int(final_rank_row["rank"].iloc[0]) if not final_rank_row.empty else None

    # -----------------------------
    # Attendance = xx/38 (treat Drop 12 as attended for everyone)
    # -----------------------------
    attendance_den = int(drop_master["drop"].nunique())  # 38

    attempted_all = int(p["attempted"].sum())  # attempts across all 38 drops

    # If Drop 12 exists for this player but was not attempted, add 1 anyway
    if (p["drop"] == 12).any():
        attempted_drop12 = int(p.loc[p["drop"] == 12, "attempted"].sum())
        if attempted_drop12 == 0:
            attempted_all += 1
    else:
        # Safety fallback (shouldn't happen): still count Drop 12 as attended
        attempted_all += 1

    attendance_pct = (attempted_all / attendance_den * 100.0) if attendance_den else 0.0

    # -----------------------------
    # Tiles (2 rows, 3 columns) — prevents text truncation
    # -----------------------------
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    r1c1.metric("Final Rank", final_rank if final_rank is not None else "—")
    r1c2.metric("Total Points", total_points)
    r1c3.metric("Total Drops Correct", total_correct)

    r2c1.metric("Accuracy (Correct / Attempted)", f"{acc:.1f}%")
    r2c2.metric("Attendance", f"{attempted_all}/{attendance_den} ({attendance_pct:.0f}%)")
    r2c3.metric("PP Hit Rate", f"{pp_hit_rate:.1f}%" if pp_used else "—")

    st.divider()

    # -----------------------------
    # Drop-by-drop log
    # -----------------------------
    st.markdown("#### Drop-by-drop log")
    log = p[["drop", "question", "response", "attempted", "power_play", "is_correct", "points"]].copy()

    # visual ticks
    log["Attempted"] = np.where(log["attempted"] == 1, "✅", "❌")
    log["Correct"] = np.where(log["attempted"] == 0, "—", np.where(log["is_correct"] == 1, "✅", "❌"))
    log["PP"] = np.where(log["power_play"] == 1, "✅", "❌")

    log = log.rename(columns={"drop": "Drop", "question": "Drop text", "response": "Response", "points": "Points"})
    log = log[["Drop", "Drop text", "Response", "Attempted", "Correct", "PP", "Points"]]

    st.dataframe(log, use_container_width=True, hide_index=True)
  
# =============================
# Leaderboard Race (Replay Mode)
# =============================
with tabs[3]:
    st.header("📈 Leaderboard Race — Replay Mode")
    st.caption("Move the slider or press Play to replay how the leaderboard changed after every drop.")

    # -----------------------------
    # Controls
    # -----------------------------
    max_drop = int(drop_master["drop"].max())

    # Session state defaults
    if "race_drop" not in st.session_state:
        st.session_state.race_drop = 1
    if "race_playing" not in st.session_state:
        st.session_state.race_playing = False

    cA, cB, cC = st.columns([2, 1, 1])

    with cA:
        top_n = st.slider("Show Top N players", min_value=5, max_value=25, value=12, step=1)

    with cB:
        if st.session_state.race_playing:
            if st.button("⏸ Pause"):
                st.session_state.race_playing = False
        else:
            if st.button("▶️ Play"):
                st.session_state.race_playing = True

    with cC:
        speed = st.selectbox("Speed", ["Slow", "Normal", "Fast"], index=1)

    speed_map = {"Slow": 0.8, "Normal": 0.35, "Fast": 0.15}
    delay = speed_map[speed]

    # -----------------------------
    # Helper: leaderboard as of drop d
    # -----------------------------
    def leaderboard_asof(d: int) -> pd.DataFrame:
        tmp = merged[merged["drop"] <= d].copy()
        scores = (
            tmp.groupby("player_name", as_index=False)["points"].sum()
            .rename(columns={"points": "Score"})
        )
        scores["Rank"] = scores["Score"].rank(method="dense", ascending=False).astype(int)
        scores = scores.sort_values(["Rank", "player_name"]).reset_index(drop=True)
        return scores

    # Manual override slider (always allowed)
    current_drop = st.slider(
        "Replay drop",
        min_value=1,
        max_value=max_drop,
        value=int(st.session_state.race_drop),
        step=1,
    )
    st.session_state.race_drop = current_drop

    # -----------------------------
    # Compute current + previous standings (for movement arrows)
    # -----------------------------
    now_lb = leaderboard_asof(current_drop)
    prev_lb = leaderboard_asof(max(current_drop - 1, 1))

    now_lb = now_lb.merge(
        prev_lb[["player_name", "Rank"]].rename(columns={"Rank": "PrevRank"}),
        on="player_name",
        how="left",
    )

    now_lb["PrevRank"] = now_lb["PrevRank"].fillna(now_lb["Rank"]).astype(int)
    now_lb["Move"] = now_lb["PrevRank"] - now_lb["Rank"]  # positive = moved up

    def arrow(m):
        if m > 0:
            return f"⬆️ {m}"
        if m < 0:
            return f"⬇️ {abs(m)}"
        return "—"

    now_lb["Δ"] = now_lb["Move"].apply(arrow)

    # -----------------------------
    # Top N view + podium
    # -----------------------------
    view = now_lb.sort_values(["Rank", "player_name"]).head(top_n).copy()
    podium = now_lb[now_lb["Rank"].isin([1, 2, 3])].sort_values(["Rank", "player_name"]).copy()

    # -----------------------------
    # Podium cards
    # -----------------------------
    st.markdown("### 🏆 Podium (as of this drop)")
    p1, p2, p3 = st.columns(3)

    def names_for_rank(rk):
        x = podium[podium["Rank"] == rk]
        if x.empty:
            return "—"
        return "<br>".join([f"**{row.player_name}** — {int(row.Score)} pts" for _, row in x.iterrows()])

    p1.markdown(
        f"""
        <div style="border:1px solid rgba(245,197,66,0.30); border-radius:14px; padding:12px;
                    background: linear-gradient(90deg, rgba(245,197,66,0.25), rgba(0,0,0,0));
                    box-shadow: 0 0 18px rgba(245,197,66,0.18);">
          <div style="font-weight:900; font-size:16px;">🥇 GOLD (Rank 1)</div>
          <div style="margin-top:8px; font-size:15px;">{names_for_rank(1)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p2.markdown(
        f"""
        <div style="border:1px solid rgba(255,255,255,0.16); border-radius:14px; padding:12px;
                    background: linear-gradient(90deg, rgba(200,200,200,0.16), rgba(0,0,0,0));">
          <div style="font-weight:900; font-size:16px;">🥈 SILVER (Rank 2)</div>
          <div style="margin-top:8px; font-size:15px;">{names_for_rank(2)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p3.markdown(
        f"""
        <div style="border:1px solid rgba(205,127,50,0.25); border-radius:14px; padding:12px;
                    background: linear-gradient(90deg, rgba(205,127,50,0.18), rgba(0,0,0,0));">
          <div style="font-weight:900; font-size:16px;">🥉 BRONZE (Rank 3)</div>
          <div style="margin-top:8px; font-size:15px;">{names_for_rank(3)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------
    # Race board: ranked bars
    # -----------------------------
    st.markdown(f"### 🎬 Race Board — After Drop {current_drop}")

    import altair as alt

    chart_df = view.sort_values(["Score", "player_name"], ascending=[False, True]).copy()

    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("player_name:N", sort="-x", title=""),
            x=alt.X("Score:Q", title="Points"),
            tooltip=["Rank:Q", "player_name:N", "Score:Q", "Δ:O"],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(align="left", dx=6)
        .encode(
            y=alt.Y("player_name:N", sort="-x"),
            x=alt.X("Score:Q"),
            text=alt.Text("Δ:O"),
        )
    )

    st.altair_chart(bars + labels, use_container_width=True)

    # -----------------------------
    # Table view
    # -----------------------------
    st.markdown("#### 📋 Current Top Table")
    table = view[["Rank", "player_name", "Score", "Δ"]].rename(columns={"player_name": "Name"})
    st.dataframe(table, use_container_width=True, hide_index=True)

    # -----------------------------
    # Auto-play engine
    # -----------------------------
    import time

    if st.session_state.race_playing:
        if st.session_state.race_drop >= max_drop:
            st.session_state.race_playing = False
        else:
            time.sleep(delay)
            st.session_state.race_drop += 1
            st.rerun()

# =============================
# Answer Key
# =============================
with tabs[4]:
    st.header("📜 Answer Key")
    st.caption("All drops + correct options + actuals + response stats (responses, correct, power plays).")

    # -----------------------------
    # Build per-drop stats from merged
    # -----------------------------
    # responses = attempted sum
    # correct   = is_correct sum
    # pp_used   = power_play sum
    drop_stats = (
        merged.groupby("drop", as_index=False)
        .agg(
            responses=("attempted", "sum"),
            correct=("is_correct", "sum"),
            pp_used=("power_play", "sum"),
        )
    )

    # pp_success = (power_play == 1 AND is_correct == 1)
    pp_success_df = (
        merged.assign(pp_success_row=((merged["power_play"] == 1) & (merged["is_correct"] == 1)).astype(int))
        .groupby("drop", as_index=False)
        .agg(pp_success=("pp_success_row", "sum"))
    )

    # merge stats together
    drop_stats = drop_stats.merge(pp_success_df, on="drop", how="left")
    drop_stats["pp_success"] = drop_stats["pp_success"].fillna(0).astype(int)

    # -----------------------------
    # Join with drop_master (show ALL drops)
    # -----------------------------
    # NOTE: assumes drop_master has columns: drop, question, correct_option, actuals, status
    ak = (
        drop_master[["drop", "question", "correct_option", "actuals", "status"]]
        .merge(drop_stats, on="drop", how="left")
        .sort_values("drop")
        .copy()
    )

    # fill blanks (if a drop has no rows in merged)
    for col in ["responses", "correct", "pp_used", "pp_success"]:
        ak[col] = ak[col].fillna(0).astype(int)

    # -----------------------------
    # CSS
    # -----------------------------
    st.markdown(
        """
        <style>
          .ak-grid{
            display:grid;
            grid-template-columns: repeat(3, minmax(260px, 1fr));
            gap: 14px;
            margin-top: 12px;
          }
          @media (max-width: 1100px){
            .ak-grid{ grid-template-columns: repeat(2, minmax(240px, 1fr)); }
          }
          @media (max-width: 700px){
            .ak-grid{ grid-template-columns: 1fr; }
          }

          .ak-tile{
            border-radius: 16px;
            padding: 14px 14px 12px 14px;
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            box-shadow: 0 12px 26px rgba(0,0,0,0.38);
            min-height: 190px;
          }

          .ak-tile.scrapped{
            border-color: rgba(255,110,110,0.70);
            background: linear-gradient(180deg, rgba(255,70,70,0.35), rgba(255,70,70,0.12));
          }

          .ak-drop{
            font-weight: 950;
            font-size: 18px;
            color: #f5c542;
            letter-spacing: 0.04em;
          }

          .ak-q{
            font-size: 14px;
            opacity: 0.95;
            line-height: 1.3;
            margin: 8px 0 10px 0;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
            min-height: 54px;
          }

          .ak-pill{
            display:inline-block;
            margin: 2px 8px 10px 0;
            font-weight: 900;
            font-size: 14px;
            padding: 7px 10px;
            border-radius: 11px;
            background: rgba(245,197,66,0.10);
            border: 1px solid rgba(245,197,66,0.22);
          }

          .ak-pill.actual{
            background: rgba(120,200,255,0.08);
            border: 1px solid rgba(120,200,255,0.18);
          }

          .ak-metrics{
            display:flex;
            justify-content:space-between;
            gap: 12px;
            margin-top: 8px;
            font-size: 14px;
            opacity: 0.95;
          }
          .ak-pair{
            display:flex;
            gap: 10px;
            align-items:baseline;
            flex-wrap: wrap;
          }
          .ak-pair b{ font-weight: 950; }
          .ak-ppgood{ color:#ffd88a; font-weight: 900; }
          .ak-scraplabel{ color:#ffe1e1; font-weight: 950; margin: 2px 0 6px 0; }
          .ak-sep{ opacity:0.7; padding: 0 6px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Render tiles
    # -----------------------------
    st.markdown('<div class="ak-grid">', unsafe_allow_html=True)

    for _, r in ak.iterrows():
        d = int(r["drop"])
        q = str(r["question"])
        ans = str(r["correct_option"])
        actuals = "" if pd.isna(r["actuals"]) else str(r["actuals"]).strip()
        status = str(r["status"]).lower()

        responses = int(r["responses"])
        correct = int(r["correct"])
        pp_used = int(r["pp_used"])
        pp_success = int(r["pp_success"])

        tile_class = "ak-tile scrapped" if status == "scrapped" else "ak-tile"
        scrap_line = '<div class="ak-scraplabel">🗑️ SCRAPPED DROP</div>' if status == "scrapped" else ""

        # If actuals empty, show a dash
        actual_text = actuals if actuals else "—"

        tile_html = (
            f'<div class="{tile_class}">'
            f'  <div class="ak-drop">Drop {d}</div>'
            f'  {scrap_line}'
            f'  <div class="ak-q">{q}</div>'
            f'  <div>'
            f'    <span class="ak-pill">Answer: {ans}</span>'
            f'    <span class="ak-pill actual">Actual: {actual_text}</span>'
            f'  </div>'
            f'  <div class="ak-metrics">'
            f'    <div class="ak-pair">'
            f'      <span>Responses:</span><b>{responses}</b>'
            f'      <span class="ak-sep">|</span>'
            f'      <span>Correct:</span><b>{correct}</b>'
            f'    </div>'
            f'    <div class="ak-pair">'
            f'      <span>PP used:</span><b>{pp_used}</b>'
            f'      <span class="ak-sep">|</span>'
            f'      <span class="ak-ppgood">PP success:</span><b class="ak-ppgood">{pp_success}</b>'
            f'    </div>'
            f'  </div>'
            f'</div>'
        )

        st.markdown(tile_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)









































