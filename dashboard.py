"""
Streamlit dashboard — read-only SQLite (Community Cloud safe).
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "tnd.db"


@st.cache_data(ttl=60)
def load_frames(history_days: int):
    if not DB_PATH.exists():
        return None, None
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        # Get historical fx_rates
        q_fx = f"""
        SELECT date, fix_mid, ib_rate
        FROM fx_rates
        WHERE date >= date('now', '-{history_days} days')
        ORDER BY date ASC
        """
        fx_df = pd.read_sql_query(q_fx, conn)
        fx_df["date"] = pd.to_datetime(fx_df["date"])
        
        # Get predictions
        q_pred = """
        SELECT date, intrinsic_v1, intrinsic_v2, w_eurusd, w_gbpusd, w_usdjpy, kf_spread
        FROM predictions
        ORDER BY date ASC
        """
        pred_df = pd.read_sql_query(q_pred, conn)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        
        # Merge
        df = fx_df.merge(pred_df, on="date", how="left")
        df["spread_ib_fix"] = df["ib_rate"] - df["fix_mid"]
        
        # Get the last valid fix_mid
        q_last_fix = "SELECT fix_mid FROM fx_rates WHERE fix_mid IS NOT NULL ORDER BY date DESC LIMIT 1"
        last_fix_df = pd.read_sql_query(q_last_fix, conn)
        last_fix = float(last_fix_df.iloc[0]["fix_mid"]) if not last_fix_df.empty else None
        
        return df, last_fix
    finally:
        conn.close()


def main():
    st.set_page_config(page_title="USD/TND FX Model", layout="wide")
    st.title("USD/TND - Daily FX Model")

    history_days = 90  # Fixed window

    df, last_fix = load_frames(history_days)
    if df is None or df.empty:
        st.warning(
            "No data in SQLite. Run the pipeline locally and commit `data/tnd.db`, "
            "or seed historical `fx_rates` first."
        )
        return

    last = df.iloc[-1]
    pred = float(last["intrinsic_v2"]) if pd.notna(last.get("intrinsic_v2")) else None
    prev_fix = last_fix
    cur_fix = float(last["fix_mid"]) if pd.notna(last.get("fix_mid")) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Today's prediction (intrinsic_v2)",
        f"{pred:.4f}" if pred is not None else "N/A",
    )
    c2.metric(
        "Latest BCT fixing",
        f"{prev_fix:.4f}" if prev_fix is not None else "N/A",
    )
    chg = None
    if pred is not None and prev_fix is not None and prev_fix != 0:
        chg = (pred / prev_fix - 1.0) * 100.0
    c3.metric("Est. change vs latest fix", f"{chg:.3f}%" if chg is not None else "N/A")
    c4.metric("Model R²", "see Excel / logs")

    h180 = df.tail(180)
    melt = h180.melt(
        id_vars=["date"],
        value_vars=["fix_mid", "intrinsic_v2", "ib_rate"],
        var_name="series",
        value_name="value",
    )
    fig = px.line(
        melt,
        x="date",
        y="value",
        color="series",
        title="Intrinsic value vs BCT fixing",
        labels={"value": "TND per USD", "date": "Date"},
        color_discrete_map={
            "fix_mid": "#1f77b4",
            "intrinsic_v2": "#ff7f0e",
            "ib_rate": "#7f7f7f",
        },
    )
    for tr in fig.data:
        if getattr(tr, "name", None) == "ib_rate":
            tr.line.dash = "dash"
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    tail = df.tail(history_days)
    wtail = df.dropna(subset=["w_eurusd", "w_gbpusd", "w_usdjpy"]).tail(90)[["date", "w_eurusd", "w_gbpusd", "w_usdjpy"]]
    with left:
        if not wtail.empty:
            wm = wtail.melt(id_vars=["date"], var_name="weight", value_name="v")
            fig2 = px.line(wm, x="date", y="v", color="weight", title="Rolling weights")
            st.plotly_chart(fig2, use_container_width=True)
    with right:
        sp = tail.dropna(subset=["spread_ib_fix"])
        if not sp.empty:
            col = sp["spread_ib_fix"].apply(lambda x: "IB > Fix" if x > 0 else "IB <= Fix")
            fig3 = px.bar(
                sp,
                x="date",
                y="spread_ib_fix",
                color=col,
                color_discrete_map={"IB > Fix": "red", "IB <= Fix": "blue"},
                title="Spread (IB - Fix)",
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Latest predictions (30 rows)")
    show = df.tail(30)[
        ["date", "fix_mid", "intrinsic_v1", "intrinsic_v2", "spread_ib_fix"]
    ].copy()
    show = show.rename(
        columns={
            "fix_mid": "BCT Fix",
            "intrinsic_v1": "Intrinsic V1",
            "intrinsic_v2": "Intrinsic V2",
            "spread_ib_fix": "Spread",
        }
    )
    show["R²"] = ""
    st.dataframe(show, use_container_width=True)

    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download visible table as CSV",
        data=csv,
        file_name="tnd_dashboard_export.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
