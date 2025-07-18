import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from time import sleep

# ------------------------------------------------------------
#  Page & theme config
# ------------------------------------------------------------
st.set_page_config(
    page_title="IDS Pro",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
#  Cached helpers
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    with open("encoders.pkl", "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
    scaler = load_scaler()
    label_encoders = load_encoders()
    EXPECTED_FEATURES = (
        model.feature_names_in_.tolist()
        if hasattr(model, "feature_names_in_")
        else [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
        ]
    )
except Exception as e:
    st.error(f"Model files not found: {e}")
    st.stop()

# ------------------------------------------------------------
#  Session state
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = {}

# ------------------------------------------------------------
#  Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("🛠 IDS Pro")
page = st.sidebar.radio("", ["📊 Dashboard", "📤 Upload", "ℹ About"], label_visibility="collapsed")

# ------------------------------------------------------------
#  ABOUT PAGE
# ------------------------------------------------------------
if page == "ℹ About":
    st.header("About IDS Pro")
    st.markdown(
        """
    *IDS Pro* is a modern, real-time intrusion-detection analytics platform.

    - Built with Python, Streamlit & Scikit-learn  
    - Supports live traffic simulation, rich visualizations and downloadable reports  
    - Fully customizable and production-ready
    """
    )

# ------------------------------------------------------------
#  UPLOAD PAGE
# ------------------------------------------------------------
elif page == "📤 Upload":
    st.header("📤 Upload New Data")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        submit = st.form_submit_button("Upload & Process")
    if submit and uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # ------- NEW SAFEGUARD -------
            if "attack_category" not in df.columns:
                # create a fallback based on the optional label column, else "unknown"
                df["attack_category"] = np.where(
                    df.get("label", "normal") == "normal",
                    "normal",
                    df.get("label", "unknown"),
                )
            # ------------------------------

            missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

            # Encode categoricals
            df_proc = df[EXPECTED_FEATURES].copy()
            for col in ["protocol_type", "service", "flag"]:
                if col in label_encoders:
                    le = label_encoders[col]
                    df_proc[col] = df_proc[col].apply(
                        lambda x: x if x in le.classes_ else "unknown"
                    )
                    df_proc[col] = le.transform(df_proc[col])

            X = scaler.transform(df_proc)
            preds = model.predict(X)
            proba = (
                model.predict_proba(X).max(axis=1)
                if hasattr(model, "predict_proba")
                else np.nan * np.ones_like(preds)
            )

            enriched = df.assign(
                Predicted_Label=np.where(preds == 0, "normal", "attack"),
                Predicted_Label_Num=preds,
                Confidence=proba,
            )
            st.session_state.history[uploaded_file.name] = enriched
            st.success("File processed successfully!")
        except Exception as e:
            st.exception(e)
# ------------------------------------------------------------
#  DASHBOARD PAGE
# ------------------------------------------------------------
# ------------------------------------------------------------
#  DASHBOARD PAGE
# ------------------------------------------------------------
else:
    st.header("📊 Intrusion Detection Dashboard")

    # ---------------------------------------------------------
    #  DUMMY DASHBOARD when no data is available
    # ---------------------------------------------------------
    if not st.session_state.history:
        st.info("No data uploaded yet.  Upload a CSV to see live analytics.")

        # --- Fake KPI row ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total records", "—")
        col2.metric("Attack rate", "— %")
        col3.metric("Dataset", "—")

        # --- Fake charts ---
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Attack vs Normal (placeholder)")
            fig_dummy_pie = px.pie(
                names=["normal", "attack"],
                values=[75, 25],
                hole=0.4,
                color_discrete_map={"normal": "#2ca02c", "attack": "#d62728"},
                title="Example distribution"
            )
            st.plotly_chart(fig_dummy_pie, use_container_width=True)

        with col_b:
            st.subheader("Attacks by Protocol (placeholder)")
            dummy_bar = pd.DataFrame({
                "protocol_type": ["tcp", "udp", "icmp"],
                "count": [120, 80, 30]
            })
            fig_dummy_bar = px.bar(
                dummy_bar,
                x="count",
                y="protocol_type",
                orientation="h",
                color="count",
                color_continuous_scale="Reds",
                title="Example protocol breakdown"
            )
            st.plotly_chart(fig_dummy_bar, use_container_width=True)

        # --- Placeholder live plot ---
        st.subheader("📉 Real-time Attack Detection (placeholder)")
        dummy_ts = pd.DataFrame({
            "step": range(20),
            "label": np.random.choice([0, 1], 20, p=[0.8, 0.2])
        })
        fig_dummy_live = px.scatter(
            dummy_ts,
            x="step",
            y="label",
            color="label",
            color_discrete_map={0: "green", 1: "red"},
            title="Simulated live traffic"
        )
        fig_dummy_live.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_dummy_live, use_container_width=True)

        st.stop()   # << skip the rest of the dashboard
    # ---------------------------------------------------------
    #  REAL DASHBOARD starts here (only runs after upload)
    # ---------------------------------------------------------
    ...

    # --- Sidebar filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    data_name = st.sidebar.selectbox("Dataset", list(st.session_state.history.keys()))
    df = st.session_state.history[data_name]

    # Build attack list with categories
    attack_list = (
        df[df["Predicted_Label"] == "attack"]["attack_category"]
        .dropna()
        .unique()
        .tolist()
    )
    attack_type_options = ["normal"] + attack_list

    proto_filter = st.sidebar.multiselect(
        "Protocol", df["protocol_type"].unique(), df["protocol_type"].unique()
    )
    service_filter = st.sidebar.multiselect(
        "Service", df["service"].unique(), df["service"].unique()
    )

    search = st.sidebar.text_input("Search IP / flag")
    live_plot_toggle = st.sidebar.checkbox("Live intrusion plot")

    # Apply filters
    filtered = df[
        (df["Predicted_Label"].isin(["normal", "attack"]))
        & (df["protocol_type"].isin(proto_filter))
        & (df["service"].isin(service_filter))
    ]
    if attack_filter != attack_type_options:
        filtered = filtered[
            (filtered["Predicted_Label"] == "normal")
            | (filtered["attack_category"].isin(attack_filter))
        ]
    if search:
        mask = (
            filtered.astype(str)
            .apply(lambda x: x.str.contains(search, case=False))
            .any(axis=1)
        )
        filtered = filtered[mask]

    # ------------------------------------------------------------
    #  KPI
    # ------------------------------------------------------------
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total records", len(filtered))
    kpi2.metric("Attack rate", f"{(filtered['Predicted_Label_Num'].mean()*100):.1f}%")
    kpi3.metric("Dataset", data_name)
    if filtered["Predicted_Label_Num"].mean() > 0.5:
        st.warning("⚠ High attack rate detected!")

    # ------------------------------------------------------------
    #  PLOTS
    # ------------------------------------------------------------
    col1, col2 = st.columns(2)

    # 1) Pie chart
    with col1:
        st.subheader("Attack vs Normal")
        pie = px.pie(
            filtered,
            names="Predicted_Label",
            hole=0.4,
            color_discrete_map={"normal": "green", "attack": "red"},
        )
        st.plotly_chart(pie, use_container_width=True)

    # 2) Bar chart by protocol
    with col2:
        st.subheader("Attacks by Protocol")
        bar_df = (
            filtered[filtered["Predicted_Label"] == "attack"]
            .groupby("protocol_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=True)
            .tail(10)
        )
        bar = px.bar(
            bar_df,
            x="count",
            y="protocol_type",
            orientation="h",
            color="count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(bar, use_container_width=True)

    # ------------------------------------------------------------------
    #  LIVE INTRUSION FLUCTUATION  (with Pause/Resume)
    # ------------------------------------------------------------------
    if live_plot_toggle:
        st.subheader("📉 Real-time Attack Detection (Last 20 Connections)")

        # --- Pause / Resume control ---
        if "live_paused" not in st.session_state:
            st.session_state.live_paused = False

        col_pause, _ = st.columns([1, 5])
        with col_pause:
            if st.button("⏸ Pause" if not st.session_state.live_paused else "▶ Resume"):
                st.session_state.live_paused = not st.session_state.live_paused
                st.rerun()  # force redraw with new state

        live_plot = st.empty()
        window_size = 20
        total_points = len(filtered)

        # If paused, just show the latest static snapshot
        if st.session_state.live_paused:
            window_data = filtered.tail(window_size)
            fig = go.Figure()

            # Lines
            for j in range(1, len(window_data)):
                prev = window_data["Predicted_Label_Num"].iloc[j - 1]
                curr = window_data["Predicted_Label_Num"].iloc[j]
                color = "red" if curr == 1 else "green"
                fig.add_trace(
                    go.Scatter(
                        x=[j - 1, j],
                        y=[prev, curr],
                        mode="lines",
                        line=dict(color=color, width=3),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            # Hover text with attack category
            hover_text = [
                f"Connection {idx}<br>"
                f"Status: {'Attack' if y==1 else 'Normal'}<br>"
                f"Category: {row.attack_category}<br>"
                f"Protocol: {row.protocol_type}"
                for idx, (y, row) in enumerate(
                    zip(
                        window_data["Predicted_Label_Num"],
                        window_data.itertuples(index=False),
                    )
                )
            ]

            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(window_data)),
                    y=window_data["Predicted_Label_Num"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=[
                            "red" if x == 1 else "green"
                            for x in window_data["Predicted_Label_Num"]
                        ],
                        line=dict(width=1, color="DarkSlateGrey"),
                    ),
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    name="Status",
                )
            )
            fig.update_layout(
                xaxis=dict(range=[0, window_size - 1], title="Connection Sequence"),
                yaxis=dict(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["Normal", "Attack"]),
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            live_plot.plotly_chart(fig, use_container_width=True)

        else:
            # ---- ANIMATED LOOP ----
            for i in range(window_size, min(total_points, 100)):
                window_data = filtered.iloc[i - window_size : i]

                fig = go.Figure()
                # Lines
                for j in range(1, len(window_data)):
                    prev = window_data["Predicted_Label_Num"].iloc[j - 1]
                    curr = window_data["Predicted_Label_Num"].iloc[j]
                    color = "red" if curr == 1 else "green"
                    fig.add_trace(
                        go.Scatter(
                            x=[j - 1, j],
                            y=[prev, curr],
                            mode="lines",
                            line=dict(color=color, width=3),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                # Hover text with attack category
                hover_text = [
                    f"Connection {idx}<br>"
                    f"Status: {'Attack' if y==1 else 'Normal'}<br>"
                    f"Category: {row.attack_category}<br>"
                    f"Protocol: {row.protocol_type}"
                    for idx, (y, row) in enumerate(
                        zip(
                            window_data["Predicted_Label_Num"],
                            window_data.itertuples(index=False),
                        )
                    )
                ]

                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(window_data)),
                        y=window_data["Predicted_Label_Num"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=[
                                "red" if x == 1 else "green"
                                for x in window_data["Predicted_Label_Num"]
                            ],
                            line=dict(width=1, color="DarkSlateGrey"),
                        ),
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        name="Status",
                    )
                )
                fig.update_layout(
                    xaxis=dict(range=[0, window_size - 1], title="Connection Sequence"),
                    yaxis=dict(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["Normal", "Attack"]),
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                live_plot.plotly_chart(fig, use_container_width=True)
                sleep(0.5)

    # 4) Interactive duration histogram
    st.subheader("Connection Duration (interactive)")
    hist = px.histogram(
        filtered,
        x="duration",
        color="Predicted_Label",
        marginal="box",
        hover_data=filtered.columns,
        nbins=50,
        color_discrete_map={"normal": "green", "attack": "red"},
    )
    hist.update_layout(bargap=0.1)
    st.plotly_chart(hist, use_container_width=True)

    # 5) Top-10 numeric correlation heatmap
    st.subheader("Top numeric feature correlations")
    num_cols = filtered.select_dtypes(include=np.number).columns
    top_num = (
        pd.Series(model.feature_importances_, index=EXPECTED_FEATURES)
        .nlargest(10)
        .index.intersection(num_cols)
    )
    if len(top_num) > 1:
        corr = filtered[top_num].corr()
        heat = px.imshow(
            corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(heat, use_container_width=True)

    # 6) Feature importance
    with st.expander("Interactive feature importance"):
        n = st.slider("Top N features", 5, 30, 15)
        fi_df = (
            pd.DataFrame(
                {"Feature": EXPECTED_FEATURES, "Importance": model.feature_importances_}
            )
            .sort_values("Importance", ascending=True)
            .tail(n)
        )
        fi_plot = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Bluered",
        )
        st.plotly_chart(fi_plot, use_container_width=True)

    # ------------------------------------------------------------
    #  DOWNLOADS & RAW TABLE
    # ------------------------------------------------------------
    csv = filtered.to_csv(index=False)
    st.sidebar.download_button(
        "⬇ Download filtered CSV",
        data=csv,
        file_name="alerts.csv",
        mime="text/csv",
    )
    with st.expander("Show raw data"):
        st.dataframe(filtered)
