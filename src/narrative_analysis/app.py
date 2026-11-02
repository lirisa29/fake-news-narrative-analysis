import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import networkx as nx

from pyvis.network import Network
from collections import defaultdict

from narrative_analysis.graph_builder import build_knowledge_graph


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(layout="wide")

st.title("Fake News Narrative Intelligence Dashboard")

st.caption(
    "Explore semantic narrative clusters, relationships and supporting evidence. Hover over a relationship arrow to inspect its supporting article evidence."
)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    tuples = pd.read_csv("data/processed/tuples.csv")

    required_tuple_columns = [
        "tuple_id",
        "subject",
        "relationship",
        "object",
        "article_id",
        "source",
        "date",
        "sentence"
    ]

    missing = [
        col for col in required_tuple_columns
        if col not in tuples.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in tuples.csv: {missing}"
        )

    tuples["date"] = pd.to_datetime(
        tuples["date"],
        utc=True,
        errors="coerce"
    )

    # Remove dates outside the expected dataset period
    tuples.loc[
        ~tuples["date"].dt.year.between(2016, 2020),
        "date"
    ] = pd.NaT

    return tuples


tuples = load_data()


# ==========================================================
# 3. SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Dashboard Filters")

sources = sorted(
    tuples["source"].dropna().unique()
)

selected_sources = st.sidebar.multiselect(
    "Select sources",
    sources,
    default=sources
)

valid_dates = tuples["date"].dropna()

if valid_dates.empty:

    st.error("No valid dates found in tuples.csv.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

start_date = st.sidebar.date_input(
    "Start date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

if start_date > end_date:

    st.sidebar.error(
        "Start date must be before end date."
    )

    st.stop()

max_edges = st.sidebar.select_slider(
    "Maximum graph relationships",
    options=[250, 500, 750, 1000, 1500, 2000],
    value=500
)

max_evidence = 3


# ==========================================================
# 4. FILTER DATA
# ==========================================================

filtered = tuples[
    (tuples["source"].isin(selected_sources))
    &
    (tuples["date"].dt.date >= start_date)
    &
    (tuples["date"].dt.date <= end_date)
].copy()


# ==========================================================
# 5. BUILD KNOWLEDGE GRAPH
# ==========================================================

G = build_knowledge_graph(
    filtered,
    max_edges=max_edges,
    max_evidence=max_evidence
)


# ==========================================================
# 6. CREATE KNOWLEDGE GRAPH
# ==========================================================

def create_knowledge_graph(data):

    if data.empty:

        st.warning(
            "No data matches the selected filters."
        )

        return

    G = build_knowledge_graph(
        data,
        max_edges=max_edges
    )

    if G.number_of_nodes() == 0:

        st.warning(
            "No relationships available."
        )

        return

    # ------------------------------------------------------
    # GRAPH INFORMATION
    # ------------------------------------------------------

    st.caption(
        f"Displaying {G.number_of_nodes():,} entities "
        f"and {G.number_of_edges():,} relationships "
        f"from {len(data):,} filtered tuples."
    )

    # ------------------------------------------------------
    # INITIALISE PYVIS
    # ------------------------------------------------------

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True,
        notebook=False
    )

    net.from_nx(G)

    # ------------------------------------------------------
    # NODE STYLING
    # ------------------------------------------------------

    for node in net.nodes:

        node["shape"] = "dot"
        node["size"] = 25
        node["color"] = "#1f77b4"

    # ------------------------------------------------------
    # EDGE STYLING
    # ------------------------------------------------------

    for edge in net.edges:

        edge["arrows"] = "to"
        edge["font"] = {
            "size": 12,
            "align": "middle"
        }

    # ------------------------------------------------------
    # OPTIMISED PHYSICS
    # ------------------------------------------------------
    
    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "solver": "barnesHut",

            "barnesHut": {
                "gravitationalConstant": -10000,
                "centralGravity": 0.2,
                "springLength": 150,
                "springConstant": 0.025,
                "damping": 0.9,
                "avoidOverlap": 0.2
            },

            "minVelocity": 0.5,

            "stabilization": {
                "enabled": true,
                "iterations": 500,
                "updateInterval": 50,
                "fit": true
            },

            "timestep": 0.25,
            "adaptiveTimestep": true
        },

        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        },

        "edges": {
            "smooth": {
                "enabled": false
            }
        }
    }
    """)


    # ------------------------------------------------------
    # DISPLAY GRAPH
    # ------------------------------------------------------

    try:

        graph_html = net.generate_html()

        # Customise hover tooltip appearance
        tooltip_css = """
        <style>
        .vis-tooltip {
            max-width: 500px !important;
            max-height: 350px !important;

            white-space: pre-wrap !important;
            overflow-wrap: anywhere !important;
            word-break: normal !important;

            overflow-y: auto !important;
            overflow-x: hidden !important;

            padding: 12px !important;

            font-size: 13px !important;
            line-height: 1.6 !important;

            background-color: #ffffff !important;
            color: #222222 !important;

            border: 1px solid #cccccc !important;
            border-radius: 8px !important;

            box-shadow: 0 3px 12px rgba(0,0,0,0.15) !important;
        }
        </style>
        """

        graph_html = graph_html.replace(
            "</head>",
            tooltip_css + "</head>"
        )

        components.html(
            graph_html,
            height=620,
            scrolling=True
        )

    except Exception as e:

        st.error(
            f"Error drawing graph: {e}"
        )


# ==========================================================
# 7. MAIN DASHBOARD
# ==========================================================

st.divider()

create_knowledge_graph(filtered)