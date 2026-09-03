"""
frontend/styles.py

Charcoal/slate + teal/emerald/amber/coral/purple "marine intelligence /
defense-tech" theme. Deliberately NOT a blue-dominant ocean template.
"""

CUSTOM_CSS = """
<style>
:root {
    --bg: #101418;
    --panel: #1B2228;
    --panel-hover: #212A31;
    --border: #2A343C;
    --teal: #16A6A0;
    --emerald: #35B779;
    --amber: #F2B84B;
    --coral: #E76F51;
    --purple: #8B6FD8;
    --text: #F2F4F3;
    --text2: #A8B0B5;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; letter-spacing: -0.01em; }
p, span, label, div { color: var(--text); }
small, .caption, .ss-muted { color: var(--text2) !important; }

.ss-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 4px 18px 4px; border-bottom: 1px solid var(--border); margin-bottom: 14px;
}
.ss-logo {
    width: 40px; height: 40px; border-radius: 9px;
    background: rgba(22,166,160,.14); border: 1px solid rgba(22,166,160,.4);
    display: flex; align-items: center; justify-content: center;
    color: var(--teal); font-weight: 800; font-size: 16px; flex-shrink: 0;
}
.ss-title { font-weight: 700; font-size: 18px; letter-spacing: -0.02em; color: var(--text); }
.ss-subtitle { font-size: 12px; color: var(--text2); }

.ss-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 18px; margin-bottom: 14px;
}
.ss-stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--text2); }
.ss-stat-value { font-size: 28px; font-weight: 700; margin-top: 4px; color: var(--text); }

.ss-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700;
    border: 1px solid;
}
.ss-badge-low       { color: var(--emerald); border-color: rgba(53,183,121,.4); background: rgba(53,183,121,.12); }
.ss-badge-medium    { color: var(--amber);   border-color: rgba(242,184,75,.4); background: rgba(242,184,75,.12); }
.ss-badge-high, .ss-badge-critical { color: var(--coral); border-color: rgba(231,111,81,.4); background: rgba(231,111,81,.12); }
.ss-badge-unknown   { color: var(--purple);  border-color: rgba(139,111,216,.45); background: rgba(139,111,216,.14); }
.ss-badge-artificial{ color: var(--coral);   border-color: rgba(231,111,81,.4); background: rgba(231,111,81,.10); }
.ss-badge-natural   { color: var(--emerald); border-color: rgba(53,183,121,.4); background: rgba(53,183,121,.10); }
.ss-badge-uncertain { color: var(--amber);   border-color: rgba(242,184,75,.4); background: rgba(242,184,75,.10); }
.ss-badge-mode      { color: var(--teal);    border-color: rgba(22,166,160,.4); background: rgba(22,166,160,.12); }

.ss-pipeline {
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12px; color: var(--text2);
}
.ss-pipeline .step {
    background: var(--panel-hover); border: 1px solid var(--border); border-radius: 8px;
    padding: 6px 10px; color: var(--text);
}
.ss-pipeline .arrow { color: var(--teal); }

.stButton>button {
    background: var(--teal); color: #06201E; border: none; border-radius: 8px;
    font-weight: 700; padding: 0.55em 1.1em;
}
.stButton>button:hover { background: #12908B; color: #06201E; }

div[data-testid="stFileUploaderDropzone"] {
    background: var(--panel); border: 1.5px dashed var(--border); border-radius: 12px;
}

.ss-status-note {
    font-size: 12px; color: var(--text2); background: var(--panel-hover);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin-top: 6px;
}
</style>
"""
