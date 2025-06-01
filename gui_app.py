import streamlit as st
import threading
import time
import os
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

from scapy.all import sniff, IP, TCP, UDP

# File paths
ROOT = "C:\\Users\\majum\\OneDrive\\Pictures\\RealTime_IDS"
MODEL_DIR = os.path.join(ROOT, "Dataset/Phase2_Models")
MODEL_PATH = os.path.join(MODEL_DIR, "model_randomforest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.txt")

FLOW_TIMEOUT = 5

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

active_flows = defaultdict(list)
flow_lock = threading.Lock()
detection_logs = []

# Feature extraction
def extract_features(flow_packets):
    if len(flow_packets) < 1:
        return None

    start_time = flow_packets[0]['time']
    end_time = flow_packets[-1]['time']
    duration = end_time - start_time if end_time > start_time else 1e-6

    fwd_packets = [pkt for pkt in flow_packets if pkt['direction'] == 'fwd']
    bwd_packets = [pkt for pkt in flow_packets if pkt['direction'] == 'bwd']

    total_fwd_pkts = len(fwd_packets)
    total_bwd_pkts = len(bwd_packets)
    total_fwd_bytes = sum(pkt['length'] for pkt in fwd_packets)
    total_bwd_bytes = sum(pkt['length'] for pkt in bwd_packets)
    all_lengths = [pkt['length'] for pkt in flow_packets]
    pkt_len_mean = np.mean(all_lengths) if all_lengths else 0
    pkt_per_sec = (total_fwd_pkts + total_bwd_pkts) / duration

    features = {
        "Flow Duration": duration,
        "Total Fwd Packets": total_fwd_pkts,
        "Total Backward Packets": total_bwd_pkts,
        "Total Length of Fwd Packets": total_fwd_bytes,
        "Total Length of Bwd Packets": total_bwd_bytes,
        "Average Packet Length": pkt_len_mean,
        "Flow Packets/s": pkt_per_sec
    }

    return features

# Predict function
def predict_flow(flow_key, flow_packets):
    features = extract_features(flow_packets)
    if not features:
        return

    row = pd.DataFrame([features])
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]

    X_scaled = scaler.transform(row)
    prediction = model.predict(X_scaled)[0]

    now = datetime.now().strftime('%H:%M:%S')
    if prediction == 1:
        msg = f"🚨 ALERT: Intrusion detected on flow {flow_key} at {now}"
    else:
        msg = f"✅ Normal flow {flow_key} at {now}"

    detection_logs.append(msg)

# Expired flow monitoring
def monitor_flows():
    while st.session_state.running:
        now = time.time()
        expired = []

        with flow_lock:
            for key, pkts in active_flows.items():
                if len(pkts) == 0:
                    continue
                last_seen = pkts[-1]['time']
                if now - last_seen > FLOW_TIMEOUT:
                    expired.append(key)

        for key in expired:
            with flow_lock:
                pkts = active_flows.pop(key)
            predict_flow(key, pkts)

        time.sleep(1)

# Packet handler
def handle_packet(pkt):
    if not IP in pkt:
        return

    proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "OTHER"
    if proto == "OTHER":
        return

    ip_layer = pkt[IP]
    sport = pkt.sport if TCP in pkt or UDP in pkt else 0
    dport = pkt.dport if TCP in pkt or UDP in pkt else 0

    flow_key = (ip_layer.src, ip_layer.dst, sport, dport, proto)
    reverse_key = (ip_layer.dst, ip_layer.src, dport, sport, proto)

    pkt_data = {
        'time': time.time(),
        'length': len(pkt),
        'direction': 'fwd'
    }

    with flow_lock:
        if reverse_key in active_flows:
            pkt_data['direction'] = 'bwd'
            active_flows[reverse_key].append(pkt_data)
        else:
            active_flows[flow_key].append(pkt_data)

# Start IDS system
def start_ids():
    st.session_state.running = True
    threading.Thread(target=monitor_flows, daemon=True).start()
    sniff(prn=handle_packet, store=False, stop_filter=lambda x: not st.session_state.running)

# Stop IDS system
def stop_ids():
    st.session_state.running = False

# ------------------ Streamlit GUI -------------------
st.set_page_config(page_title="Real-Time IDS", layout="wide")

if "running" not in st.session_state:
    st.session_state.running = False

st.title("🔐 Real-Time Intrusion Detection System (RandomForest)")

col1, col2 = st.columns(2)
with col1:
    if not st.session_state.running:
        if st.button("▶️ Start IDS"):
            threading.Thread(target=start_ids, daemon=True).start()
    else:
        if st.button("⏹️ Stop IDS"):
            stop_ids()

with col2:
    st.markdown(f"**Status:** {'🟢 Running' if st.session_state.running else '🔴 Stopped'}")

st.subheader("📋 Detection Logs")
st.text_area("Real-time Alerts", "\n".join(detection_logs[-30:]), height=400)
