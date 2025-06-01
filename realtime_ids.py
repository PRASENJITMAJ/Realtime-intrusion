import os
import time
import joblib
import threading
from collections import defaultdict
from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import numpy as np

ROOT = "C:\\Users\\majum\\OneDrive\\Pictures\\RealTime_IDS"
MODEL_DIR = os.path.join(ROOT, "Dataset", "Phase2_Models")
FLOW_TIMEOUT = 5

scaler = None
pca = None
rbf_sampler = None
encoder_dim = 32

def is_classifier_model(path):
    try:
        obj = joblib.load(os.path.join(MODEL_DIR, path))
        return hasattr(obj, "predict")
    except:
        return False

available_models = [
    f for f in os.listdir(MODEL_DIR)
    if f.endswith('.pkl') and "model" in f.lower() and is_classifier_model(f)
]

print("📦 Available Models:")
for i, model_file in enumerate(available_models):
    print(f"{i+1}. {model_file}")
choice = int(input("🔍 Select a model to use (enter the number): ")) - 1
model_file = available_models[choice]
model = joblib.load(os.path.join(MODEL_DIR, model_file))

base_name = model_file.replace(".pkl", "").replace("model_", "").replace("_model", "")
scaler_path = os.path.join(MODEL_DIR, f"scaler_{base_name}.pkl")
features_path = os.path.join(MODEL_DIR, f"features_{base_name}.txt")
if not os.path.exists(scaler_path):
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if not os.path.exists(features_path):
    features_path = os.path.join(MODEL_DIR, "features.txt")

scaler = joblib.load(scaler_path)
with open(features_path, "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

for file in os.listdir(MODEL_DIR):
    path = os.path.join(MODEL_DIR, file)
    try:
        obj = joblib.load(path)
        if "pca" in file and hasattr(obj, "transform") and hasattr(obj, "n_components_"):
            if obj.n_components_ == model.n_features_in_:
                pca = obj
                print(f"✅ Loaded PCA: {file}")
        elif "rbf" in file and hasattr(obj, "transform"):
            test_input = np.random.rand(1, encoder_dim)
            try:
                transformed = obj.transform(test_input)
                if transformed.shape[1] == model.n_features_in_:
                    rbf_sampler = obj
                    print(f"✅ Loaded RBF Sampler: {file}")
            except:
                continue
    except:
        continue

active_flows = defaultdict(list)
flow_lock = threading.Lock()

def extract_features(flow_packets):
    if not flow_packets:
        return None
    start_time = flow_packets[0]['time']
    end_time = flow_packets[-1]['time']
    duration = max(end_time - start_time, 1e-6)
    fwd = [pkt for pkt in flow_packets if pkt['direction'] == 'fwd']
    bwd = [pkt for pkt in flow_packets if pkt['direction'] == 'bwd']
    return {
        "Flow Duration": duration,
        "Total Fwd Packets": len(fwd),
        "Total Backward Packets": len(bwd),
        "Total Length of Fwd Packets": sum(pkt['length'] for pkt in fwd),
        "Total Length of Bwd Packets": sum(pkt['length'] for pkt in bwd),
        "Average Packet Length": np.mean([pkt['length'] for pkt in flow_packets]) if flow_packets else 0,
        "Flow Packets/s": (len(flow_packets) / duration)
    }

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
    if pca:
        X_scaled = pca.transform(X_scaled)
    elif rbf_sampler:
        encoded_input = X_scaled[:, :encoder_dim]  # Assuming 32-dim encoder output
        X_scaled = rbf_sampler.transform(encoded_input)
    try:
        prediction = model.predict(X_scaled)[0]
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    if prediction == 1:
        print(f"🚨 ALERT: Intrusion detected on flow {flow_key} at {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"✅ Normal flow {flow_key}")

def monitor_flows():
    while True:
        now = time.time()
        expired = []
        with flow_lock:
            for key, pkts in active_flows.items():
                if pkts and now - pkts[-1]['time'] > FLOW_TIMEOUT:
                    expired.append(key)
        for key in expired:
            with flow_lock:
                pkts = active_flows.pop(key)
            predict_flow(key, pkts)
        time.sleep(1)

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

print(f"\n🟢 Real-time intrusion detection started using model: {model_file}")
threading.Thread(target=monitor_flows, daemon=True).start()
sniff(prn=handle_packet, store=False)
