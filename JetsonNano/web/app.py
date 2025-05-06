from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3
import psutil
import subprocess
import re
import os
import ipaddress

app = Flask(__name__)
auth = HTTPBasicAuth()

DATABASE = "users.db"

pipeline_metrics = {
    "fps": 0.0,
    "latency": 0.0,
    "face_count": 0,
    "skeleton_count": 0
}
# --- Authentication using SQLite ---

def get_user_password(username):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

@auth.verify_password
def verify_password(username, password):
    stored_password = get_user_password(username)
    if stored_password and check_password_hash(stored_password, password):
        return username
    return None

# --- Helper Function to Gather Device Info ---

def get_device_info():
    # Temperature (assuming the temperature file exists and is in millidegrees Celsius)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_millideg = int(f.read().strip())
        temperature = round(temp_millideg / 1000.0, 1)
    except Exception:
        temperature = "N/A"

    # CPU usage using psutil (averaged over 1 second)
    cpu_usage = psutil.cpu_percent(interval=1)

    # RAM usage using psutil (in GB)
    mem = psutil.virtual_memory()
    used_ram = round(mem.used / (1024**3), 1)
    total_ram = round(mem.total / (1024**3), 1)
    ram_usage = "{}{}/{}GB".format("", used_ram, total_ram)

    # GPU usage using tegrastats (parsing output for GR3D_FREQ)
    try:
        proc = subprocess.Popen(
            ["sudo", "-u", "jetson", "tegrastats", "--interval", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        gpu_usage = "N/A"
        # Try reading up to 5 lines to find one with GR3D_FREQ
        for _ in range(5):
            line = proc.stdout.readline().strip()
            if "GR3D_FREQ" in line:
                match = re.search(r'GR3D_FREQ\s*(\d+)%', line)
                if match:
                    gpu_usage = match.group(1) + "%"
                    break
        proc.terminate()
    except Exception:
        gpu_usage = "N/A"


    return {
        "temperature": "{} °C".format(temperature) if temperature != "N/A" else "N/A",
        "cpu": "{}%".format(cpu_usage),
        "gpu": gpu_usage,
        "ram": ram_usage
    }

# --- Routes ---

# Serve the main HTML page
@app.route("/")
@auth.login_required
def index():
    # Extract the host IP from the request host.
    host_ip = request.host.split(":")[0]
    # Build the HLS stream URL. Adjust the port (8888) and path as needed.
    stream_url = "http://{}:8888/live/index.m3u8".format(host_ip)
    return render_template("index.html", stream_url=stream_url)

# Serve other static files if needed
@app.route("/static/<path:path>")
@auth.login_required
def serve_static(path):
    return send_from_directory("static", path)

# API endpoint: Get device info
@app.route("/api/device_info", methods=["GET"])
@auth.login_required
def device_info():
    info = get_device_info()
    return jsonify(info)

# API endpoint: Restart the system (Jetson Nano)
@app.route("/api/restart_system", methods=["POST"])
@auth.login_required
def restart_system():
    try:
        subprocess.Popen(["sudo", "reboot"])
        return jsonify({"status": "Reboot initiated"}), 200
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

# API endpoint: Restart a systemd service
@app.route("/api/restart_service", methods=["POST"])
@auth.login_required
def restart_service():
    data = request.get_json()
    service = data.get("service")
    if not service:
        return jsonify({"status": "Error", "error": "No service specified"}), 400
    try:
        subprocess.check_output(["sudo", "systemctl", "restart", service])
        return jsonify({"status": "Service '{}' restarted".format(service)}), 200
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

# API endpoint: Update configuration (e.g. environment variable) and restart a service
@app.route("/api/update_config", methods=["POST"])
@auth.login_required
def update_config():
    data = request.get_json()
    variable = data.get("variable")
    value = data.get("value")
    service = data.get("service")
    if not all([variable, value, service]):
        return jsonify({"status": "Error", "error": "Missing parameters"}), 400

    config_file = "/path/to/your/config.env"  # <-- Update this path as needed
    try:
        # Read current config file
        with open(config_file, "r") as f:
            lines = f.readlines()

        # Update or append the variable
        found = False
        with open(config_file, "w") as f:
            for line in lines:
                if line.startswith(variable + "="):
                    f.write("{}={}\n".format(variable, value))
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write("{}={}\n".format(variable, value))

        # Restart the specified service to apply changes
        subprocess.check_output(["sudo", "systemctl", "restart", service])
        return jsonify({"status": "Config updated and service '{}' restarted".format(service)}), 200
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

# API endpoint: Update eth0 network parameters
@app.route("/api/update_eth0", methods=["POST"])
@auth.login_required
def update_eth0():
    data = request.get_json()
    ip = data.get("ip")
    subnet = data.get("subnet")
    gateway = data.get("gateway")
    try:
        # Validate IP addresses
        ipaddress.IPv4Address(ip)
        ipaddress.IPv4Address(gateway)
        ipaddress.IPv4Address(subnet)
        # Convert subnet mask to prefix length
        prefix_len = sum(bin(int(octet)).count("1") for octet in subnet.split('.'))
    except Exception:
        return jsonify({"status": "Error", "error": "Invalid IP address or subnet mask"}), 400

    try:
        # Update network configuration (this may disconnect your session)
        subprocess.check_output(["sudo", "ip", "addr", "flush", "dev", "eth0"])
        new_ip = "{}{}/{}".format("", ip, prefix_len)
        subprocess.check_output(["sudo", "ip", "addr", "add", new_ip, "dev", "eth0"])
        subprocess.call(["sudo", "ip", "route", "del", "default"])
        subprocess.check_output(["sudo", "ip", "route", "add", "default", "via", gateway])
        return jsonify({"status": "Network updated", "ip": new_ip, "gateway": gateway}), 200
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

# API endpoint: Change password for the admin user
@app.route("/api/change_password", methods=["POST"])
@auth.login_required
def change_password():
    data = request.get_json()
    current_password = data.get("current_password")
    new_password = data.get("new_password")
    stored_password = get_user_password(auth.current_user())
    if not stored_password or not check_password_hash(stored_password, current_password):
        return jsonify({"status": "Error", "error": "Current password is incorrect"}), 400
    new_hash = generate_password_hash(new_password)
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password = ? WHERE username = ?", (new_hash, auth.current_user()))
        conn.commit()
        conn.close()
        return jsonify({"status": "Password updated successfully"}), 200
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

# Logout endpoint for HTTP Basic Auth (forces re-prompt)
@app.route("/logout")
@auth.login_required
def logout():
    return ('Logged out', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})

@app.route("/metrics", methods=["GET", "POST"])
def handle_metrics():
    global pipeline_metrics

    if request.method == "POST":
        # The pipeline sends data here as JSON: { "fps":..., "latency":..., "face_count":..., ... }
        data = request.get_json(force=True)
        pipeline_metrics["fps"] = data.get("fps", 0.0)
        pipeline_metrics["latency"] = data.get("latency", 0.0)
        pipeline_metrics["face_count"] = data.get("face_count", 0)
        pipeline_metrics["skeleton_count"] = data.get("skeleton_count", 0)
        return jsonify({"status": "OK"})

    else:
        # GET request from the frontend (index.html) to retrieve current pipeline metrics
        return jsonify(pipeline_metrics)

@app.route("/api/sync_settings", methods=["POST"])
@auth.login_required
def sync_settings():
    """
    Reads JSON from request body and updates /srv/anonymize/.env accordingly.
    Then restarts anonymize.service.
    """
    data = request.get_json(force=True)
    
    env_path = "/srv/anonymize/.env"
    if not os.path.exists(env_path):
        return jsonify({"status": "Error", "error": f"{env_path} not found"}), 400

    # Read all lines from .env
    with open(env_path, "r") as f:
        lines = f.readlines()

    # We'll store the updated lines here
    updated_lines = []
    # Keep track of which keys we updated
    updated_keys = set()

    # For each line, if it matches a KEY= format we check if KEY is in data
    for line in lines:
        # Strip but keep newline for re-add
        stripped = line.strip()
        
        if stripped.startswith("#") or "=" not in stripped:
            # It's a comment or blank or invalid line, keep as-is
            updated_lines.append(line)
            continue
        
        # Parse env var line
        # Typically it's KEY=VALUE
        key, _, _value = stripped.partition("=")
        key = key.strip()
        
        if key in data:
            # Convert booleans to string "True"/"False", else keep as is
            new_val = data[key]
            if isinstance(new_val, bool):
                new_val = "True" if new_val else "False"
            else:
                new_val = str(new_val)
            
            updated_lines.append(f"{key}={new_val}\n")
            updated_keys.add(key)
        else:
            # Not in the data from the POST request, keep as is
            updated_lines.append(line)

    # If data contains keys that weren't in the .env at all, append them
    for key, val in data.items():
        if key not in updated_keys:
            # Convert booleans to string "True"/"False"
            if isinstance(val, bool):
                val = "True" if val else "False"
            else:
                val = str(val)
            updated_lines.append(f"{key}={val}\n")

    try:
        # Write everything back
        with open(env_path, "w") as f:
            f.writelines(updated_lines)
        
        # Now restart the service
        subprocess.check_call(["sudo", "systemctl", "restart", "anonymize.service"])
        
        return jsonify({"status": "OK", "message": "Settings updated and anonymize.service restarted"}), 200
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

def read_env_file(env_path):
    """
    Reads a .env file line-by-line and returns a dict of {KEY: VALUE}.
    Lines that are commented out (#) or invalid are skipped.
    """
    config = {}
    if not os.path.exists(env_path):
        return config

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue
            # Must have KEY=VALUE
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                config[key] = val
    return config

@app.route("/api/get_settings", methods=["GET"])
@auth.login_required
def get_settings():
    env_path = "/srv/anonymize/.env"
    config_data = read_env_file(env_path)  # the function shown above
    return jsonify(config_data), 200

if __name__ == "__main__":
    # Running on port 80 typically requires root privileges (or use a reverse proxy)
    app.run(host="0.0.0.0", port=80, debug=True)
