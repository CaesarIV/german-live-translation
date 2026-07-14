import sounddevice as sd

print("Host APIs:")
for i, h in enumerate(sd.query_hostapis()):
    print(f"  [{i}] {h['name']:22} default_in={h['default_input_device']:>3} default_out={h['default_output_device']:>3}")

print("\nDevices (in = capturable):")
for i, d in enumerate(sd.query_devices()):
    io = []
    if d["max_input_channels"] > 0:
        io.append(f"in:{d['max_input_channels']}")
    if d["max_output_channels"] > 0:
        io.append(f"out:{d['max_output_channels']}")
    tag = "   <== VB-CABLE" if "CABLE" in d["name"].upper() else ""
    print(f"  [{i:2}] {d['name'][:48]:48} {'/'.join(io):10} api={d['hostapi']} sr={int(d['default_samplerate'])}{tag}")
