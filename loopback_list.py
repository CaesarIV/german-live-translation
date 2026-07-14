import soundcard as sc

print("DEFAULT SPEAKER (what loopback will capture):")
print("  ", sc.default_speaker().name)

print("\nALL SPEAKERS (render endpoints):")
for s in sc.all_speakers():
    print("  ", s.name)

print("\nLOOPBACK-CAPABLE MICROPHONES:")
for m in sc.all_microphones(include_loopback=True):
    print("  ", "[LOOP]" if m.isloopback else "[ mic]", m.name)
