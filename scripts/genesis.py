import os
import json
import s1_governance


def ensure_s2_structure(base_path="s2_information"):
    # Create base structure
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "agency"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "agents"), exist_ok=True)

    # Create agency subdirs and shards
    for subdir in ["g1_information", "g4_information", "g5_information"]:
        for i in range(256):
            shard = f"{i:02x}"
            os.makedirs(os.path.join(base_path, "agency", subdir, shard), exist_ok=True)

    # Create epigenome if missing
    epigenome_path = os.path.join(base_path, "agency", "g2_information", "g2_information.dat")
    os.makedirs(os.path.dirname(epigenome_path), exist_ok=True)
    if not os.path.exists(epigenome_path):
        s1_governance.build_epigenome_projection(epigenome_path)


if __name__ == "__main__":
    ensure_s2_structure()
    print("S2 structure and epigenome projection ensured.")
