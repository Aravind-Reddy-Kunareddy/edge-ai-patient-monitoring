from edge_sim_py import EdgeServer

# ==========================================
# 1. CREATE EDGE SERVER
# ==========================================

edge_server = EdgeServer(
    obj_id=1,
    model_name="Edge Server",
    cpu=4,
    memory=4096,
    disk=10000
)


# ==========================================
# 2. CREATE CLOUD SERVER
# ==========================================

cloud_server = EdgeServer(
    obj_id=2,
    model_name="Cloud Server",
    cpu=16,
    memory=16384,
    disk=50000
)


# ==========================================
# 3. DISPLAY NETWORK
# ==========================================

print("=" * 45)
print("PATIENT MONITORING EDGE-CLOUD NETWORK")
print("=" * 45)

print("\nEDGE SERVER")
print("CPU Cores:", edge_server.cpu)
print("Memory:", edge_server.memory, "MB")
print("Disk:", edge_server.disk, "MB")

print("\nCLOUD SERVER")
print("CPU Cores:", cloud_server.cpu)
print("Memory:", cloud_server.memory, "MB")
print("Disk:", cloud_server.disk, "MB")

print("\nNetwork nodes created successfully!")