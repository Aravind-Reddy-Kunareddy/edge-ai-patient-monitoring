import inspect
import edge_sim_py

print("NetworkLink initialization:")
print(inspect.getsource(
    edge_sim_py.NetworkLink.__init__
))

print("\nNetworkLink serialization:")
print(inspect.getsource(
    edge_sim_py.NetworkLink._to_dict
))