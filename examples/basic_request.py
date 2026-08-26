from oran_control import NetworkRequest
print(NetworkRequest().require_embb_throughput(10).summary())
print()
print(NetworkRequest().limit_embb_buffer(50).summary())
