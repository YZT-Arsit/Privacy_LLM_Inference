# Private-network topology (Alibaba VPC)

- A10 GPU ECS: 172.30.25.154/20 (eth0), NVIDIA A10 24GB
- TDX ECS: 172.30.25.153/20 (eth0), real Intel TDX
- Same VPC + same vSwitch; direct L2 path via eth0, **no NAT/proxy/jump on the data plane**
- RTT 0.143 ms, MTU 8500 (jumbo), 0% packet loss
- Data plane uses PRIVATE IPs only; Mac is control-plane only (reaches A10 via ProxyJump through TDX public)
