#!/usr/bin/env python3
"""Control-plane wrapper for the second researcher-owned A10/TDX pair."""
from __future__ import annotations

import gate_e2e_protected as gate


gate.CM_A10 = "/tmp/cm-a10-pair2"
gate.A10 = "root@172.30.25.155"
gate.A10_PUB = "root@8.147.119.67"
gate.TDX = "root@8.147.112.139"
gate.TDX_PRIV = "172.30.25.156"


if __name__ == "__main__":
    gate.main()
