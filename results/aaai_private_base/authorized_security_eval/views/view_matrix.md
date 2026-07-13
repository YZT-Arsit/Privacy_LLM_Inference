# Enforced views

| View | Role | Plaintext weights | Final output | GPU-visible transformed tensors |
|---|---|---:|---:|---:|
| V0 | label-only primary baseline | no | yes | no |
| V1 | deployed API output | no | yes | no |
| V2 | untrusted GPU transformed view | no | no | yes |
| V3 | trusted evaluator / positive control | yes | yes | evaluator-defined |

Access is default-deny and checked both when records are constructed and when a
field is read. Live model/oracle objects are rejected.
