# Main-experiment health monitor (read-only)

_Last pass: 2026-07-12T13:15:00+00:00 UTC_

**driver/orchestrator not detected locally; verify run state (read-only).**

- driver pids: none · orchestrator pids: none
- GPU: `0 %, 0 MiB, 23028 MiB, 29`
- remote checkpoints: ``

| cell | state | steps | dev acc | nll | sel step | mtime |
|---|---|---|---|---|---|---|
| sst2_conv_L0 | present | 0 | 0.8853211009174312 | 0.8760087408057047 | -1 | 2026-07-12T19:49:35 |
| sst2_conv_L5_s1234 | present | 1000 | 0.9162844036697247 | 0.2374358276981826 | 400 | 2026-07-12T20:00:15 |
| sst2_conv_L12_s1234 | present | 1000 | 0.9151376146788991 | 0.2396792232853557 | 400 | 2026-07-12T20:10:17 |
| sst2_conv_L5_s2025 | present | 1800 | 0.9048165137614679 | 0.2534260511945147 | 1200 | 2026-07-12T20:28:17 |
| sst2_conv_L12_s2025 | present | 1000 | 0.9231651376146789 | 0.20849722534964937 | 400 | 2026-07-12T20:38:30 |
| sst2_conv_L5_s7 | present | 2000 | 0.9254587155963303 | 0.2096826840841442 | 1400 | 2026-07-12T20:59:15 |
| sst2_conv_L12_s7 | present | 1400 | 0.9162844036697247 | 0.2394520438729076 | 800 | 2026-07-12T21:12:51 |

## last task output
```
          "n_correct": 758
        }
      }
    ]
  }
}
```

## remote processes (read-only)
```
8    10:51:17  0.0  0.0 kworker/0:0H-events_highpri
     23    10:51:17  0.0  0.0 kworker/1:0-events
     24    10:51:17  0.0  0.0 kworker/1:0H-events_highpri
     30    10:51:17  0.0  0.0 kworker/2:0H-events_highpri
     36    10:51:17  0.0  0.0 kworker/3:0H-events_highpri
     42    10:51:17  0.0  0.0 kworker/4:0H-events_highpri
```