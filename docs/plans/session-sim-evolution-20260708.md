# Evolution

| file | turns ok/total | tool threads completed | median latency of OK turns | distinct error types |
|---|---:|---:|---:|---|
| baseline-10k.json | 0/9 | 0 | n/a | context_length_exceeded, token_quota_exceeded |
| live-10k.json | 1/9 | 0 | 2.53s | token_quota_exceeded |
| v36c-10k.json | 11/11 | 2 | 2.67s | none |
| v36d-10k.json | 11/11 | 2 | 0.90s | none |

The progression is a clean reliability climb: baseline fails every turn, mostly from context-length overflows plus token quota errors; live gets one successful turn but is still quota-bound; v36c reaches 11/11 successful turns with both tool threads completing; and v36d keeps that reliability while cutting median OK latency from 2.67s to 0.90s. The remaining caveat is that v36c/v36d tool-thread revert checks still failed in the summaries, so the serving path improved, but the tool-result semantics need separate attention.
