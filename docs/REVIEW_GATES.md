# Review gates for specialized exact paths

A new exact backend or fast path must show its eligibility rule, its failure boundary, and its fallback behavior before integration.

Required evidence:

- a bounded independent reference comparison;
- an adversarial case that destroys the claimed structure;
- unchanged QRegister fallback behavior;
- ancestor/root immutability checks where storage is shared;
- resource accounting that separates setup from repeated execution;
- native validation after representation or storage transitions;
- no silent approximation when the controlling structural variable exceeds its configured bound.

A performance result does not replace an exactness gate.
