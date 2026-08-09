# Next representation phase

The next major scaling backend should address bounded entanglement rather than add another narrow local fast path.

The intended target is an exact tensor-network or Schmidt-width representation with an explicit contraction or bond bound. It must compare against QRegister on bounded systems, expose the governing width/resource metric, fail closed when that metric exceeds its configured limit, and keep QRegister as the exact fallback.
