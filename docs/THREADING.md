# Threading and concurrency

QSA supports concurrency across independent state objects. It does not add an
internal mutex to every mutable register operation because that would penalize
the normal single-owner execution path.

## Safe patterns

- Use one `QRegister` or `QubitRegister` from one thread at a time.
- Execute one immutable `OperationPlan` over different registers concurrently.
- Use `execute_many` for QSA-managed parallel execution across independent
  registers.
- Use separate `GroverSearch` and `SymmetryState` instances in separate threads.
- Read `qstate_last_error()` in the same thread that made the failed C call.

## Unsafe patterns

- Mutating the same register concurrently from multiple threads.
- Closing a Python/native handle while another thread is using it.
- Sharing output buffers between concurrent C calls without synchronization.
- Passing the same register more than once in a parallel ensemble.

## Python behavior

Python objects serialize close/execute transitions for their own native handle,
but this is not a substitute for application-level ownership. Keep a mutable
state object assigned to one worker during an operation sequence.
