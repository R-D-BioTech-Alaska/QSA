#include "qubit/c_api.h"

int main(void) {
    qstate_handle state = qstate_create(2);
    if (state == 0) {
        return 1;
    }
    if (qstate_apply_h(state, 0) != 0 || qstate_apply_cnot(state, 0, 1) != 0) {
        qstate_destroy(state);
        return 2;
    }
    qstate_destroy(state);
    return qstate_abi_version_major() == 1 ? 0 : 3;
}
