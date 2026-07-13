#include "qubit/c_api.h"
#include "qubit/qstate.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

qubit::QRegister* as_state(qstate_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("Qubit state handle is null");
    }
    return static_cast<qubit::QRegister*>(handle);
}

template <typename Function>
int guarded(Function&& function) {
    try {
        function();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "Unknown Qubit native engine error";
        return -1;
    }
}

}  // namespace

extern "C" {

qstate_handle qstate_create(size_t qubit_count) {
    try {
        auto* state = new qubit::QRegister(qubit_count);
        last_error.clear();
        return state;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown Qubit native engine error";
        return nullptr;
    }
}

void qstate_destroy(qstate_handle handle) {
    delete static_cast<qubit::QRegister*>(handle);
}

const char* qstate_last_error(void) {
    return last_error.c_str();
}

int qstate_apply_x(qstate_handle handle, uint32_t qubit) {
    return guarded([&] { as_state(handle)->apply_x(qubit); });
}

int qstate_apply_y(qstate_handle handle, uint32_t qubit) {
    return guarded([&] { as_state(handle)->apply_y(qubit); });
}

int qstate_apply_z(qstate_handle handle, uint32_t qubit) {
    return guarded([&] { as_state(handle)->apply_z(qubit); });
}

int qstate_apply_h(qstate_handle handle, uint32_t qubit) {
    return guarded([&] { as_state(handle)->apply_h(qubit); });
}

int qstate_apply_s(qstate_handle handle, uint32_t qubit) {
    return guarded([&] { as_state(handle)->apply_s(qubit); });
}

int qstate_apply_t(qstate_handle handle, uint32_t qubit) {
    return guarded([&] { as_state(handle)->apply_t(qubit); });
}

int qstate_apply_rx(qstate_handle handle, uint32_t qubit, double theta) {
    return guarded([&] { as_state(handle)->apply_rx(qubit, theta); });
}

int qstate_apply_ry(qstate_handle handle, uint32_t qubit, double theta) {
    return guarded([&] { as_state(handle)->apply_ry(qubit, theta); });
}

int qstate_apply_rz(qstate_handle handle, uint32_t qubit, double theta) {
    return guarded([&] { as_state(handle)->apply_rz(qubit, theta); });
}

int qstate_apply_cnot(qstate_handle handle, uint32_t control, uint32_t target) {
    return guarded([&] { as_state(handle)->apply_cnot(control, target); });
}

int qstate_apply_cz(qstate_handle handle, uint32_t first, uint32_t second) {
    return guarded([&] { as_state(handle)->apply_cz(first, second); });
}

int qstate_apply_swap(qstate_handle handle, uint32_t first, uint32_t second) {
    return guarded([&] { as_state(handle)->apply_swap(first, second); });
}

int qstate_probability_one(qstate_handle handle, uint32_t qubit, double* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Probability output pointer is null");
        }
        *result = as_state(handle)->probability_one(qubit);
    });
}

int qstate_measure(qstate_handle handle, uint32_t qubit, double sample, int* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Measurement output pointer is null");
        }
        *result = as_state(handle)->measure(qubit, sample);
    });
}

int qstate_amplitude(qstate_handle handle, uint64_t basis_index, double* real, double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Amplitude output pointer is null");
        }
        const qubit::QComplex value = as_state(handle)->amplitude(basis_index);
        *real = value.re;
        *imag = value.im;
    });
}

size_t qstate_qubit_count(qstate_handle handle) {
    try {
        return as_state(handle)->qubit_count();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_component_count(qstate_handle handle) {
    try {
        return as_state(handle)->component_count();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_component_size(qstate_handle handle, uint32_t qubit) {
    try {
        return as_state(handle)->component_size(qubit);
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_component_nonzero_count(qstate_handle handle, uint32_t qubit) {
    try {
        return as_state(handle)->component_nonzero_count(qubit);
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_estimated_bytes(qstate_handle handle) {
    try {
        return as_state(handle)->estimated_bytes();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_description_size(qstate_handle handle) {
    try {
        return as_state(handle)->describe().size() + 1U;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

int qstate_description_write(qstate_handle handle, char* output, size_t output_size) {
    return guarded([&] {
        if (output == nullptr) {
            throw qubit::QStateError("Description output pointer is null");
        }
        const std::string description = as_state(handle)->describe();
        if (output_size < description.size() + 1U) {
            throw qubit::QStateError("Description output buffer is too small");
        }
        std::memcpy(output, description.c_str(), description.size() + 1U);
    });
}

size_t qstate_qsc_size(qstate_handle handle) {
    try {
        return as_state(handle)->encode_qsc().size();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

int qstate_qsc_write(qstate_handle handle, uint8_t* output, size_t output_size) {
    return guarded([&] {
        if (output == nullptr) {
            throw qubit::QStateError("QSC output pointer is null");
        }
        const std::vector<std::uint8_t> encoded = as_state(handle)->encode_qsc();
        if (output_size < encoded.size()) {
            throw qubit::QStateError("QSC output buffer is too small");
        }
        std::copy(encoded.begin(), encoded.end(), output);
    });
}

qstate_handle qstate_qsc_read(const uint8_t* data, size_t data_size) {
    try {
        if (data == nullptr || data_size == 0U) {
            throw qubit::QStateError("QSC input buffer is empty");
        }
        qubit::QRegister decoded = qubit::QRegister::decode_qsc(
            std::span<const std::uint8_t>(data, data_size));
        auto* state = new qubit::QRegister(std::move(decoded));
        last_error.clear();
        return state;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown Qubit native engine error";
        return nullptr;
    }
}

}  // extern "C"
