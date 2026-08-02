#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace qubit::detail {

template <class T>
class SharedVector {
public:
    using value_type = T;
    using storage_type = std::vector<T>;
    using size_type = typename storage_type::size_type;
    using iterator = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;
    using reference = typename storage_type::reference;
    using const_reference = typename storage_type::const_reference;

    SharedVector() : storage_(std::make_shared<storage_type>()) {}
    explicit SharedVector(storage_type values)
        : storage_(std::make_shared<storage_type>(std::move(values))) {}

    SharedVector(const SharedVector&) noexcept = default;
    SharedVector(SharedVector&&) noexcept = default;
    SharedVector& operator=(const SharedVector&) noexcept = default;
    SharedVector& operator=(SharedVector&&) noexcept = default;

    SharedVector& operator=(storage_type values) {
        storage_ = std::make_shared<storage_type>(std::move(values));
        return *this;
    }

    [[nodiscard]] bool empty() const noexcept { return read().empty(); }
    [[nodiscard]] size_type size() const noexcept { return read().size(); }
    [[nodiscard]] size_type capacity() const noexcept { return read().capacity(); }
    [[nodiscard]] long owner_count() const noexcept {
        return storage_ ? storage_.use_count() : 0L;
    }
    [[nodiscard]] bool unique() const noexcept {
        return storage_ && storage_.use_count() == 1L;
    }

    [[nodiscard]] const_reference operator[](size_type index) const {
        return read()[index];
    }
    [[nodiscard]] reference operator[](size_type index) {
        return write()[index];
    }
    [[nodiscard]] const_reference front() const { return read().front(); }
    [[nodiscard]] reference front() { return write().front(); }
    [[nodiscard]] const_reference back() const { return read().back(); }
    [[nodiscard]] reference back() { return write().back(); }

    [[nodiscard]] const T* data() const noexcept { return read().data(); }
    [[nodiscard]] T* data() { return write().data(); }

    [[nodiscard]] const_iterator begin() const noexcept { return read().begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return read().end(); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return read().cbegin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return read().cend(); }
    [[nodiscard]] iterator begin() { return write().begin(); }
    [[nodiscard]] iterator end() { return write().end(); }

    void reserve(size_type capacity_value) { write().reserve(capacity_value); }
    void resize(size_type size_value) { write().resize(size_value); }
    void resize(size_type size_value, const T& value) {
        write().resize(size_value, value);
    }
    void assign(size_type count, const T& value) {
        write().assign(count, value);
    }
    void clear() { write().clear(); }
    void shrink_to_fit() { write().shrink_to_fit(); }
    void push_back(const T& value) { write().push_back(value); }
    void push_back(T&& value) { write().push_back(std::move(value)); }

    template <class... Args>
    reference emplace_back(Args&&... args) {
        storage_type& values = write();
        values.emplace_back(std::forward<Args>(args)...);
        return values.back();
    }

    void pop_back() { write().pop_back(); }
    iterator erase(const_iterator position) {
        const storage_type& current = read();
        const size_type offset =
            static_cast<size_type>(position - current.cbegin());
        storage_type& values = write();
        return values.erase(
            values.cbegin() + static_cast<std::ptrdiff_t>(offset));
    }
    iterator erase(const_iterator first, const_iterator last) {
        const storage_type& current = read();
        const size_type first_offset =
            static_cast<size_type>(first - current.cbegin());
        const size_type last_offset =
            static_cast<size_type>(last - current.cbegin());
        storage_type& values = write();
        return values.erase(
            values.cbegin() + static_cast<std::ptrdiff_t>(first_offset),
            values.cbegin() + static_cast<std::ptrdiff_t>(last_offset));
    }

    void swap(SharedVector& other) noexcept { storage_.swap(other.storage_); }

    [[nodiscard]] std::span<const T> view() const noexcept {
        const storage_type& values = read();
        return {values.data(), values.size()};
    }

    operator std::span<const T>() const noexcept { return view(); }
    operator const storage_type&() const noexcept { return read(); }
    operator storage_type&() { return write(); }

private:
    [[nodiscard]] const storage_type& read() const noexcept {
        if (storage_) {
            return *storage_;
        }
        static const storage_type empty;
        return empty;
    }

    [[nodiscard]] storage_type& write() {
        if (!storage_) {
            storage_ = std::make_shared<storage_type>();
        } else if (storage_.use_count() != 1L) {
            storage_ = std::make_shared<storage_type>(*storage_);
        }
        return *storage_;
    }

    std::shared_ptr<storage_type> storage_;
};

}  // namespace qubit::detail
