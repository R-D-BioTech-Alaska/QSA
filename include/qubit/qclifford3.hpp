#pragma once

#include "qubit/qcyclotomic3.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct QSymplectic3Config {
    std::size_t max_entries{1U << 24U};
};

class QSymplectic3 {
public:
    QSymplectic3(
        std::size_t sites,
        std::vector<std::uint8_t> entries,
        QSymplectic3Config config = {})
        : sites_(sites), entries_(std::move(entries)), config_(config) {
        const std::size_t dimension = checked_dimension();
        const std::size_t count = checked_product(dimension, dimension);
        if (config_.max_entries == 0U || count > config_.max_entries || entries_.size() != count) {
            throw QMathError("qutrit symplectic matrix exceeds its exact resource contract");
        }
        for (std::uint8_t& value : entries_) value = static_cast<std::uint8_t>(value % 3U);
        if (!preserves_form()) throw QMathError("qutrit matrix is not symplectic");
    }

    [[nodiscard]] static QSymplectic3 identity(std::size_t sites, QSymplectic3Config config = {}) {
        if (sites == 0U || sites > std::numeric_limits<std::size_t>::max() / 2U) {
            throw QMathError("qutrit symplectic site count is invalid");
        }
        const std::size_t dimension = sites * 2U;
        const std::size_t count = checked_product(dimension, dimension);
        if (config.max_entries == 0U || count > config.max_entries) {
            throw QMathError("qutrit symplectic identity exceeds its exact resource contract");
        }
        std::vector<std::uint8_t> entries(count, 0U);
        for (std::size_t i = 0U; i < dimension; ++i) entries[i * dimension + i] = 1U;
        return QSymplectic3(sites, std::move(entries), config);
    }

    [[nodiscard]] std::size_t sites() const noexcept { return sites_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return sites_ * 2U; }
    [[nodiscard]] const QSymplectic3Config& config() const noexcept { return config_; }

    [[nodiscard]] std::uint8_t at(std::size_t row, std::size_t column) const {
        const std::size_t n = dimension();
        if (row >= n || column >= n) throw QMathError("qutrit symplectic matrix index is out of range");
        return entries_[row * n + column];
    }

    [[nodiscard]] QSymplectic3 composed(const QSymplectic3& rhs) const {
        require_sites(rhs);
        const std::size_t n = dimension();
        std::vector<std::uint8_t> result(entries_.size(), 0U);
        for (std::size_t row = 0U; row < n; ++row) {
            for (std::size_t column = 0U; column < n; ++column) {
                unsigned value = 0U;
                for (std::size_t k = 0U; k < n; ++k) {
                    value += static_cast<unsigned>(at(row, k)) * static_cast<unsigned>(rhs.at(k, column));
                }
                result[row * n + column] = static_cast<std::uint8_t>(value % 3U);
            }
        }
        return QSymplectic3(sites_, std::move(result), conservative_config(rhs));
    }

    [[nodiscard]] QSymplectic3 inverse() const {
        const std::size_t n = dimension();
        std::vector<std::uint8_t> left = entries_;
        std::vector<std::uint8_t> right(n * n, 0U);
        for (std::size_t i = 0U; i < n; ++i) right[i * n + i] = 1U;

        for (std::size_t column = 0U; column < n; ++column) {
            std::size_t pivot = column;
            while (pivot < n && left[pivot * n + column] == 0U) ++pivot;
            if (pivot == n) throw QMathError("qutrit symplectic matrix is singular");
            if (pivot != column) {
                for (std::size_t k = 0U; k < n; ++k) {
                    std::swap(left[column * n + k], left[pivot * n + k]);
                    std::swap(right[column * n + k], right[pivot * n + k]);
                }
            }

            const std::uint8_t scale = left[column * n + column] == 2U ? 2U : 1U;
            for (std::size_t k = 0U; k < n; ++k) {
                left[column * n + k] = mul(left[column * n + k], scale);
                right[column * n + k] = mul(right[column * n + k], scale);
            }

            for (std::size_t row = 0U; row < n; ++row) {
                if (row == column) continue;
                const std::uint8_t factor = left[row * n + column];
                if (factor == 0U) continue;
                for (std::size_t k = 0U; k < n; ++k) {
                    left[row * n + k] = sub(left[row * n + k], mul(factor, left[column * n + k]));
                    right[row * n + k] = sub(right[row * n + k], mul(factor, right[column * n + k]));
                }
            }
        }
        return QSymplectic3(sites_, std::move(right), config_);
    }

    [[nodiscard]] bool identity_exact() const noexcept {
        const std::size_t n = dimension();
        for (std::size_t row = 0U; row < n; ++row) {
            for (std::size_t column = 0U; column < n; ++column) {
                const std::uint8_t expected = row == column ? 1U : 0U;
                if (entries_[row * n + column] != expected) return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "qsymplectic3:1," + std::to_string(sites_) + ':';
        for (std::size_t i = 0U; i < entries_.size(); ++i) {
            if (i != 0U) out += ',';
            out += static_cast<char>('0' + entries_[i]);
        }
        return out;
    }

    friend bool operator==(const QSymplectic3&, const QSymplectic3&) = default;

private:
    [[nodiscard]] static std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
        if (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
            throw QMathError("qutrit symplectic matrix size overflow");
        }
        return lhs * rhs;
    }

    [[nodiscard]] std::size_t checked_dimension() const {
        if (sites_ == 0U || sites_ > std::numeric_limits<std::size_t>::max() / 2U) {
            throw QMathError("qutrit symplectic site count is invalid");
        }
        return sites_ * 2U;
    }

    [[nodiscard]] static std::uint8_t mul(std::uint8_t lhs, std::uint8_t rhs) noexcept {
        return static_cast<std::uint8_t>((static_cast<unsigned>(lhs) * static_cast<unsigned>(rhs)) % 3U);
    }

    [[nodiscard]] static std::uint8_t sub(std::uint8_t lhs, std::uint8_t rhs) noexcept {
        return static_cast<std::uint8_t>((static_cast<unsigned>(lhs) + 3U - static_cast<unsigned>(rhs)) % 3U);
    }

    [[nodiscard]] std::uint8_t column_form(std::size_t left, std::size_t right) const noexcept {
        unsigned positive = 0U;
        unsigned negative = 0U;
        for (std::size_t site = 0U; site < sites_; ++site) {
            positive += static_cast<unsigned>(entries_[(sites_ + site) * dimension() + left]) *
                        static_cast<unsigned>(entries_[site * dimension() + right]);
            negative += static_cast<unsigned>(entries_[site * dimension() + left]) *
                        static_cast<unsigned>(entries_[(sites_ + site) * dimension() + right]);
        }
        return static_cast<std::uint8_t>((positive + 3U - (negative % 3U)) % 3U);
    }

    [[nodiscard]] std::uint8_t canonical_form(std::size_t left, std::size_t right) const noexcept {
        if (left < sites_ && right >= sites_ && left == right - sites_) return 2U;
        if (left >= sites_ && right < sites_ && left - sites_ == right) return 1U;
        return 0U;
    }

    [[nodiscard]] bool preserves_form() const noexcept {
        const std::size_t n = dimension();
        for (std::size_t left = 0U; left < n; ++left) {
            for (std::size_t right = 0U; right < n; ++right) {
                if (column_form(left, right) != canonical_form(left, right)) return false;
            }
        }
        return true;
    }

    void require_sites(const QSymplectic3& rhs) const {
        if (sites_ != rhs.sites_) throw QMathError("qutrit symplectic matrices have different site counts");
    }

    [[nodiscard]] QSymplectic3Config conservative_config(const QSymplectic3& rhs) const noexcept {
        return QSymplectic3Config{std::min(config_.max_entries, rhs.config_.max_entries)};
    }

    std::size_t sites_{0U};
    std::vector<std::uint8_t> entries_{};
    QSymplectic3Config config_{};
};

struct QClifford3Config {
    std::size_t max_steps{1U << 20U};
    QSymplectic3Config symplectic{};
};

class QClifford3Map {
public:
    [[nodiscard]] static QClifford3Map identity(const QWeylSpace& space, QClifford3Config config = {}) {
        return QClifford3Map(space, canonical_images(space), config, true);
    }

    [[nodiscard]] static QClifford3Map fourier(
        const QWeylSpace& space,
        std::size_t site,
        QClifford3Config config = {}) {
        require_qutrit_space(space);
        require_site(space, site);
        auto images = canonical_images(space);
        const std::size_t sites = space.site_count();
        const QWeylOperator x = canonical_generator(space, site);
        const QWeylOperator z = canonical_generator(space, sites + site);
        images[site] = z;
        images[sites + site] = x.inverse();
        return QClifford3Map(space, std::move(images), config, true);
    }

    [[nodiscard]] static QClifford3Map phase(
        const QWeylSpace& space,
        std::size_t site,
        QClifford3Config config = {}) {
        require_qutrit_space(space);
        require_site(space, site);
        auto images = canonical_images(space);
        images[site] = QWeylOperator::local(space, site, 1, 1);
        return QClifford3Map(space, std::move(images), config, true);
    }

    [[nodiscard]] static QClifford3Map sum(
        const QWeylSpace& space,
        std::size_t control,
        std::size_t target,
        QClifford3Config config = {}) {
        require_qutrit_space(space);
        require_site(space, control);
        require_site(space, target);
        if (control == target) throw QMathError("qutrit SUM control and target must differ");
        auto images = canonical_images(space);
        const std::size_t sites = space.site_count();
        const QWeylOperator xc = canonical_generator(space, control);
        const QWeylOperator xt = canonical_generator(space, target);
        const QWeylOperator zc = canonical_generator(space, sites + control);
        const QWeylOperator zt = canonical_generator(space, sites + target);
        images[control] = xc.multiplied(xt);
        images[sites + target] = zc.inverse().multiplied(zt);
        return QClifford3Map(space, std::move(images), config, true);
    }

    [[nodiscard]] const QWeylSpace& space() const noexcept { return space_; }
    [[nodiscard]] const QClifford3Config& config() const noexcept { return config_; }

    [[nodiscard]] QWeylOperator transform(const QWeylOperator& value) const {
        require_space(value.space());
        QWeylOperator result(space_, value.phase_turns(), std::vector<QWeylExponent>(space_.site_count()));
        for (std::size_t site = 0U; site < space_.site_count(); ++site) {
            const QWeylExponent exponent = value.exponents()[site];
            if (exponent.shift != 0U) result = result.multiplied(images_[site].power(exponent.shift));
            if (exponent.clock != 0U) {
                result = result.multiplied(images_[space_.site_count() + site].power(exponent.clock));
            }
        }
        return result;
    }

    [[nodiscard]] QWeyl3Algebra transform(const QWeyl3Algebra& algebra) const {
        require_space(algebra.space());
        QWeyl3Algebra result(space_);
        for (const QWeyl3Term& term : algebra.terms()) result.add(transform(term.basis), term.coefficient);
        return result;
    }

    [[nodiscard]] QClifford3Map composed(const QClifford3Map& rhs) const {
        require_space(rhs.space_);
        std::vector<QWeylOperator> images;
        images.reserve(images_.size());
        for (const QWeylOperator& image : rhs.images_) images.push_back(transform(image));
        return QClifford3Map(space_, std::move(images), conservative_config(rhs), false);
    }

    [[nodiscard]] QSymplectic3 symplectic() const {
        const std::size_t sites = space_.site_count();
        const std::size_t n = sites * 2U;
        std::vector<std::uint8_t> entries(n * n, 0U);
        for (std::size_t column = 0U; column < n; ++column) {
            for (std::size_t site = 0U; site < sites; ++site) {
                entries[site * n + column] = static_cast<std::uint8_t>(images_[column].exponents()[site].shift);
                entries[(sites + site) * n + column] = static_cast<std::uint8_t>(images_[column].exponents()[site].clock);
            }
        }
        return QSymplectic3(sites, std::move(entries), config_.symplectic);
    }

    [[nodiscard]] QClifford3Map inverse() const {
        const QSymplectic3 inverse_matrix = symplectic().inverse();
        const std::size_t sites = space_.site_count();
        std::vector<QWeylOperator> images;
        images.reserve(sites * 2U);
        for (std::size_t column = 0U; column < sites * 2U; ++column) {
            std::vector<QWeylExponent> exponents(sites);
            for (std::size_t site = 0U; site < sites; ++site) {
                exponents[site].shift = inverse_matrix.at(site, column);
                exponents[site].clock = inverse_matrix.at(sites + site, column);
            }
            QWeylOperator candidate(space_, QRational(0), std::move(exponents));
            const QWeylOperator mapped = transform(candidate);
            const QWeylOperator target = canonical_generator(space_, column);
            if (!mapped.equivalent_up_to_global_phase(target)) {
                throw QMathError("qutrit Clifford inverse failed symplectic reconstruction");
            }
            images.emplace_back(space_, -mapped.phase_turns(), candidate.exponents());
        }
        return QClifford3Map(space_, std::move(images), config_, false);
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "qclifford3:1:" + space_.canonical() + '(';
        for (std::size_t i = 0U; i < images_.size(); ++i) {
            if (i != 0U) out += ';';
            out += images_[i].canonical();
        }
        out += ')';
        return out;
    }

private:
    QClifford3Map(
        QWeylSpace space,
        std::vector<QWeylOperator> images,
        QClifford3Config config,
        bool validate)
        : space_(std::move(space)), images_(std::move(images)), config_(config) {
        require_qutrit_space(space_);
        if (config_.max_steps == 0U || config_.symplectic.max_entries == 0U) {
            throw QMathError("qutrit Clifford resource caps must be positive");
        }
        if (images_.size() != space_.site_count() * 2U) {
            throw QMathError("qutrit Clifford generator image count is invalid");
        }
        for (const QWeylOperator& image : images_) {
            require_space(image.space());
            (void)QCyclotomic3::from_turns(image.phase_turns());
        }
        if (validate) (void)symplectic();
    }

    [[nodiscard]] static QWeylOperator canonical_generator(const QWeylSpace& space, std::size_t index) {
        const std::size_t sites = space.site_count();
        if (index >= sites * 2U) throw QMathError("qutrit Clifford generator index is out of range");
        return index < sites
            ? QWeylOperator::local(space, index, 1, 0)
            : QWeylOperator::local(space, index - sites, 0, 1);
    }

    [[nodiscard]] static std::vector<QWeylOperator> canonical_images(const QWeylSpace& space) {
        require_qutrit_space(space);
        std::vector<QWeylOperator> result;
        result.reserve(space.site_count() * 2U);
        for (std::size_t i = 0U; i < space.site_count() * 2U; ++i) result.push_back(canonical_generator(space, i));
        return result;
    }

    static void require_qutrit_space(const QWeylSpace& space) {
        for (const std::uint32_t dimension : space.dimensions()) {
            if (dimension != 3U) throw QMathError("qutrit Clifford map requires local dimension three");
        }
    }

    static void require_site(const QWeylSpace& space, std::size_t site) {
        if (site >= space.site_count()) throw QMathError("qutrit Clifford site is out of range");
    }

    void require_space(const QWeylSpace& space) const {
        if (space != space_) throw QMathError("qutrit Clifford objects belong to different local spaces");
    }

    [[nodiscard]] QClifford3Config conservative_config(const QClifford3Map& rhs) const noexcept {
        return QClifford3Config{
            std::min(config_.max_steps, rhs.config_.max_steps),
            QSymplectic3Config{std::min(config_.symplectic.max_entries, rhs.config_.symplectic.max_entries)},
        };
    }

    QWeylSpace space_;
    std::vector<QWeylOperator> images_{};
    QClifford3Config config_{};
};

enum class QClifford3GateKind : std::uint8_t {
    Fourier = 0,
    Phase = 1,
    Sum = 2,
};

struct QClifford3Gate {
    QClifford3GateKind kind{QClifford3GateKind::Fourier};
    std::size_t first{0U};
    std::size_t second{0U};
};

struct QClifford3CompileReceipt {
    std::size_t sites{0U};
    std::size_t steps{0U};
    std::size_t generator_images{0U};
    bool exact{false};
    bool ready{false};
};

class QClifford3Program {
public:
    explicit QClifford3Program(QWeylSpace space, QClifford3Config config = {})
        : space_(std::move(space)), config_(config) {
        if (config_.max_steps == 0U) throw QMathError("qutrit Clifford step cap must be positive");
        (void)QClifford3Map::identity(space_, config_);
    }

    void append_fourier(std::size_t site) { append(QClifford3Gate{QClifford3GateKind::Fourier, site, 0U}); }
    void append_phase(std::size_t site) { append(QClifford3Gate{QClifford3GateKind::Phase, site, 0U}); }
    void append_sum(std::size_t control, std::size_t target) {
        append(QClifford3Gate{QClifford3GateKind::Sum, control, target});
    }

    [[nodiscard]] std::size_t step_count() const noexcept { return gates_.size(); }

    [[nodiscard]] QClifford3Map compile(QClifford3CompileReceipt* receipt = nullptr) const {
        QClifford3Map result = QClifford3Map::identity(space_, config_);
        for (const QClifford3Gate& gate : gates_) {
            QClifford3Map step = gate.kind == QClifford3GateKind::Fourier
                ? QClifford3Map::fourier(space_, gate.first, config_)
                : gate.kind == QClifford3GateKind::Phase
                    ? QClifford3Map::phase(space_, gate.first, config_)
                    : QClifford3Map::sum(space_, gate.first, gate.second, config_);
            result = step.composed(result);
        }
        if (receipt != nullptr) {
            receipt->sites = space_.site_count();
            receipt->steps = gates_.size();
            receipt->generator_images = space_.site_count() * 2U;
            receipt->exact = true;
            receipt->ready = true;
        }
        return result;
    }

private:
    void append(QClifford3Gate gate) {
        if (gates_.size() >= config_.max_steps) throw QMathError("qutrit Clifford program step cap exceeded");
        if (gate.kind == QClifford3GateKind::Fourier) (void)QClifford3Map::fourier(space_, gate.first, config_);
        else if (gate.kind == QClifford3GateKind::Phase) (void)QClifford3Map::phase(space_, gate.first, config_);
        else (void)QClifford3Map::sum(space_, gate.first, gate.second, config_);
        gates_.push_back(gate);
    }

    QWeylSpace space_;
    QClifford3Config config_{};
    std::vector<QClifford3Gate> gates_{};
};

}  // namespace qubit
