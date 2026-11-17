// Copyright (C) 2026 - ZLEqualizer
// This file is part of ZLEqualizer
//
// ZLEqualizer is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License Version 3.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace zldsp::splitter {
    /**
     * Complementary transient/tonal splitter based on the fast/slow level
     * followers used by Airwindows Point.
     *
     * The fast follower tracks attacks while the slow follower represents the
     * local sustain level. Their ratio produces a transient mask. The tonal
     * output is calculated as input - transient, rather than being detected
     * independently, so the two outputs always reconstruct the input (up to
     * floating-point rounding).
     *
     * This class deliberately performs no gain or EQ processing. Callers can
     * feed the two outputs into independent processing chains and sum them.
     */
    template <typename FloatType>
    class PointTransientToneSplitter {
    public:
        void prepare(const double sample_rate) {
            sample_rate_ = std::max(sample_rate, 1.0);
            updateCoefficients();
            reset();
        }

        void setAttack(const double milliseconds) {
            attack_ms_ = std::clamp(milliseconds, 0.05, 100.0);
            updateCoefficients();
        }

        void setSustain(const double milliseconds) {
            sustain_ms_ = std::clamp(milliseconds, 1.0, 2000.0);
            updateCoefficients();
        }

        /**
         * Sensitivity maps the fast/slow envelope ratio into the transient
         * mask. 1 is conservative; larger values extract more of each attack.
         */
        void setSensitivity(const FloatType sensitivity) {
            sensitivity_ = std::clamp(sensitivity, FloatType(0), FloatType(8));
        }

        void reset() {
            fast_envelope_.fill(FloatType(0));
            slow_envelope_.fill(FloatType(0));
        }

        void process(const std::array<FloatType*, 2>& input,
                     const std::array<FloatType*, 2>& transient,
                     const std::array<FloatType*, 2>& tonal,
                     const std::size_t num_samples) {
            for (std::size_t channel = 0; channel < 2; ++channel) {
                auto fast = fast_envelope_[channel];
                auto slow = slow_envelope_[channel];
                const auto* in = input[channel];
                auto* trans = transient[channel];
                auto* tone = tonal[channel];

                for (std::size_t i = 0; i < num_samples; ++i) {
                    const auto sample = in[i];
                    const auto level = std::abs(sample);

                    // Asymmetric followers: attacks rise quickly, while the
                    // release follows the configured decay time.
                    const auto fast_coefficient = level > fast ? fast_attack_coefficient_
                                                               : fast_release_coefficient_;
                    const auto slow_coefficient = level > slow ? slow_attack_coefficient_
                                                               : slow_release_coefficient_;
                    fast = fast_coefficient * fast + (FloatType(1) - fast_coefficient) * level;
                    slow = slow_coefficient * slow + (FloatType(1) - slow_coefficient) * level;

                    const auto ratio = fast / std::max(slow, FloatType(1e-12));
                    const auto mask = std::clamp((ratio - FloatType(1)) * sensitivity_,
                                                 FloatType(0), FloatType(1));
                    const auto transient_sample = sample * mask;
                    trans[i] = transient_sample;
                    tone[i] = sample - transient_sample;
                }

                fast_envelope_[channel] = fast;
                slow_envelope_[channel] = slow;
            }
        }

    private:
        double sample_rate_{48000.0};
        double attack_ms_{2.0};
        double sustain_ms_{120.0};
        FloatType sensitivity_{FloatType(1)};

        FloatType fast_attack_coefficient_{};
        FloatType fast_release_coefficient_{};
        FloatType slow_attack_coefficient_{};
        FloatType slow_release_coefficient_{};
        std::array<FloatType, 2> fast_envelope_{};
        std::array<FloatType, 2> slow_envelope_{};

        static FloatType coefficient(const double milliseconds, const double sample_rate) {
            const auto seconds = std::max(milliseconds * 0.001, 1.0 / sample_rate);
            return static_cast<FloatType>(std::exp(-1.0 / (seconds * sample_rate)));
        }

        void updateCoefficients() {
            // The fast path is intentionally much quicker than the sustain
            // path, mirroring Point's nib/nob detector relationship.
            fast_attack_coefficient_ = coefficient(attack_ms_, sample_rate_);
            fast_release_coefficient_ = coefficient(attack_ms_ * 4.0, sample_rate_);
            slow_attack_coefficient_ = coefficient(std::max(attack_ms_, sustain_ms_ * 0.08), sample_rate_);
            slow_release_coefficient_ = coefficient(sustain_ms_, sample_rate_);
        }
    };
}
