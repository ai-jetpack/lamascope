#pragma once
#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

// Seconds since application start.
inline double NowSeconds() {
    static const auto start = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Fixed-capacity time series. One writer thread, any number of readers.
// Critical sections are a few copies, so a mutex is fine at our rates.
class Series {
public:
    explicit Series(size_t capacity = 6000) : cap_(capacity) {
        t_.reserve(cap_);
        v_.reserve(cap_);
    }

    void Push(double t, float v) {
        std::lock_guard lk(mu_);
        if (t_.size() < cap_) {
            t_.push_back((float)t);
            v_.push_back(v);
        } else {
            t_[head_] = (float)t;
            v_[head_] = v;
            head_ = (head_ + 1) % cap_;
        }
        last_ = v;
        has_ = true;
    }

    // Points with t >= tmin, in chronological order.
    void Copy(double tmin, std::vector<float>& xs, std::vector<float>& ys) const {
        std::lock_guard lk(mu_);
        xs.clear();
        ys.clear();
        size_t n = t_.size();
        for (size_t k = 0; k < n; ++k) {
            size_t i = (n < cap_) ? k : (head_ + k) % cap_;
            if (t_[i] >= tmin) {
                xs.push_back(t_[i]);
                ys.push_back(v_[i]);
            }
        }
    }

    float MaxSince(double tmin) const {
        std::lock_guard lk(mu_);
        float m = 0;
        for (size_t i = 0; i < t_.size(); ++i)
            if (t_[i] >= tmin) m = std::max(m, v_[i]);
        return m;
    }

    float Last() const { std::lock_guard lk(mu_); return last_; }
    bool HasData() const { std::lock_guard lk(mu_); return has_; }

private:
    mutable std::mutex mu_;
    size_t cap_;
    size_t head_ = 0;
    std::vector<float> t_, v_;
    float last_ = 0;
    bool has_ = false;
};
