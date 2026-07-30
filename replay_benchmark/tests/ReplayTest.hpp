// ReplayTest.hpp — Minimal header-only test framework for replay_benchmark.
//
// Inspired by Google Test but zero-dependency (only C++11 standard library).
// Supports:
//   - TEST(SuiteName, CaseName) { ... }
//   - EXPECT_EQ(a, b) / EXPECT_NE(a, b) / EXPECT_TRUE(x) / EXPECT_FALSE(x)
//   - EXPECT_NEAR(a, b, tol)
//   - ASSERT_* variants (return from the current test function on failure)
//   --filter SuiteName.CaseName to run a subset
//   Colored output, pass/fail counts, exit code
//
// Usage:
//   #include "ReplayTest.hpp"
//   TEST(PMU, SinglePassConsistency) { EXPECT_EQ(1, 1); }
//   int main(int argc, char** argv) { return ReplayTest::RunAll(argc, argv); }

#ifndef REPLAY_TEST_HPP
#define REPLAY_TEST_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>

namespace ReplayTest {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Result {
    int passed = 0;
    int failed = 0;
    bool fatalFailure = false;
    std::vector<std::string> failures;
};

inline Result& currentResult() {
    static Result r;
    return r;
}

inline const char* colorGreen()  { return "\033[32m"; }
inline const char* colorRed()    { return "\033[31m"; }
inline const char* colorYellow() { return "\033[33m"; }
inline const char* colorReset()  { return "\033[0m"; }

inline bool checkTrue(bool cond, const char* expr, const char* file, int line, bool fatal) {
    if (cond) return true;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: %sEXPECT_TRUE(%s) failed%s",
                  file, line, colorRed(), expr, colorReset());
    currentResult().failures.push_back(buf);
    if (fatal) currentResult().fatalFailure = true;
    return false;
}

template <typename T, typename U>
inline bool checkEq(const T& a, const U& b, const char* aStr, const char* bStr,
                    const char* file, int line, bool fatal) {
    if (a == b) return true;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: %sEXPECT_EQ(%s, %s) failed: %s != %s%s",
                  file, line, colorRed(), aStr, bStr, aStr, bStr, colorReset());
    currentResult().failures.push_back(buf);
    if (fatal) currentResult().fatalFailure = true;
    return false;
}

template <typename T, typename U>
inline bool checkNe(const T& a, const U& b, const char* aStr, const char* bStr,
                    const char* file, int line, bool fatal) {
    if (a != b) return true;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: %sEXPECT_NE(%s, %s) failed: both equal%s",
                  file, line, colorRed(), aStr, bStr, colorReset());
    currentResult().failures.push_back(buf);
    if (fatal) currentResult().fatalFailure = true;
    return false;
}

template <typename T, typename U>
inline bool checkNear(const T& a, const U& b, double tol,
                      const char* aStr, const char* bStr,
                      const char* file, int line, bool fatal) {
    double diff = std::fabs(static_cast<double>(a) - static_cast<double>(b));
    if (diff <= tol) return true;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%d: %sEXPECT_NEAR(%s, %s, %f) failed: |diff|=%f > tol%s",
                  file, line, colorRed(), aStr, bStr, tol, diff, colorReset());
    currentResult().failures.push_back(buf);
    if (fatal) currentResult().fatalFailure = true;
    return false;
}

class Registrar {
public:
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        registry().push_back({suite, name, std::move(fn)});
    }
};

inline int RunAll(int argc, char** argv) {
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter = argv[++i];
        }
    }

    int total = 0;
    int passed = 0;
    int failed = 0;

    for (const auto& t : registry()) {
        std::string full = t.suite + "." + t.name;
        if (!filter.empty() && full.find(filter) == std::string::npos)
            continue;

        total++;
        currentResult().fatalFailure = false;
        currentResult().failures.clear();

        printf("[ RUN      ] %s\n", full.c_str());
        t.fn();

        if (!currentResult().fatalFailure && currentResult().failures.empty()) {
            passed++;
            printf("%s[       OK ]%s %s\n", colorGreen(), colorReset(), full.c_str());
        } else {
            failed++;
            printf("%s[  FAILED  ]%s %s\n", colorRed(), colorReset(), full.c_str());
            for (const auto& f : currentResult().failures) {
                printf("  %s\n", f.c_str());
            }
        }
    }

    printf("\n");
    printf("%s[==========]%s %d tests ran.\n", colorGreen(), colorReset(), total);
    printf("%s[  PASSED  ]%s %d tests.\n", colorGreen(), colorReset(), passed);
    if (failed > 0) {
        printf("%s[  FAILED  ]%s %d tests.\n", colorRed(), colorReset(), failed);
    }
    return failed > 0 ? 1 : 0;
}

} // namespace ReplayTest

// Macros — usage: TEST(SuiteName, CaseName) { ... }
#define TEST(suite, name) \
    static void suite##_##name##_fn(); \
    static ReplayTest::Registrar suite##_##name##_reg(#suite, #name, suite##_##name##_fn); \
    static void suite##_##name##_fn()

#define EXPECT_TRUE(x) do { ReplayTest::checkTrue((x), #x, __FILE__, __LINE__, false); } while (0)
#define EXPECT_FALSE(x) do { ReplayTest::checkTrue(!(x), #x, __FILE__, __LINE__, false); } while (0)
#define EXPECT_EQ(a, b) do { ReplayTest::checkEq((a), (b), #a, #b, __FILE__, __LINE__, false); } while (0)
#define EXPECT_NE(a, b) do { ReplayTest::checkNe((a), (b), #a, #b, __FILE__, __LINE__, false); } while (0)
#define EXPECT_NEAR(a, b, tol) \
    do { ReplayTest::checkNear((a), (b), (tol), #a, #b, __FILE__, __LINE__, false); } while (0)

#define ASSERT_TRUE(x) do { if (!ReplayTest::checkTrue((x), #x, __FILE__, __LINE__, true)) return; } while (0)
#define ASSERT_FALSE(x) do { if (!ReplayTest::checkTrue(!(x), #x, __FILE__, __LINE__, true)) return; } while (0)
#define ASSERT_EQ(a, b) do { if (!ReplayTest::checkEq((a), (b), #a, #b, __FILE__, __LINE__, true)) return; } while (0)
#define ASSERT_NE(a, b) do { if (!ReplayTest::checkNe((a), (b), #a, #b, __FILE__, __LINE__, true)) return; } while (0)

#endif // REPLAY_TEST_HPP
