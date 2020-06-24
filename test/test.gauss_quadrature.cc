#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <limits>

#include "fmt/format.h"
#include "gauss_quadrature_rule.hpp"

// Epsilon of double type.
static constexpr double eps_double = std::numeric_limits<double>::epsilon();

namespace gaussLegendreQuadrature {
    // Points and weights of Gauss-Legendre quadrature.
    // https://keisan.casio.com/exec/system/1280624821
    // http://mathworld.wolfram.com/Legendre-GaussQuadrature.html
    // Number of point 2
    static const double point_2[]{-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};
    static const double weight_2[]{1.0, 1.0};
    // Number of point 3
    static const double point_3[]{-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
    static const double weight_3[]{5.0/9.0, 8.0/9.0, 5.0/9.0};
    // Number of point 4
    static const double point_4[]{-std::sqrt(5.25e2+7.0e1*std::sqrt(3.0e1))/3.5e1, -std::sqrt(5.25e2-7.0e1*std::sqrt(3.0e1))/3.5e1,
        std::sqrt(5.25e2-7.0e1*std::sqrt(3.0e1))/3.5e1, std::sqrt(5.25e2+7.0e1*std::sqrt(3.0e1))/3.5e1};
    static const double weight_4[]{0.5-std::sqrt(3.0e1)/3.6e1, 0.5+std::sqrt(3.0e1)/3.6e1,0.5+std::sqrt(3.0e1)/3.6e1,0.5-std::sqrt(3.0e1)/3.6e1};
    // Number of point 5
    static const double point_5[]{-std::sqrt(2.45e2+1.4e1*std::sqrt(7.0e1))/2.1e1, -std::sqrt(2.45e2-1.4e1*std::sqrt(7.0e1))/2.1e1, 0.0,
        std::sqrt(2.45e2-1.4e1*std::sqrt(7.0e1))/2.1e1, std::sqrt(2.45e2+1.4e1*std::sqrt(7.0e1))/2.1e1};
    static const double weight_5[]{(3.22e2-1.3e1*std::sqrt(7.0e1))/9.0e2, (3.22e2+1.3e1*std::sqrt(7.0e1))/9.0e2, 1.28/2.25,
        (3.22e2+1.3e1*std::sqrt(7.0e1))/9.0e2, (3.22e2-1.3e1*std::sqrt(7.0e1))/9.0e2};
}

TEST_CASE("Compare quadrature point and weight", "[Gauss Legendre]") {
    using namespace gaussLegendreQuadrature;
    // Convert double value to string.
    auto val2str = [](double a){return fmt::format("{:+15.12f}", a);};
    auto [pnt2, wht2] = NR::gauss_legendre<2>();
    for(auto idx: {0, 1}) {
        REQUIRE(val2str(pnt2[idx])==val2str(point_2[idx]));
        REQUIRE(val2str(wht2[idx])==val2str(weight_2[idx]));
    }
    auto [pnt3, wht3] = NR::gauss_legendre<3>();
    for(auto idx: {0, 1, 2}) {
        REQUIRE(val2str(pnt3[idx]+1.0)==val2str(point_3[idx]+1.0)); // Avoid '+0.0' compare with '-0.0'.
        REQUIRE(val2str(wht3[idx])==val2str(weight_3[idx]));
    }
    auto [pnt4, wht4] = NR::gauss_legendre<4>();
    for(auto idx: {0, 1, 2, 3}) {
        REQUIRE(val2str(pnt4[idx])==val2str(point_4[idx]));
        REQUIRE(val2str(wht4[idx])==val2str(weight_4[idx]));                        
    }
    auto [pnt5, wht5] = NR::gauss_legendre<5>();
    for(auto idx: {0, 1, 3, 4}) {
        REQUIRE(val2str(pnt5[idx]+1.0)==val2str(point_5[idx]+1.0));
        REQUIRE(val2str(wht5[idx])==val2str(weight_5[idx]));
    }
}

TEST_CASE("Benchmark", "[Gauss Legendre]") {
    BENCHMARK("Gauss-Legendre  10 points") { return NR::gauss_legendre<10>();};
    BENCHMARK("Gauss-Legendre  20 points") { return NR::gauss_legendre<20>();};
    BENCHMARK("Gauss-Legendre  40 points") { return NR::gauss_legendre<40>();};
    BENCHMARK("Gauss-Legendre  80 points") { return NR::gauss_legendre<80>();};
    // BENCHMARK("Gauss-Legendre 160 points") { return NR::gauss_legendre<160>();};
    // BENCHMARK("Gauss-Legendre 320 points") { return NR::gauss_legendre<320>();};
}