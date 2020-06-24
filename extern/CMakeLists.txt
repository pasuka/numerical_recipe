include(FetchContent)

set(EXT_HEAD "Extern library: ")

if(TARGET Eigen3::Eigen)
    message(STATUS "Find Eigen3")
else()
    message(STATUS "${EXT_HEAD}Cannot find Eigen3")
    message(STATUS "${EXT_HEAD}Try to download Eigen3 from gitlab")
    FetchContent_Declare(
        Eigen3
        URL      https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
        URL_HASH MD5=9e30f67e8531477de4117506fe44669b # Hash of file not string of URL!
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

if(TARGET Catch2::Catch2)
    message(STATUS "Find Catch2")
else()
    message(STATUS "${EXT_HEAD}Cannot find Catch2")
    message(STATUS "${EXT_HEAD}Try to download Catch2 from github")
    FetchContent_Declare(
        Catch2
        URL      https://github.com/catchorg/Catch2/archive/v2.11.1.tar.gz
        URL_HASH MD5=4b097ffafd5d1102e961a14699f5b918 # Hash of file not string of URL!
    )
    FetchContent_MakeAvailable(Catch2)
endif()

if(TARGET fmt::fmt)
    message(STATUS "Find fmt")
else()
    message(STATUS "${EXT_HEAD}Cannot find fmt")
    message(STATUS "${EXT_HEAD}Try to download fmt from github")
    FetchContent_Declare(
        fmt
        URL      https://github.com/fmtlib/fmt/releases/download/6.1.2/fmt-6.1.2.zip
        URL_HASH MD5=5ec643b082a34254ad07136e7cbd4220 # Hash of file not string of URL!
    )
    FetchContent_MakeAvailable(fmt)
endif()

# #Get file's hash under bash
# md5=($(md5sum file_name))
# echo $md5