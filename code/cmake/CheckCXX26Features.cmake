# =============================================================================
# C++26 Feature Detection Macros
#
# Uses direct compiler invocation (execute_process) instead of
# check_cxx_source_compiles/try_compile to bypass CMake 3.28's
# inability to handle CXX26 dialect in subprojects.
#
# Output goes to:
#   ${CMAKE_CURRENT_BINARY_DIR}/generated/hq/cxx26_features.hpp
# =============================================================================

# We already require GCC 15+ or Clang 19+ in the parent CMakeLists.txt.
# These compilers natively support C++26. The feature tests below are
# sanity checks that confirm the compiler is correctly installed.

set(CMAKE_REQUIRED_FLAGS "-std=c++26")
message(STATUS "CMAKE_REQUIRED_FLAGS = ${CMAKE_REQUIRED_FLAGS}")

# Helper: compile a snippet and return TRUE if it succeeds
function(_cerberus_check_feature name snippet out_var)
    # Replace reserved characters for Windows filenames (: is drive-letter separator)
    string(REPLACE ":" "_" safe_name "${name}")
    set(src "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/_cerberus_check_${safe_name}.cpp")
    file(WRITE "${src}" "${snippet}")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" ${CMAKE_REQUIRED_FLAGS} -c "${src}" -o "${src}.o"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
    )
    if(result EQUAL 0)
        set(${out_var} TRUE PARENT_SCOPE)
        message(STATUS "  ${name}: YES")
    else()
        set(${out_var} FALSE PARENT_SCOPE)
        message(STATUS "  ${name}: NO")
        if(err)
            string(REPLACE "\n" ";" err_lines "${err}")
            list(LENGTH err_lines n)
            if(n GREATER 5)
                list(SUBLIST err_lines 0 5 err_lines)
            endif()
            foreach(line IN LISTS err_lines)
                message(STATUS "    > ${line}")
            endforeach()
        endif()
    endif()
endfunction()

# --- std::expected (<expected>) ---
_cerberus_check_feature("std::expected" "
    #include <expected>
    #include <string>
    std::expected<int, std::string> f() { return 42; }
    int main() { auto e = f(); if (e) { return *e - 42; } return 1; }
" UM790_HAS_STD_EXPECTED)

if(NOT UM790_HAS_STD_EXPECTED)
    message(FATAL_ERROR
        "std::expected is required but not available. "
        "Ensure GCC >= 15 or Clang >= 19 with C++26 enabled.")
endif()

# --- std::print (<print>) ---
_cerberus_check_feature("std::print" "
    #include <print>
    int main() { std::print(\"hello {}\\n\", 42); return 0; }
" UM790_HAS_STD_PRINT)

if(NOT UM790_HAS_STD_PRINT)
    message(WARNING
        "std::print is required but not available. "
        "Ensure GCC >= 15 or Clang >= 19 with C++26 enabled.")
endif()

# --- std::mdspan (<mdspan>) ---
_cerberus_check_feature("std::mdspan" "
    #include <version>
    #if !defined(__cpp_lib_mdspan) || __cpp_lib_mdspan < 202207L
    #  error \"std::mdspan feature macro not found\"
    #endif
    #include <mdspan>
    int main() {
        float buf[12] = {};
        std::mdspan<float, std::dextents<std::size_t, 2>> m(buf, 3, 4);
        return static_cast<int>(m.extent(0)) - 3;
    }
" UM790_HAS_STD_MDSPAN)

if(NOT UM790_HAS_STD_MDSPAN)
    message(WARNING
        "std::mdspan unavailable — staging buffer views will use raw spans.")
endif()

# --- std::format (<format>) ---
_cerberus_check_feature("std::format" "
    #include <format>
    #include <string>
    int main() {
        auto s = std::format(\"{} + {} = {}\", 2, 3, 5);
        return s.empty() ? 1 : 0;
    }
" UM790_HAS_STD_FORMAT)

if(NOT UM790_HAS_STD_FORMAT)
    message(FATAL_ERROR
        "std::format is required but not available. "
        "GCC >= 15 or Clang >= 19 with C++26 is required.")
endif()

# --- Coroutines (<coroutine>) ---
_cerberus_check_feature("std::coroutine" "
    #include <coroutine>
    struct task {
        struct promise_type {
            task get_return_object() { return {}; }
            std::suspend_never initial_suspend() { return {}; }
            std::suspend_never final_suspend() noexcept { return {}; }
            void return_void() {}
            void unhandled_exception() {}
        };
    };
    task coro() { co_return; }
    int main() { coro(); return 0; }
" UM790_HAS_STD_COROUTINE)

set(UM790_HAS_COROUTINES ${UM790_HAS_STD_COROUTINE})
if(NOT UM790_HAS_STD_COROUTINE)
    message(WARNING
        "Coroutine support unavailable — async pipelines may be degraded. "
        "Requires GCC >= 15 or Clang >= 19.")
endif()

# --- Report ---
message(STATUS "  C++26 features detected:")
message(STATUS "    std::expected : ${UM790_HAS_STD_EXPECTED}")
message(STATUS "    std::print    : ${UM790_HAS_STD_PRINT}")
message(STATUS "    std::mdspan   : ${UM790_HAS_STD_MDSPAN}")
message(STATUS "    std::format   : ${UM790_HAS_STD_FORMAT}")
message(STATUS "    std::coroutine: ${UM790_HAS_STD_COROUTINE}")
message(STATUS "    coroutines    : ${UM790_HAS_COROUTINES}")

# Export results for C++ source
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cxx26_features.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/hq/cxx26_features.hpp
    @ONLY
)
