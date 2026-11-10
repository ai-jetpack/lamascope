cmake_minimum_required(VERSION 3.25)
project(LlamaScope LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT: single exe, no VC++ redistributable needed.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(TP ${CMAKE_SOURCE_DIR}/third_party)

file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)

set(IMGUI_SOURCES
    ${TP}/imgui/imgui.cpp
    ${TP}/imgui/imgui_draw.cpp
    ${TP}/imgui/imgui_tables.cpp
    ${TP}/imgui/imgui_widgets.cpp
    ${TP}/imgui/backends/imgui_impl_win32.cpp
    ${TP}/imgui/backends/imgui_impl_dx11.cpp
    ${TP}/implot/implot.cpp
    ${TP}/implot/implot_items.cpp
)

add_executable(LlamaScope WIN32 ${APP_SOURCES} ${IMGUI_SOURCES})

target_include_directories(LlamaScope PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${TP}/imgui
    ${TP}/imgui/backends
    ${TP}/implot
    ${TP}/json
    ${TP}/miniz
)

target_compile_definitions(LlamaScope PRIVATE
    UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
)

target_compile_options(LlamaScope PRIVATE /W3 /utf-8 /permissive- /Zc:__cplusplus)

target_link_libraries(LlamaScope PRIVATE windowscodecs
    d3d11 dxgi d3dcompiler dwmapi pdh powrprof advapi32 shell32 ole32 comdlg32
)
