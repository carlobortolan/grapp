# v1 make build configuration

# list all the headers of the module
set(MODULE_HEADERS v1.h)

set(LIBRARY_NAME v1)
set(EXECUTABLE_NAME runv1)

# Define a library named v1 which contains the code of grapp/v1
add_library(${LIBRARY_NAME} SHARED ${MODULE_HEADERS})

# Register ${CMAKE CURRENT SOURCE DIR} as include directory for v1 so headers in that directory can be included via <headername.h>
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)

# Define an executable named runv1 for the code of run.cpp
add_executable(${EXECUTABLE_NAME} run.cpp)

# Link the library v1 into runv1
target_link_libraries(${EXECUTABLE_NAME} ${LIBRARY_NAME})
