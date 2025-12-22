cmake_minimum_required(VERSION 3.15...3.31)
project(geocell LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Get NumPy include directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# H3 library paths
set(H3_INCLUDE "${CMAKE_SOURCE_DIR}/src/h3lib/build/src/h3lib/include")
set(H3_LIB "${CMAKE_SOURCE_DIR}/src/h3lib/build/lib/libh3.a")

# Check if H3 library exists
if(NOT EXISTS ${H3_LIB})
    message(FATAL_ERROR "H3 library not found at ${H3_LIB}. Please run 'make h3lib' first.")
endif()

# Cython modules in h3 subdirectory
set(CYTHON_MODULES
    indexing
    indexing_vec
    # inspection
    # hierarchy
    # traversal
    # edges
    # vertices
    # measurements
)

# Function to compile .pyx to .c and build extension
function(add_cython_extension module_name)
    set(pyx_file "${CMAKE_SOURCE_DIR}/src/geocell/_cython/h3/${module_name}.pyx")
    set(c_file "${CMAKE_CURRENT_BINARY_DIR}/${module_name}.c")
    
    # Run cython to generate .c file
    add_custom_command(
        OUTPUT ${c_file}
        COMMAND Python::Interpreter -m cython
            "${pyx_file}"
            --output-file "${c_file}"
            -3
            -I "${CMAKE_SOURCE_DIR}/src/geocell/_cython/h3"
            -I "${H3_INCLUDE}"
        DEPENDS ${pyx_file}
        VERBATIM
    )
    
    # Create extension module
    python_add_library(${module_name} MODULE ${c_file} WITH_SOABI)
    
    target_include_directories(${module_name} PRIVATE
        ${NUMPY_INCLUDE_DIR}
        ${H3_INCLUDE}
        "${CMAKE_SOURCE_DIR}/src/geocell/_cython/h3"
    )
    
    target_link_libraries(${module_name} PRIVATE ${H3_LIB})
    
    # Platform-specific flags for Python extensions
    if(APPLE)
        # macOS: Force bundle (MODULE) instead of dylib (SHARED)
        set_target_properties(${module_name} PROPERTIES
            PREFIX ""
            LINK_FLAGS "-bundle -flat_namespace -undefined suppress"
        )
    elseif(WIN32)
        # Windows: Python extensions are DLLs with .pyd extension
        set_target_properties(${module_name} PROPERTIES
            SUFFIX ".pyd"
        )
    else()
        # Linux/Unix: Standard shared library settings (default behavior works)
        set_target_properties(${module_name} PROPERTIES
            PREFIX ""
        )
    endif()
    
    install(TARGETS ${module_name} 
            LIBRARY DESTINATION geocell/_cython/h3
            COMPONENT python)
endfunction()

# Add all modules
foreach(module ${CYTHON_MODULES})
    add_cython_extension(${module})
endforeach()

