cmake_minimum_required(VERSION 3.20)

project(RHITA_MPM_3D LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -use_fast_maths")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(Eigen3 REQUIRED NO_MODULE HINTS "C:\\libraries\\eigen-build")
if(WIN32)
    add_compile_definitions(H5_BUILT_AS_DYNAMIC_LIB)
else()
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    find_package(spdlog REQUIRED)
endif()

find_package(VTK 9.1 HINTS "C:\\libraries\\vtk-build\\lib\\cmake\\vtk-9.1" REQUIRED COMPONENTS
  CommonColor
  CommonCore
  FiltersCore
  FiltersSources
  FiltersGeometry
  FiltersGeneral
  FiltersModeling
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
  IOGeometry
  RenderingAnnotation
  RenderingLabel
  IOImage
  IOXML
  ViewsQt
  CommonDataModel
  CommonTransforms
)



# Qt GUI executable
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(iceMPM_GUI_3D
	# gui
        gui/main.cpp

        gui/windows/mainwindow.cpp
        gui/windows/mainwindow.h
        gui/windows/mainwindow.ui
        icons.qrc

        gui/vtk/vtk_representation.cpp
        gui/vtk/vtk_representation.h

        gui/backgroundworker.cpp
        gui/backgroundworker.h
        gui/parameters_wrapper.h


        # simulation
        simulation/parameters_sim_3d.h
        simulation/parameters_sim_3d.cpp
        simulation/model_3d.cpp
        simulation/model_3d.h
        simulation/point_3d.h
        simulation/point_3d.cpp

	# cuda
        simulation/givens.cuh
        simulation/helper_math.cuh
        simulation/gpu_implementation4.h
        simulation/gpu_implementation4.cu
        simulation/constitutive_model.cu

	# saving files
        simulation/snapshotmanager.h
        simulation/snapshotmanager.cpp
    )

    set_target_properties(iceMPM_GUI_3D PROPERTIES CUDA_ARCHITECTURES "89")
    set_target_properties(iceMPM_GUI_3D PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(iceMPM_GUI_3D PRIVATE gui gui/windows gui/vtk simulation libraries/thinks/poisson_disk_sampling/)
    target_link_libraries(iceMPM_GUI_3D PRIVATE Qt5::Widgets Qt5::Core)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE HINTS "C:\\libraries\\eigen-build")
    target_link_libraries(iceMPM_GUI_3D PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(iceMPM_GUI_3D PRIVATE ${VTK_LIBRARIES})

    #PBrowser
    if(WIN32)
        target_link_directories(iceMPM_GUI_3D PRIVATE "C:\\libraries\\pbrowser-build")
        target_include_directories(iceMPM_GUI_3D PRIVATE "C:\\libraries\\PBrowser")
        target_include_directories(iceMPM_GUI_3D PRIVATE "C:\\libraries\\PBrowser\\propertybrowser")
        target_include_directories(iceMPM_GUI_3D PRIVATE "C:\\libraries\\PBrowser\\objectpropertybrowser")
    else()
        target_link_directories(iceMPM_GUI_3D PRIVATE "/home/s2/Projects-Libraries/build-PBrowser-Desktop_Qt_5_15_2_GCC_64bit-Release")
        target_include_directories(iceMPM_GUI_3D PRIVATE /home/s2/Projects-Libraries/PBrowser/objectpropertybrowser)
        target_include_directories(iceMPM_GUI_3D PRIVATE /home/s2/Projects-Libraries/PBrowser/propertybrowser)
        target_include_directories(iceMPM_GUI_3D PRIVATE /home/s2/Projects-Libraries/PBrowser)
    endif()
    target_link_libraries(iceMPM_GUI_3D PRIVATE PBrowser)


    # HDF5
    if(WIN32)
        target_link_directories(iceMPM_GUI_3D PRIVATE "C:\\libraries\\hdf5-build\\bin\\Release")
        target_link_libraries(iceMPM_GUI_3D PRIVATE hdf5_cpp hdf5 )
    else()
        target_link_libraries(iceMPM_GUI_3D PRIVATE ${HDF5_LIBRARIES})
        target_include_directories(iceMPM_GUI_3D PRIVATE ${HDF5_INCLUDE_DIRS})
    endif()

    # spdlog
    if (WIN32)
        target_link_libraries(iceMPM_GUI_3D PRIVATE spdlog)
        target_include_directories(iceMPM_GUI_3D PRIVATE "C:\\libraries\\spdlog\\include")
        target_link_directories(iceMPM_GUI_3D PRIVATE "C:\\libraries\\spdlog-build\\Release")
    else()
        target_link_libraries(iceMPM_GUI_3D PRIVATE spdlog::spdlog)
    endif()


    # OpenMP
        if(WIN32)
        target_compile_options(iceMPM_GUI_3D PRIVATE /openmp)
    else()
        target_compile_options(iceMPM_GUI_3D PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
        target_link_libraries(iceMPM_GUI_3D PRIVATE gomp)
    endif()

    vtk_module_autoinit(
      TARGETS iceMPM_GUI_3D
      MODULES ${VTK_LIBRARIES}
    )



# CLI version

add_executable(cm
    cli/main.cpp
    cli/cli_convert.cpp
    cli/cli_simulate.cpp

    # simulation
    simulation/parameters_sim_3d.h
    simulation/parameters_sim_3d.cpp
    simulation/model_3d.cpp
    simulation/model_3d.h
    simulation/point_3d.h
    simulation/point_3d.cpp

    # cuda
    simulation/givens.cuh
    simulation/helper_math.cuh
    simulation/gpu_implementation4.h
    simulation/gpu_implementation4.cu
    simulation/constitutive_model.cu

    # saving files
    simulation/snapshotmanager.h
    simulation/snapshotmanager.cpp
)


set_target_properties(cm PROPERTIES CUDA_ARCHITECTURES "89")
set_target_properties(cm PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(cm PRIVATE cli simulation)

# Eigen
target_link_libraries(cm PRIVATE Eigen3::Eigen)

# VTK
target_link_libraries(cm PRIVATE ${VTK_LIBRARIES})

# HDF5
if(WIN32)
    target_link_directories(cm PRIVATE "C:\\libraries\\hdf5-build\\bin\\Release")
    target_link_libraries(cm PRIVATE hdf5_cpp hdf5 )
else()
    target_link_libraries(cm PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(cm PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# spdlog
if (WIN32)
    target_link_libraries(cm PRIVATE spdlog)
    target_include_directories(cm PRIVATE "C:\\libraries\\spdlog\\include")
    target_link_directories(cm PRIVATE "C:\\libraries\\spdlog-build\\Release")
else()
    target_link_libraries(cm PRIVATE spdlog::spdlog)
endif()

# partio
target_link_libraries(cm PRIVATE /usr/local/lib/libpartio.so.1)
target_include_directories(cm PRIVATE /home/s2/Projects-Libraries/partio/src/lib/)


# OpenMP
    if(WIN32)
    target_compile_options(cm PRIVATE /openmp)
else()
    target_compile_options(cm PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(cm PRIVATE gomp)
endif()




vtk_module_autoinit(
  TARGETS cm
  MODULES ${VTK_LIBRARIES}
)


