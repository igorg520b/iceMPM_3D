# iceMPM
Implementation of the Material Point Method for modeling ice

![Screenshot of the GUI version](/screenshot.png)

##Required libraries:

> libeigen3-dev libspdlog-dev libcxxopts-dev rapidjson-dev libhdf5-dev libvtk9-dev

Edit CMakeLists.txt to select the compute capability of the NVIDA GPU (default is 8.9).

There is a CLI version and a GUI version with similar functionality. Output files are in HDF5 format and need to be converted to be viewed in Paraview, e.g.,

> cm --convert default_output/output_h5 -v -i

If the output was saved in "raw" format (--export-raw), then the frames can be converted in parallel via:

> cm --convert-parallel default_output/output_raw_h5 -v -i

##Input files

A simulation can be started with a JSON configuration file via:

> cm --simulate startfile.json

Note that the point set in HDF5 format should be generated and placed somewhere along with the JSON configuration file. The point cloud can be generated using [this tool](https://github.com/igorg520b/GrainIdentifier).

A simulation can be "resumed" from a full snapshot via:

> cm --resume snapshot_file.h5

A snapshot is also and HDF5-file, but is different from the point cloud, as it contains the complete simulation data at a given timestep.
