# libarcade

A Rust interface to the Arcade Learning Environment.

## Building

Clone the fork of ALE with namespaced C FFI:

    ./clone-reqs.sh

In the ALE directory, build ALE using CMake (you may need to adjust the
directory of SDL in `CMakeLists.txt`):

    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .

Copy `libale_cffi_static.a` into `artifacts.x86-64-unknown-linux-gnu`, then
run `cargo build --release`.

## TODO

* [ ] Automate the build of ALE with CMake using a fancier build script.
