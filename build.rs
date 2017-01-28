extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::path::{PathBuf};
use std::process::{Command};

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  for entry in WalkDir::new("Arcade-Learning-Environment") {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }

  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = env::var("OUT_DIR").unwrap();

  let cc = env::var("CC").unwrap_or(format!("gcc"));
  let cxx = env::var("CXX").unwrap_or(format!("g++"));

  let mut ale_lib_dst_path = PathBuf::from(&out_dir);
  ale_lib_dst_path.push("libale_cffi_static.a");

  {
    let mut ale_src_path = PathBuf::from(&manifest_dir);
    ale_src_path.push("Arcade-Learning-Environment");
    assert!(Command::new("cmake")
      .current_dir(&out_dir)
      .env("CC",  &cc)
      .env("CXX", &cxx)
      .arg("-DUSE_SDL=ON")
      .arg("-DUSE_RLGLUE=OFF")
      .arg("-DBUILD_EXAMPLES=OFF")
      .arg(ale_src_path.to_str().unwrap())
      .status().unwrap().success());
    assert!(Command::new("make")
      .current_dir(&out_dir)
      .arg("-j8")
      .status().unwrap().success());
  }

  println!("cargo:rustc-link-search=native={}", out_dir);
}
