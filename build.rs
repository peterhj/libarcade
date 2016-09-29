use std::process::{Command};
use std::env;
use std::path::{PathBuf};

fn main() {
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = env::var("OUT_DIR").unwrap();

  let mut ale_lib_dst_path = PathBuf::from(&out_dir);
  ale_lib_dst_path.push("libale_cffi_static.a");
  if !ale_lib_dst_path.exists() {
    let mut ale_src_path = PathBuf::from(&manifest_dir);
    ale_src_path.push("Arcade-Learning-Environment");
    assert!(Command::new("cmake")
      .current_dir(&out_dir)
      .arg("-DUSE_SDL=ON")
      .arg("-DUSE_RLGLUE=OFF")
      .arg("-DBUILD_EXAMPLES=OFF")
      .arg(ale_src_path.to_str().unwrap())
      .status().unwrap().success());
    assert!(Command::new("make")
      .current_dir(&out_dir)
      .status().unwrap().success());
  }

  println!("cargo:rustc-link-search=native={}", out_dir);
}
