use std::env;
use std::path::{PathBuf};

fn main() {
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let mut artifacts_path = PathBuf::from(&manifest_dir);
  artifacts_path.push("artifacts.x86-64-unknown-linux-gnu");
  println!("cargo:rustc-link-search=native={}", artifacts_path.to_str().unwrap());
}
