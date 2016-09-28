#![feature(convert)]

extern crate libc;

use ffi::*;

use libc::{c_int};
use std::ffi::{CString};
use std::path::{Path};

pub mod ffi;

pub struct ArcadeContext {
  ptr:  *mut ALEInterface,
}

impl Drop for ArcadeContext {
  fn drop(&mut self) {
    unsafe { ALEInterface_delete(self.ptr) };
  }
}

impl ArcadeContext {
  pub fn new() -> ArcadeContext {
    ArcadeContext{
      ptr:  unsafe { ALEInterface_new() },
    }
  }

  pub fn set_string(&mut self, key: &str, value: &str) {
    // FIXME(20160311)
    unimplemented!();
  }

  pub fn set_int(&mut self, key: &str, value: i32) {
    // FIXME(20160311)
    unimplemented!();
  }

  pub fn set_bool(&mut self, key: &str, value: bool) {
    // FIXME(20160311)
    unimplemented!();
  }

  pub fn open_rom(&mut self, path: &Path) -> Result<(), ()> {
    //let path_osstr = path.as_os_str();
    let path_cstr = match CString::new(path.to_str().unwrap().to_owned()) {
      Ok(cstr) => cstr,
      Err(_) => return Err(()),
    };
    unsafe { ALEInterface_loadROM(self.ptr, path_cstr.as_ptr()) };
    Ok(())
  }

  pub fn reset(&mut self) {
    unsafe { ALEInterface_reset_game(self.ptr) };
  }

  pub fn is_game_over(&mut self) -> bool {
    match unsafe { ALEInterface_game_over(self.ptr) } {
      0 => false,
      _ => true,
    }
  }

  pub fn num_lives(&mut self) -> i32 {
    unsafe { ALEInterface_lives(self.ptr) }
  }

  pub fn frame_number(&mut self) -> i32 {
    unsafe { ALEInterface_getFrameNumber(self.ptr) }
  }

  pub fn episode_frame_number(&mut self) -> i32 {
    unsafe { ALEInterface_getEpisodeFrameNumber(self.ptr) }
  }

  pub fn num_legal_actions(&mut self) -> usize {
    unsafe { ALEInterface_getLegalActionSize(self.ptr) as usize }
  }

  pub fn extract_legal_action_set(&mut self, action_set: &mut [i32]) -> usize {
    let num_actions = self.num_legal_actions();
    assert!(action_set.len() >= num_actions);
    unsafe { ALEInterface_getLegalActionSet(self.ptr, action_set.as_mut_ptr()) };
    num_actions
  }

  pub fn num_minimal_actions(&mut self) -> usize {
    unsafe { ALEInterface_getMinimalActionSize(self.ptr) as usize }
  }

  pub fn extract_minimal_action_set(&mut self, action_set: &mut [i32]) -> usize {
    let num_actions = self.num_minimal_actions();
    assert!(action_set.len() >= num_actions);
    unsafe { ALEInterface_getMinimalActionSet(self.ptr, action_set.as_mut_ptr()) };
    num_actions
  }

  pub fn act(&mut self, action: i32) -> i32 {
    unsafe { ALEInterface_act(self.ptr, action) }
  }

  pub fn ram_size(&mut self) -> usize {
    unsafe { ALEInterface_getRAMSize(self.ptr) as usize }
  }

  pub fn extract_ram(&mut self, buf: &mut [u8]) {
    assert!(buf.len() >= self.ram_size());
    unsafe { ALEInterface_getRAM(self.ptr, buf.as_mut_ptr()) };
  }

  pub fn screen_width(&mut self) -> usize {
    unsafe { ALEInterface_getScreenWidth(self.ptr) as usize }
  }

  pub fn screen_height(&mut self) -> usize {
    unsafe { ALEInterface_getScreenHeight(self.ptr) as usize }
  }

  pub fn screen_size(&mut self) -> usize {
    self.screen_width() * self.screen_height()
  }

  pub fn extract_screen_raw(&mut self, buf: &mut [u8]) {
    assert!(buf.len() >= self.screen_size());
    unsafe { ALEInterface_getScreen(self.ptr, buf.as_mut_ptr()) };
  }

  pub fn extract_screen_rgb(&mut self, buf: &mut [u8]) {
    assert!(buf.len() >= self.screen_size());
    unsafe { ALEInterface_getScreenRGB(self.ptr, buf.as_mut_ptr()) };
  }

  pub fn extract_screen_grayscale(&mut self, buf: &mut [u8]) {
    assert!(buf.len() >= self.screen_size());
    unsafe { ALEInterface_getScreenGrayscale(self.ptr, buf.as_mut_ptr()) };
  }

  pub fn dump_screen_png(&mut self, path: &Path) -> Result<(), ()> {
    //let path_osstr = path.as_os_str();
    let path_cstr = match CString::new(path.to_str().unwrap().to_owned()) {
      Ok(cstr) => cstr,
      Err(_) => return Err(()),
    };
    unsafe { ALEInterface_saveScreenPNG(self.ptr, path_cstr.as_ptr()) };
    Ok(())
  }

  pub fn push_state(&mut self) {
    unsafe { ALEInterface_saveState(self.ptr) };
  }

  pub fn pop_state(&mut self) {
    unsafe { ALEInterface_loadState(self.ptr) };
  }

  pub fn save_state(&mut self) -> ArcadeSavedState {
    ArcadeSavedState{
      ptr:  unsafe { ALEInterface_cloneState(self.ptr) },
    }
  }

  pub fn load_state(&mut self, state: &ArcadeSavedState) {
    unsafe { ALEInterface_restoreState(self.ptr, state.ptr) };
  }

  pub fn save_system_state(&mut self) -> ArcadeSavedState {
    ArcadeSavedState{
      ptr:  unsafe { ALEInterface_cloneSystemState(self.ptr) },
    }
  }

  pub fn load_system_state(&mut self, state: &ArcadeSavedState) {
    unsafe { ALEInterface_restoreSystemState(self.ptr, state.ptr) };
  }
}

pub struct ArcadeSavedState {
  ptr:  *mut ALEState,
}

impl Drop for ArcadeSavedState {
  fn drop(&mut self) {
    unsafe { ALEState_delete(self.ptr) };
  }
}

impl ArcadeSavedState {
  pub fn decode(buf: &[u8]) -> ArcadeSavedState {
    ArcadeSavedState{
      ptr:  unsafe { ALEState_decodeState(buf.as_ptr() as *const _, buf.len() as c_int) },
    }
  }

  pub fn encoded_size(&mut self) -> usize {
    unsafe { ALEState_encodeStateLen(self.ptr) as usize }
  }

  pub fn encode(&mut self, buf: &mut [u8]) {
    unsafe { ALEState_encodeState(self.ptr, buf.as_mut_ptr() as *mut _, buf.len() as c_int) };
  }
}
