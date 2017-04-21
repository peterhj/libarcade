use libc::{c_char, c_uchar, c_int, c_float};

pub enum ALEInterface {}
pub enum ALEState {}

#[link(name = "stdc++")]
extern "C" {}

#[link(name = "SDL")]
extern "C" {}

#[link(name = "png")]
extern "C" {}

#[link(name = "z")]
extern "C" {}

#[link(name = "ale_ffi_static", kind = "static")]
extern "C" {
  pub fn ALEInterface_new() -> *mut ALEInterface;
  pub fn ALEInterface_delete(ale: *mut ALEInterface);
  pub fn ALEInterface_getString(ale: *mut ALEInterface, key: *const c_char) -> *const c_char;
  pub fn ALEInterface_getInt(ale: *mut ALEInterface, key: *const c_char) -> c_int;
  pub fn ALEInterface_getBool(ale: *mut ALEInterface, key: *const c_char) -> c_int;
  pub fn ALEInterface_getFloat(ale: *mut ALEInterface, key: *const c_char) -> c_float;
  pub fn ALEInterface_setString(ale: *mut ALEInterface, key: *const c_char, value: *const c_char);
  pub fn ALEInterface_setInt(ale: *mut ALEInterface, key: *const c_char, value: c_int);
  pub fn ALEInterface_setBool(ale: *mut ALEInterface, key: *const c_char, value: c_int);
  pub fn ALEInterface_setFloat(ale: *mut ALEInterface, key: *const c_char, value: c_float);
  pub fn ALEInterface_loadROM(ale: *mut ALEInterface, rom_file: *const c_char);
  pub fn ALEInterface_act(ale: *mut ALEInterface, action: c_int) -> c_int;
  pub fn ALEInterface_act2(ale: *mut ALEInterface, actionA: c_int, actionB: c_int, rewardA: *mut c_int, rewardB: *mut c_int);
  pub fn ALEInterface_game_over(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_reset_game(ale: *mut ALEInterface);
  pub fn ALEInterface_getLegalActionSet(ale: *mut ALEInterface, actions: *mut c_int);
  pub fn ALEInterface_getLegalActionSize(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_getMinimalActionSet(ale: *mut ALEInterface, actions: *mut c_int);
  pub fn ALEInterface_getMinimalActionSize(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_getFrameNumber(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_lives(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_getEpisodeFrameNumber(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_getScreen(ale: *mut ALEInterface, screen_data: *mut c_uchar);
  pub fn ALEInterface_getRAM(ale: *mut ALEInterface, ram: *mut c_uchar);
  pub fn ALEInterface_getRAMSize(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_getScreenWidth(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_getScreenHeight(ale: *mut ALEInterface) -> c_int;
  pub fn ALEInterface_getScreenRGB(ale: *mut ALEInterface, output_buffer: *mut c_uchar);
  pub fn ALEInterface_getScreenGrayscale(ale: *mut ALEInterface, output_buffer: *mut c_uchar);
  pub fn ALEInterface_saveState(ale: *mut ALEInterface);
  pub fn ALEInterface_loadState(ale: *mut ALEInterface);
  pub fn ALEInterface_cloneState(ale: *mut ALEInterface) -> *mut ALEState;
  pub fn ALEInterface_restoreState(ale: *mut ALEInterface, state: *mut ALEState);
  pub fn ALEInterface_cloneSystemState(ale: *mut ALEInterface) -> *mut ALEState;
  pub fn ALEInterface_restoreSystemState(ale: *mut ALEInterface, state: *mut ALEState);
  pub fn ALEState_delete(state: *mut ALEState);
  pub fn ALEInterface_saveScreenPNG(ale: *mut ALEInterface, filename: *const c_char);
  pub fn ALEState_encodeState(state: *mut ALEState, buf: *mut c_char, buf_len: c_int);
  pub fn ALEState_encodeStateLen(state: *mut ALEState) -> c_int;
  pub fn ALEState_decodeState(serialized: *const c_char, len: c_int) -> *mut ALEState;
  pub fn ALE_setLoggerMode(mode: c_int);
}
