use super::{ArcadeContext, ArcadeSavedState};

use genrl::env::{Env, Action, DiscreteAction, EnvRepr, Discounted, NormalizeDiscounted};

use rand::{Rng};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::rc::{Rc};

const NUM_ACTIONS:  usize = 18;
const MAX_ACTION:   u32   = 18;

#[derive(Clone)]
pub struct ArcadeConfig {
  pub skip_frames:  usize,
  pub rom_path:     PathBuf,
  pub context:      Option<Rc<RefCell<ArcadeContext>>>,
}

impl Default for ArcadeConfig {
  fn default() -> ArcadeConfig {
    ArcadeConfig{
      skip_frames:  4,
      rom_path:     PathBuf::from(""),
      context:      None,
    }
  }
}

#[derive(Clone, Copy)]
pub struct ArcadeAction {
  id:   i32,
}

impl Action for ArcadeAction {
  fn dim() -> usize {
    NUM_ACTIONS
  }
}

impl DiscreteAction for ArcadeAction {
  fn from_idx(idx: u32) -> ArcadeAction {
    assert!(idx < MAX_ACTION);
    ArcadeAction{id: idx as i32}
  }

  fn idx(&self) -> u32 {
    self.id as u32
  }
}

pub trait ArcadeFeatures: Clone + Default {
  fn reset(&mut self);
  fn update(&mut self, context: &mut ArcadeContext);
}

#[derive(Clone)]
pub struct RamArcadeFeatures {
  stride:   usize,
  window:   usize,
  obs_buf:  Vec<u8>,
}

impl Default for RamArcadeFeatures {
  fn default() -> RamArcadeFeatures {
    let mut obs_buf = Vec::with_capacity(128 * 4);
    for _ in 0 .. 128 * 4 {
      obs_buf.push(0);
    }
    RamArcadeFeatures{
      stride:   128,
      window:   4,
      obs_buf:  obs_buf,
    }
  }
}

impl ArcadeFeatures for RamArcadeFeatures {
  fn reset(&mut self) {
    for p in 0 .. self.stride * self.window {
      self.obs_buf[p] = 0;
    }
  }

  fn update(&mut self, context: &mut ArcadeContext) {
    let ram_sz = context.ram_size();
    assert_eq!(self.stride, ram_sz);
    for frame in (0 .. self.window-1).rev() {
      let (pre_buf, post_buf) = self.obs_buf.split_at_mut((frame+1) * self.stride);
      post_buf[ .. self.stride].copy_from_slice(&pre_buf[frame * self.stride .. ]);
    }
    context.extract_ram(&mut self.obs_buf[ .. self.stride]);
  }
}

#[derive(Clone)]
pub struct GrayArcadeFeatures {
  width:    usize,
  height:   usize,
  window:   usize,
  obs_buf:  Vec<u8>,
}

impl Default for GrayArcadeFeatures {
  fn default() -> GrayArcadeFeatures {
    let mut obs_buf = Vec::with_capacity(210 * 160 * 4);
    for _ in 0 .. 210 * 160 * 4 {
      obs_buf.push(0);
    }
    GrayArcadeFeatures{
      width:    210,
      height:   160,
      window:   4,
      obs_buf:  obs_buf,
    }
  }
}

impl ArcadeFeatures for GrayArcadeFeatures {
  fn reset(&mut self) {
    for p in 0 .. self.width * self.height * self.window {
      self.obs_buf[p] = 0;
    }
  }

  fn update(&mut self, context: &mut ArcadeContext) {
    let frame_sz = context.screen_size();
    assert_eq!(self.width * self.height, frame_sz);
    for frame in (0 .. self.window-1).rev() {
      let (pre_buf, post_buf) = self.obs_buf.split_at_mut((frame+1) * frame_sz);
      post_buf[ .. frame_sz].copy_from_slice(&pre_buf[frame * frame_sz .. ]);
    }
    context.extract_screen_grayscale(&mut self.obs_buf[ .. frame_sz]);
  }
}

#[derive(Clone)]
pub struct RgbArcadeFeatures {
  width:    usize,
  height:   usize,
  window:   usize,
  obs_buf:  Vec<u8>,
}

impl Default for RgbArcadeFeatures {
  fn default() -> RgbArcadeFeatures {
    let mut obs_buf = Vec::with_capacity(210 * 160 * 4);
    for _ in 0 .. 210 * 160 * 4 {
      obs_buf.push(0);
    }
    RgbArcadeFeatures{
      width:    210,
      height:   160,
      window:   4,
      obs_buf:  obs_buf,
    }
  }
}

impl ArcadeFeatures for RgbArcadeFeatures {
  fn reset(&mut self) {
    for p in 0 .. self.width * self.height * 3 * self.window {
      self.obs_buf[p] = 0;
    }
  }

  fn update(&mut self, context: &mut ArcadeContext) {
    let frame_sz = context.screen_size() * 3;
    assert_eq!(self.width * self.height * 3, frame_sz);
    for frame in (0 .. self.window-1).rev() {
      let (pre_buf, post_buf) = self.obs_buf.split_at_mut((frame+1) * frame_sz);
      post_buf[ .. frame_sz].copy_from_slice(&pre_buf[frame * frame_sz .. ]);
    }
    context.extract_screen_rgb(&mut self.obs_buf[ .. frame_sz]);
  }
}

pub struct ArcadeEnv<F> {
  cfg:      ArcadeConfig,
  context:  Option<Rc<RefCell<ArcadeContext>>>,
  state:    Option<ArcadeSavedState>,
  features: F,
}

impl<F> Default for ArcadeEnv<F> where F: ArcadeFeatures {
  fn default() -> ArcadeEnv<F> {
    ArcadeEnv{
      cfg:      Default::default(),
      context:  None,
      state:    None,
      features: Default::default(),
    }
  }
}

impl<F> Clone for ArcadeEnv<F> where F: ArcadeFeatures {
  fn clone(&self) -> ArcadeEnv<F> {
    if let Some(ref ctx) = self.context {
      let state = ctx.borrow_mut().save_system_state();
      ArcadeEnv{
        cfg:      self.cfg.clone(),
        context:  Some(ctx.clone()),
        state:    Some(state),
        features: self.features.clone(),
      }
    } else {
      ArcadeEnv{
        cfg:      self.cfg.clone(),
        context:  None,
        state:    None,
        features: self.features.clone(),
      }
    }
  }
}

impl<F> Env for ArcadeEnv<F> where F: ArcadeFeatures {
  type Init = ArcadeConfig;
  type Action = ArcadeAction;
  //type Response = f32;
  type Response = Discounted<f32>;
  //type Response = NormalizeDiscounted<f32>;

  fn reset<R>(&mut self, init: &ArcadeConfig, rng: &mut R) where R: Rng {
    let new_rom = init.rom_path != self.cfg.rom_path;
    self.cfg = init.clone();
    self.context = self.cfg.context.clone();
    if new_rom {
      assert!(self.context.as_ref().unwrap().borrow_mut().open_rom(&self.cfg.rom_path, false).is_ok());
    }
    self.context.as_ref().unwrap().borrow_mut().reset();
    self.state = Some(self.context.as_ref().unwrap().borrow_mut().save_system_state());
    self.features.reset();
  }

  fn is_terminal(&mut self) -> bool {
    let is_term = self.context.as_ref().unwrap().borrow_mut().is_game_over();
    is_term
  }

  fn is_legal_action(&mut self, action: &ArcadeAction) -> bool {
    true
  }

  //fn step(&mut self, action: &ArcadeAction) -> Result<Option<f32>, ()> {
  fn step(&mut self, action: &ArcadeAction) -> Result<Option<Discounted<f32>>, ()> {
  //fn step(&mut self, action: &ArcadeAction) -> Result<Option<NormalizeDiscounted<f32>>, ()> {
    if let Some(ref state) = self.state {
      self.context.as_ref().unwrap().borrow_mut().load_system_state(state);
    }
    let mut res = 0;
    for _ in 0 .. self.cfg.skip_frames {
      res += self.context.as_ref().unwrap().borrow_mut().act(action.id);
    }
    self.state = Some(self.context.as_ref().unwrap().borrow_mut().save_system_state());
    self.features.update(&mut *self.context.as_ref().unwrap().borrow_mut());
    //Ok(Some(res as f32))
    Ok(Some(Discounted::new(res as f32, 0.99)))
    //Ok(Some(NormalizeDiscounted::new(res as f32, 0.99)))
  }
}

impl EnvRepr<u8> for ArcadeEnv<RamArcadeFeatures> {
  fn observable_len(&mut self) -> usize {
    128 * 4
  }

  fn extract_observable(&mut self, obs_buf: &mut [u8]) {
    obs_buf.copy_from_slice(&self.features.obs_buf)
  }
}

impl EnvRepr<f32> for ArcadeEnv<RamArcadeFeatures> {
  fn observable_len(&mut self) -> usize {
    128 * 4
  }

  fn extract_observable(&mut self, obs_buf: &mut [f32]) {
    for p in 0 .. 128 * 4 {
      obs_buf[p] = self.features.obs_buf[p] as f32;
    }
  }
}

impl EnvRepr<f32> for ArcadeEnv<GrayArcadeFeatures> {
  fn observable_len(&mut self) -> usize {
    210 * 160 * 4
  }

  fn extract_observable(&mut self, obs_buf: &mut [f32]) {
    for p in 0 .. 210 * 160 * 4 {
      obs_buf[p] = self.features.obs_buf[p] as f32;
    }
  }
}
