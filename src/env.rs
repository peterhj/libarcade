use super::{CachedArcadeContext, ArcadeContext, ArcadeSavedState};

use genrl::env::{Env, Action, DiscreteAction, EnvInputRepr, EnvRepr, Discounted, NormalizeDiscounted};
use operator::prelude::*;

use rand::{Rng};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::rc::{Rc};

const NUM_ACTIONS:  usize = 18;
const MAX_ACTION:   u32   = 18;

#[derive(Clone, Debug)]
pub struct ArcadeConfig {
  pub skip_frames:  usize,
  pub noop_max:     usize,
  pub rom_path:     PathBuf,
  //pub cache:        Option<Rc<RefCell<CachedArcadeContext>>>,
}

impl Default for ArcadeConfig {
  fn default() -> ArcadeConfig {
    ArcadeConfig{
      skip_frames:  4,
      noop_max:     30,
      rom_path:     PathBuf::from(""),
      //cache:        None,
    }
  }
}

#[derive(Clone, Copy)]
pub struct ArcadeAction {
  id:   i32,
}

impl ArcadeAction {
  pub fn noop() -> ArcadeAction {
    ArcadeAction{id: 0}
  }
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
      width:    160,
      height:   210,
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
    let mut obs_buf = Vec::with_capacity(210 * 160 * 12);
    for _ in 0 .. 210 * 160 * 12 {
      obs_buf.push(0);
    }
    RgbArcadeFeatures{
      width:    160,
      height:   210,
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

pub struct ArcadeEnvInner<F> {
  cfg:      ArcadeConfig,
  cache:    Option<Rc<RefCell<CachedArcadeContext>>>,
  state:    Option<ArcadeSavedState>,
  features: F,
}

pub struct ArcadeEnv<F> {
  inner:    RefCell<ArcadeEnvInner<F>>,
}

impl<F> Default for ArcadeEnv<F> where F: ArcadeFeatures {
  fn default() -> ArcadeEnv<F> {
    ArcadeEnv{
      inner:    RefCell::new(ArcadeEnvInner{
        cfg:        Default::default(),
        cache:      None,
        state:      None,
        features:   Default::default(),
      }),
    }
  }
}

impl<F> Clone for ArcadeEnv<F> where F: ArcadeFeatures {
  fn clone(&self) -> ArcadeEnv<F> {
    let mut inner = self.inner.borrow_mut();
    if let Some(ref cache) = inner.cache {
      let state = cache.borrow_mut().context.save_system_state();
      ArcadeEnv{
        inner:    RefCell::new(ArcadeEnvInner{
          cfg:        inner.cfg.clone(),
          cache:      Some(cache.clone()),
          state:      Some(state),
          features:   inner.features.clone(),
        }),
      }
    } else {
      ArcadeEnv{
        inner:    RefCell::new(ArcadeEnvInner{
          cfg:        inner.cfg.clone(),
          cache:      None,
          state:      None,
          features:   inner.features.clone(),
        }),
      }
    }
  }
}

impl<F> ArcadeEnv<F> where F: ArcadeFeatures {
  pub fn num_minimal_actions(&self) -> usize {
    let mut inner = self.inner.borrow_mut();
    let &mut ArcadeEnvInner{ref cfg, ref cache, ref mut state, ref mut features} = &mut *inner;
    assert!(cache.is_some());
    let cache = cache.as_ref().unwrap();
    let mut cache = cache.borrow_mut();
    cache.context.num_minimal_actions()
  }

  pub fn extract_minimal_action_set(&self, actions: &mut [i32]) -> usize {
    let mut inner = self.inner.borrow_mut();
    let &mut ArcadeEnvInner{ref cfg, ref cache, ref mut state, ref mut features} = &mut *inner;
    assert!(cache.is_some());
    let cache = cache.as_ref().unwrap();
    let mut cache = cache.borrow_mut();
    cache.context.extract_minimal_action_set(actions)
  }
}

impl<F> Env for ArcadeEnv<F> where F: ArcadeFeatures {
  type Init = ArcadeConfig;
  type Action = ArcadeAction;
  type Response = f32;

  fn reset<R>(&self, init: &ArcadeConfig, rng: &mut R) where R: Rng {
    let noop_frames = {
      let mut inner = self.inner.borrow_mut();
      let &mut ArcadeEnvInner{ref mut cache, ref mut cfg, ref mut state, ref mut features} = &mut *inner;
      *cfg = init.clone();
      if cache.is_none() {
        *cache = Some(Rc::new(RefCell::new(CachedArcadeContext{
          context:      ArcadeContext::new(),
          rom_path:     None,
        })));
      }
      assert!(cache.is_some());
      let cache = cache.as_ref().unwrap();
      let mut cache = cache.borrow_mut();
      if cache.rom_path.is_none() || &cfg.rom_path != cache.rom_path.as_ref().unwrap() {
        assert!(cache.context.open_rom(&cfg.rom_path).is_ok());
        cache.rom_path = Some(cfg.rom_path.clone());
      }
      cache.context.reset();
      let saved_state = cache.context.save_system_state();
      *state = Some(saved_state);
      features.reset();
      if cfg.noop_max > 0 {
        rng.gen_range(0, cfg.noop_max + 1)
      } else {
        0
      }
    };
    for _ in 0 .. noop_frames {
      let _ = self.step(&ArcadeAction::noop()).unwrap();
    }
    let mut inner = self.inner.borrow_mut();
    let &mut ArcadeEnvInner{ref cache, ref mut state, ..} = &mut *inner;
    let cache = cache.as_ref().unwrap();
    let mut cache = cache.borrow_mut();
    let saved_state = cache.context.save_system_state();
    *state = Some(saved_state);
  }

  fn is_terminal(&self) -> bool {
    let mut inner = self.inner.borrow_mut();
    let &mut ArcadeEnvInner{ref cfg, ref cache, ref mut state, ref mut features} = &mut *inner;
    assert!(cache.is_some());
    let cache = cache.as_ref().unwrap();
    let mut cache = cache.borrow_mut();
    if let &mut Some(ref mut state) = state {
      cache.context.load_system_state(state);
    } else {
      unreachable!();
    }
    //let is_term = cache.context.is_game_over() || cache.context.num_lives() < 1;
    let is_term = cache.context.is_game_over();
    is_term
  }

  fn is_legal_action(&self, action: &ArcadeAction) -> bool {
    true
  }

  fn step(&self, action: &ArcadeAction) -> Result<Option<f32>, ()> {
    let mut inner = self.inner.borrow_mut();
    let &mut ArcadeEnvInner{ref cfg, ref cache, ref mut state, ref mut features} = &mut *inner;
    assert!(cache.is_some());
    let cache = cache.as_ref().unwrap();
    let mut cache = cache.borrow_mut();
    if let &mut Some(ref mut state) = state {
      cache.context.load_system_state(state);
    } else {
      unreachable!();
    }
    let mut res = 0;
    for _ in 0 .. cfg.skip_frames {
      let r = cache.context.act(action.id);
      res += r;
    }
    // XXX: Clip rewards to positive or negative one.
    /*let mut clip_res = 0;
    if res > 0 {
      clip_res = 1;
    } else if res < 0 {
      clip_res = -1;
    }*/
    *state = Some(cache.context.save_system_state());
    features.update(&mut cache.context);
    Ok(Some(res as f32))
  }
}

impl EnvInputRepr<[f32]> for ArcadeEnv<RamArcadeFeatures> {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (1, 1, 128 * 4)
  }
}

impl SampleExtractInput<[f32]> for ArcadeEnv<RamArcadeFeatures> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let inner = self.inner.borrow();
    for p in 0 .. 128 * 4 {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(128 * 4)
  }
}

impl EnvInputRepr<[f32]> for ArcadeEnv<GrayArcadeFeatures> {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (160, 210, 4)
  }
}

impl SampleExtractInput<[f32]> for ArcadeEnv<GrayArcadeFeatures> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let inner = self.inner.borrow();
    for p in 0 .. 210 * 160 * 4 {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(210 * 160 * 4)
  }
}

impl EnvInputRepr<[f32]> for ArcadeEnv<RgbArcadeFeatures> {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (160, 210, 12)
  }
}

impl SampleExtractInput<[f32]> for ArcadeEnv<RgbArcadeFeatures> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    assert!(210 * 160 * 3 * 4 <= output.len());
    let inner = self.inner.borrow();
    for p in 0 .. 210 * 160 * 12 {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(210 * 160 * 12)
  }
}

/*impl EnvRepr<u8> for ArcadeEnv<RamArcadeFeatures> {
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
}*/
