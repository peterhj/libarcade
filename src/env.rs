use super::{CachedArcadeContext, ArcadeContext, ArcadeSavedState};

use genrl::env::{Env, Action, DiscreteAction, EnvInputRepr, EnvRepr, Discounted, NormalizeDiscounted};
use operator::prelude::*;

use rand::{Rng};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Mutex};

thread_local! {
  pub static CONTEXT:   RefCell<ArcadeContext> = RefCell::new(ArcadeContext::new());
  pub static ROM_PATH:  RefCell<Option<PathBuf>> = RefCell::new(None);
}

const NUM_ACTIONS:  usize = 18;

#[derive(Clone, Debug)]
pub struct ArcadeConfig {
  pub skip_frames:  usize,
  pub noop_max:     usize,
  pub rom_path:     PathBuf,
}

impl Default for ArcadeConfig {
  fn default() -> ArcadeConfig {
    ArcadeConfig{
      skip_frames:  4,
      noop_max:     0,
      rom_path:     PathBuf::from(""),
    }
  }
}

pub trait GenericArcadeAction: DiscreteAction {
  fn noop() -> Self where Self: Sized;
  fn id(&self) -> i32;
}

#[derive(Clone, Copy)]
pub struct ArcadeAction {
  id:   i32,
}

impl ArcadeAction {
}

impl Action for ArcadeAction {
  fn dim() -> usize {
    NUM_ACTIONS
  }
}

impl DiscreteAction for ArcadeAction {
  fn from_idx(idx: u32) -> ArcadeAction {
    assert!(idx < Self::dim() as u32);
    ArcadeAction{id: idx as i32}
  }

  fn idx(&self) -> u32 {
    self.id as u32
  }
}

impl GenericArcadeAction for ArcadeAction {
  fn noop() -> ArcadeAction {
    ArcadeAction{id: 0}
  }

  fn id(&self) -> i32 {
    self.id
  }
}

#[derive(Clone, Copy)]
pub struct PongArcadeAction {
  idx:  u32,
}

impl PongArcadeAction {
  pub fn noop() -> PongArcadeAction {
    PongArcadeAction{idx: 0}
  }
}

const PONG_ACTION_IDS: [i32; 6] = [0, 1, 3, 4, 11, 12];

impl Action for PongArcadeAction {
  fn dim() -> usize {
    6
  }
}

impl DiscreteAction for PongArcadeAction {
  fn from_idx(idx: u32) -> PongArcadeAction {
    assert!(idx < Self::dim() as u32);
    PongArcadeAction{idx: idx}
  }

  fn idx(&self) -> u32 {
    self.idx
  }
}

impl GenericArcadeAction for PongArcadeAction {
  fn noop() -> PongArcadeAction {
    PongArcadeAction{idx: 0}
  }

  fn id(&self) -> i32 {
    PONG_ACTION_IDS[self.idx as usize]
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

pub struct ArcadeEnvInner<A, F> {
  cfg:      ArcadeConfig,
  state:    Option<ArcadeSavedState>,
  features: F,
  _marker:  PhantomData<A>,
}

pub struct ArcadeEnv<A, F> {
  inner:    RefCell<ArcadeEnvInner<A, F>>,
}

impl<A, F> Default for ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  fn default() -> ArcadeEnv<A, F> {
    ArcadeEnv{
      inner:    RefCell::new(ArcadeEnvInner{
        cfg:        Default::default(),
        state:      None,
        features:   Default::default(),
        _marker:    PhantomData,
      }),
    }
  }
}

impl<A, F> Clone for ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  fn clone(&self) -> ArcadeEnv<A, F> {
    let mut inner = self.inner.borrow_mut();
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      let state = ctx.save_system_state();
      ArcadeEnv{
        inner:  RefCell::new(ArcadeEnvInner{
          cfg:      inner.cfg.clone(),
          state:    Some(state),
          features: inner.features.clone(),
          _marker:  PhantomData,
        }),
      }
    })
  }
}

impl<A, F> ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  pub fn num_minimal_actions(&self) -> usize {
    let mut inner = self.inner.borrow_mut();
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      ctx.num_minimal_actions()
    })
  }

  pub fn extract_minimal_action_set(&self, actions: &mut [i32]) -> usize {
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      ctx.extract_minimal_action_set(actions)
    })
  }
}

impl<A, F> Env for ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  type Init = ArcadeConfig;
  type Action = A;
  type Response = f32;

  fn reset<R>(&self, init: &ArcadeConfig, rng: &mut R) where R: Rng {
    let noop_frames = {
      let mut inner = self.inner.borrow_mut();
      let &mut ArcadeEnvInner{ref mut cfg, ref mut state, ref mut features, ..} = &mut *inner;
      *cfg = init.clone();
      CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ROM_PATH.with(|rom_path| {
          let mut rom_path = rom_path.borrow_mut();
      //if cache.rom_path.is_none() || &cfg.rom_path != cache.rom_path.as_ref().unwrap() {
          if rom_path.is_none() || &cfg.rom_path != rom_path.as_ref().unwrap() {
            assert!(ctx.open_rom(&cfg.rom_path).is_ok());
            *rom_path = Some(cfg.rom_path.clone());
          }
        });
        ctx.reset();
        let saved_state = ctx.save_system_state();
        *state = Some(saved_state);
        features.reset();
        features.update(&mut ctx);
      });
      if cfg.noop_max > 0 {
        rng.gen_range(0, cfg.noop_max + 1)
      } else {
        0
      }
    };
    for _ in 0 .. noop_frames {
      let _ = self.step(&A::noop()).unwrap();
    }
  }

  fn is_terminal(&self) -> bool {
    let mut inner = self.inner.borrow_mut();
    let &mut ArcadeEnvInner{ref cfg, ref mut state, ref mut features, ..} = &mut *inner;
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      if let &mut Some(ref mut state) = state {
        ctx.load_system_state(state);
      } else {
        unreachable!();
      }
      let is_term = ctx.is_game_over();
      is_term
    })
  }

  fn is_legal_action(&self, action: &A) -> bool {
    true
  }

  fn step(&self, action: &A) -> Result<Option<f32>, ()> {
    let mut inner = self.inner.borrow_mut();
    let &mut ArcadeEnvInner{ref cfg, ref mut state, ref mut features, ..} = &mut *inner;
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      if let &mut Some(ref mut state) = state {
        ctx.load_system_state(state);
      } else {
        unreachable!();
      }
      let mut res = 0;
      for _ in 0 .. cfg.skip_frames {
        let r = ctx.act(action.id());
        res += r;
      }
      *state = Some(ctx.save_system_state());
      features.update(&mut ctx);
      Ok(Some(res as f32))
    })
  }
}

impl<A> EnvInputRepr<[f32]> for ArcadeEnv<A, RamArcadeFeatures> where A: GenericArcadeAction {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (1, 1, 128 * 4)
  }
}

impl<A> SampleExtractInput<[f32]> for ArcadeEnv<A, RamArcadeFeatures> where A: GenericArcadeAction {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let inner = self.inner.borrow();
    for p in 0 .. 128 * 4 {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(128 * 4)
  }
}

impl<A> EnvInputRepr<[f32]> for ArcadeEnv<A, GrayArcadeFeatures> where A: GenericArcadeAction {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (160, 210, 4)
  }
}

impl<A> SampleExtractInput<[f32]> for ArcadeEnv<A, GrayArcadeFeatures> where A: GenericArcadeAction {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let inner = self.inner.borrow();
    for p in 0 .. 210 * 160 * 4 {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(210 * 160 * 4)
  }
}

impl<A> EnvInputRepr<[f32]> for ArcadeEnv<A, RgbArcadeFeatures> where A: GenericArcadeAction {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (160, 210, 12)
  }
}

impl<A> SampleExtractInput<[f32]> for ArcadeEnv<A, RgbArcadeFeatures> where A: GenericArcadeAction {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    assert!(210 * 160 * 3 * 4 <= output.len());
    let inner = self.inner.borrow();
    for p in 0 .. 210 * 160 * 12 {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(210 * 160 * 12)
  }
}

pub struct SharedArcadeEnv<A, F> {
  inner:    Mutex<ArcadeEnvInner<A, F>>,
}

impl<A, F> Default for SharedArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  fn default() -> SharedArcadeEnv<A, F> {
    SharedArcadeEnv{
      inner:    Mutex::new(ArcadeEnvInner{
        cfg:        Default::default(),
        state:      None,
        features:   Default::default(),
        _marker:    PhantomData,
      }),
    }
  }
}

impl<A, F> Clone for SharedArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  fn clone(&self) -> SharedArcadeEnv<A, F> {
    let mut inner = self.inner.lock().unwrap();
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      let state = ctx.save_system_state();
      SharedArcadeEnv{
        inner:  Mutex::new(ArcadeEnvInner{
          cfg:      inner.cfg.clone(),
          state:    Some(state),
          features: inner.features.clone(),
          _marker:  PhantomData,
        }),
      }
    })
  }
}

impl<A, F> SharedArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  pub fn num_minimal_actions(&self) -> usize {
    let mut inner = self.inner.lock().unwrap();
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      ctx.num_minimal_actions()
    })
  }

  pub fn extract_minimal_action_set(&self, actions: &mut [i32]) -> usize {
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      ctx.extract_minimal_action_set(actions)
    })
  }
}

impl<A, F> Env for SharedArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  type Init = ArcadeConfig;
  type Action = A;
  type Response = f32;

  fn reset<R>(&self, init: &ArcadeConfig, rng: &mut R) where R: Rng {
    let noop_frames = {
      let mut inner = self.inner.lock().unwrap();
      let &mut ArcadeEnvInner{ref mut cfg, ref mut state, ref mut features, ..} = &mut *inner;
      *cfg = init.clone();
      CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ROM_PATH.with(|rom_path| {
          let mut rom_path = rom_path.borrow_mut();
      //if cache.rom_path.is_none() || &cfg.rom_path != cache.rom_path.as_ref().unwrap() {
          if rom_path.is_none() || &cfg.rom_path != rom_path.as_ref().unwrap() {
            assert!(ctx.open_rom(&cfg.rom_path).is_ok());
            *rom_path = Some(cfg.rom_path.clone());
          }
        });
        ctx.reset();
        let saved_state = ctx.save_system_state();
        *state = Some(saved_state);
        features.reset();
        features.update(&mut ctx);
      });
      if cfg.noop_max > 0 {
        rng.gen_range(0, cfg.noop_max + 1)
      } else {
        0
      }
    };
    for _ in 0 .. noop_frames {
      let _ = self.step(&A::noop()).unwrap();
    }
  }

  fn is_terminal(&self) -> bool {
    let mut inner = self.inner.lock().unwrap();
    let &mut ArcadeEnvInner{ref cfg, ref mut state, ref mut features, ..} = &mut *inner;
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      if let &mut Some(ref mut state) = state {
        ctx.load_system_state(state);
      } else {
        unreachable!();
      }
      let is_term = ctx.is_game_over();
      is_term
    })
  }

  fn is_legal_action(&self, action: &A) -> bool {
    true
  }

  fn step(&self, action: &A) -> Result<Option<f32>, ()> {
    let mut inner = self.inner.lock().unwrap();
    let &mut ArcadeEnvInner{ref cfg, ref mut state, ref mut features, ..} = &mut *inner;
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      if let &mut Some(ref mut state) = state {
        ctx.load_system_state(state);
      } else {
        unreachable!();
      }
      let mut res = 0;
      for _ in 0 .. cfg.skip_frames {
        let r = ctx.act(action.id());
        res += r;
      }
      *state = Some(ctx.save_system_state());
      features.update(&mut ctx);
      Ok(Some(res as f32))
    })
  }
}

impl<A> EnvInputRepr<[f32]> for SharedArcadeEnv<A, GrayArcadeFeatures> where A: GenericArcadeAction {
  fn _shape3d(&self) -> (usize, usize, usize) {
    (160, 210, 4)
  }
}

impl<A> SampleExtractInput<[f32]> for SharedArcadeEnv<A, GrayArcadeFeatures> where A: GenericArcadeAction {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let inner = self.inner.lock().unwrap();
    for p in 0 .. 210 * 160 * 4 {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(210 * 160 * 4)
  }
}
