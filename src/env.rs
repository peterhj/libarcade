use super::{CachedArcadeContext, ArcadeContext, ArcadeSavedState};

use genrl::env::{Env, Action, DiscreteAction, EnvInputRepr, EnvRepr, Discounted, NormalizeDiscounted};
use genrl::features::{EnvObsRepr};
use operator::prelude::*;
use rng::xorshift::*;
use stb_image::image::{Image};

use rand::{Rng};
use std::cell::{RefCell};
use std::cmp::{max};
use std::collections::{VecDeque};
use std::fs::{File};
use std::io::{Write};
use std::marker::{PhantomData};
use std::mem::{transmute};
use std::ops::{Deref};
use std::path::{Path, PathBuf};
use std::rc::{Rc};
use std::sync::{Mutex};

thread_local! {
  pub static CONTEXT:   RefCell<ArcadeContext> = RefCell::new(ArcadeContext::new());
  pub static ROM_PATH:  RefCell<Option<PathBuf>> = RefCell::new(None);
}

const NUM_ACTIONS:  usize = 18;

#[derive(Clone, Debug)]
pub struct ArcadeConfig {
  pub skip_frames:      usize,
  pub history_len:      usize,
  pub crop_screen:      Option<(usize, usize, isize, isize)>,
  pub resize_screen:    Option<(usize, usize)>,
  pub rom_path:         PathBuf,
}

impl Default for ArcadeConfig {
  fn default() -> ArcadeConfig {
    ArcadeConfig{
      history_len:      4,
      skip_frames:      4,
      crop_screen:      None,
      resize_screen:    None,
      rom_path:         PathBuf::from(""),
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

const FOUR_ACTION_IDS: [i32; 4] = [0, 1, 3, 4];

#[derive(Clone, Copy)]
pub struct FourArcadeAction {
  idx:  u32,
}

impl FourArcadeAction {
  pub fn noop() -> FourArcadeAction {
    FourArcadeAction{idx: 0}
  }
}

impl Action for FourArcadeAction {
  fn dim() -> usize {
    6
  }
}

impl DiscreteAction for FourArcadeAction {
  fn from_idx(idx: u32) -> FourArcadeAction {
    assert!(idx < Self::dim() as u32);
    FourArcadeAction{idx: idx}
  }

  fn idx(&self) -> u32 {
    self.idx
  }
}

impl GenericArcadeAction for FourArcadeAction {
  fn noop() -> FourArcadeAction {
    FourArcadeAction{idx: 0}
  }

  fn id(&self) -> i32 {
    FOUR_ACTION_IDS[self.idx as usize]
  }
}

pub type BreakoutArcadeAction = FourArcadeAction;

const SIX_ACTION_IDS: [i32; 6] = [0, 1, 3, 4, 11, 12];

#[derive(Clone, Copy)]
pub struct SixArcadeAction {
  idx:  u32,
}

impl SixArcadeAction {
  pub fn noop() -> SixArcadeAction {
    SixArcadeAction{idx: 0}
  }
}

impl Action for SixArcadeAction {
  fn dim() -> usize {
    6
  }
}

impl DiscreteAction for SixArcadeAction {
  fn from_idx(idx: u32) -> SixArcadeAction {
    assert!(idx < Self::dim() as u32);
    SixArcadeAction{idx: idx}
  }

  fn idx(&self) -> u32 {
    self.idx
  }
}

impl GenericArcadeAction for SixArcadeAction {
  fn noop() -> SixArcadeAction {
    SixArcadeAction{idx: 0}
  }

  fn id(&self) -> i32 {
    SIX_ACTION_IDS[self.idx as usize]
  }
}

pub type PongArcadeAction = SixArcadeAction;
pub type SpaceInvadersArcadeAction = SixArcadeAction;

pub trait ArcadeFeatures: Clone + Default {
  fn reset(&mut self);
  fn resize(&mut self, history_len: usize);
  fn update(&mut self, context: &mut ArcadeContext);
  fn obs(&self) -> &[u8];
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

  fn resize(&mut self, history_len: usize) {
    if self.window == history_len {
      return;
    }
    self.window = history_len;
    self.obs_buf.resize(128 * history_len, 0);
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

  fn obs(&self) -> &[u8] {
    &self.obs_buf
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

  fn resize(&mut self, history_len: usize) {
    if self.window == history_len {
      return;
    }
    self.window = history_len;
    self.obs_buf.resize(160 * 210 * history_len, 0);
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

  fn obs(&self) -> &[u8] {
    &self.obs_buf
  }
}

#[derive(Clone)]
pub struct RevGrayArcadeFeatures {
  width:    usize,
  height:   usize,
  window:   usize,
  obs_buf:  Vec<u8>,
}

impl Default for RevGrayArcadeFeatures {
  fn default() -> RevGrayArcadeFeatures {
    let mut obs_buf = Vec::with_capacity(210 * 160 * 4);
    for _ in 0 .. 210 * 160 * 4 {
      obs_buf.push(0);
    }
    RevGrayArcadeFeatures{
      width:    160,
      height:   210,
      window:   4,
      obs_buf:  obs_buf,
    }
  }
}

impl ArcadeFeatures for RevGrayArcadeFeatures {
  fn reset(&mut self) {
    for p in 0 .. self.width * self.height * self.window {
      self.obs_buf[p] = 0;
    }
  }

  fn resize(&mut self, history_len: usize) {
    if self.window == history_len {
      return;
    }
    self.window = history_len;
    self.obs_buf.resize(160 * 210 * history_len, 0);
  }

  fn update(&mut self, context: &mut ArcadeContext) {
    let frame_sz = context.screen_size();
    assert_eq!(self.width * self.height, frame_sz);
    for frame in 0 .. self.window-1 {
      let (pre_buf, post_buf) = self.obs_buf.split_at_mut((frame+1) * frame_sz);
      //post_buf[ .. frame_sz].copy_from_slice(&pre_buf[frame * frame_sz .. ]);
      pre_buf[frame * frame_sz .. (frame+1) * frame_sz].copy_from_slice(&post_buf[ .. frame_sz]);
    }
    //context.extract_screen_grayscale(&mut self.obs_buf[ .. frame_sz]);
    context.extract_screen_grayscale(&mut self.obs_buf[(self.window-1) * frame_sz .. ]);
  }

  fn obs(&self) -> &[u8] {
    &self.obs_buf
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

  fn resize(&mut self, history_len: usize) {
    if self.window == history_len {
      return;
    }
    self.window = history_len;
    self.obs_buf.resize(3 * 160 * 210 * history_len, 0);
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

  fn obs(&self) -> &[u8] {
    &self.obs_buf
  }
}

/*pub struct MinifiedArcadeEnvInner<A> {
  cfg:      ArcadeConfig,
  history:  VecDeque<(ArcadeSavedState, A)>,
  state:    Option<ArcadeSavedState>,
  _marker:  PhantomData<A>,
}

pub struct MinifiedArcadeEnv<A> {
  inner:    RefCell<MinifiedArcadeEnvInner<A>>,
}*/

pub struct ArcadeEnvInner<A, F> {
  cfg:      ArcadeConfig,
  ctx:      ArcadeContext,
  rom_path: Option<PathBuf>,
  //history:  VecDeque<(ArcadeSavedState, A)>,
  //state:    Option<ArcadeSavedState>,
  features: F,
  lifelost: bool,
  _marker:  PhantomData<A>,
}

impl<A, F> ArcadeEnvInner<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  /*pub fn deminify(mini_env: &MinifiedArcadeEnv<A>) -> ArcadeEnv<A, F> {
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      let mut mini_env = mini_env.inner.borrow_mut();
      let mut features = F::default();
      features.resize(mini_env.cfg.history_len);
      features.reset();
      let mut new_history = VecDeque::with_capacity(4);
      for h in mini_env.history.iter_mut() {
        ctx.load_system_state(&mut h.0);
        features.update(&mut ctx);
        new_history.push_back((ctx.save_system_state(), h.1));
      }
      let new_state = if let Some(ref mut state) = mini_env.state {
        ctx.load_system_state(state);
        features.update(&mut ctx);
        Some(ctx.save_system_state())
      } else {
        None
      };
      ArcadeEnv{
        inner:  RefCell::new(ArcadeEnvInner{
          cfg:      mini_env.cfg.clone(),
          history:  new_history,
          state:    new_state,
          features: features,
          _marker:  PhantomData,
        }),
      }
    })
  }

  pub fn minify(&mut self) -> MinifiedArcadeEnv<A> {
    CONTEXT.with(|ctx| {
      let mut ctx = ctx.borrow_mut();
      let mut new_history = VecDeque::with_capacity(4);
      for h in self.history.iter_mut() {
        ctx.load_system_state(&mut h.0);
        new_history.push_back((ctx.save_system_state(), h.1));
      }
      let new_state = if let Some(ref mut state) = self.state {
        ctx.load_system_state(state);
        Some(ctx.save_system_state())
      } else {
        None
      };
      MinifiedArcadeEnv{
        inner: RefCell::new(MinifiedArcadeEnvInner{
          cfg:      self.cfg.clone(),
          history:  new_history,
          state:    new_state,
          _marker:  PhantomData,
        }),
      }
    })
  }*/

  pub fn reset<R>(&mut self, init: &ArcadeConfig, rng: &mut R) where R: Rng {
  //pub fn reset(&mut self, init: &ArcadeConfig, rng: &mut Xorshiftplus128Rng) {
    //println!("DEBUG: restarting...");
    //let &mut ArcadeEnvInner{ref mut cfg, ref mut state, ref mut history, ref mut features, ..} = self;
    self.cfg = init.clone();
    if self.rom_path.is_none() || &self.cfg.rom_path != self.rom_path.as_ref().unwrap() {
      assert!(self.ctx.open_rom(&self.cfg.rom_path).is_ok());
      self.rom_path = Some(self.cfg.rom_path.clone());
    }
    if !self.lifelost || self.ctx.is_game_over() {
      self.ctx.reset();
    }
      //let saved_state = self.ctx.save_system_state();
      //self.state = Some(saved_state);
      //self.history.clear();
    self.features.reset();
    self.features.update(&mut self.ctx);
    self.lifelost = false;
    //});
    /*let noop_frames = {
      // XXX(20161028): We want all frames in the input to have something
      // rather than have any zero padding. Also, add one to the history length
      // in case of color averaging.
      /*if cfg.noop_max > cfg.history_len + 1 {
        rng.gen_range(cfg.history_len + 1, cfg.noop_max + 1)
      } else {
        cfg.history_len + 1
      }*/
      let mut xrng: &mut Xorshiftplus128Rng = unsafe { transmute(rng) };
      xrng._randint(cfg.history_len + 1, max(cfg.history_len + 1, cfg.noop_max))
      //rng.gen_range(cfg.history_len + 1, max(cfg.history_len + 1, cfg.noop_max) + 1)
    };
    for _ in 0 .. noop_frames {
      let _ = self.step(&A::noop()).unwrap();
    }*/
  }

  pub fn is_terminal(&mut self) -> bool {
    //CONTEXT.with(|ctx| {
      //let &mut ArcadeEnvInner{ref cfg, ref mut state, ref mut features, ..} = self;
      //let mut ctx = ctx.borrow_mut();
      /*if let &mut Some(ref mut state) = state {
        ctx.load_system_state(state);
      } else {
        unreachable!();
      }*/
      self.ctx.is_game_over() || self.lifelost
    //})
  }

  pub fn step(&mut self, action: &A) -> Result<Option<f32>, ()> {
    //CONTEXT.with(|ctx| {
      //let &mut ArcadeEnvInner{ref cfg, ref mut state, ref mut history, ref mut features, ..} = self;
      //let mut ctx = ctx.borrow_mut();
      /*if let &mut Some(ref mut state) = state {
        ctx.load_system_state(state);
      } else {
        unreachable!();
      }*/
      /*assert!(history.len() <= self.cfg.history_len);
      if history.len() == self.cfg.history_len {
        let _ = history.pop_front();
      }
      history.push_back((ctx.save_system_state(), *action));*/
      // FIXME(20161027): Move frame skipping into the "ale.cfg" file.
      /*let mut res = 0;
      for _ in 0 .. cfg.skip_frames {
        let r = ctx.act(action.id());
        res += r;
      }*/
      let prev_lives = self.ctx.num_lives();
      let res = self.ctx.act(action.id());
      let next_lives = self.ctx.num_lives();
      if next_lives < prev_lives {
        self.lifelost = true;
      }
      //self.state = Some(self.ctx.save_system_state());
      self.features.update(&mut self.ctx);
      Ok(Some(res as f32))
    //})
  }

  pub fn save_png(&mut self, path: &Path) {
    let frame_sz = 160 * 210;
    let mut pixels = Vec::with_capacity(3 * frame_sz);
    for &x in &self.features.obs()[ .. frame_sz] {
      let y = x as i32;
      assert!(y >= 0 && y <= 255);
      pixels.push(y as u8);
      pixels.push(y as u8);
      pixels.push(y as u8);
    }
    let im = Image::new(160, 210, 3, pixels);
    let png_buf = match im.write_png() {
      Err(_) => panic!("failed to generate png"),
      Ok(png) => png,
    };
    let mut f = File::create(path).unwrap();
    f.write_all(&png_buf).unwrap();
  }
}

pub struct ArcadeEnv<A, F> {
  inner:    RefCell<ArcadeEnvInner<A, F>>,
}

impl<A, F> Default for ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  fn default() -> ArcadeEnv<A, F> {
    ArcadeEnv{
      inner:    RefCell::new(ArcadeEnvInner{
        cfg:        Default::default(),
        ctx:        ArcadeContext::new(),
        rom_path:   None,
        //history:    VecDeque::with_capacity(4),
        //state:      None,
        features:   Default::default(),
        lifelost:   false,
        _marker:    PhantomData,
      }),
    }
  }
}

/*impl<A, F> Clone for ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  fn clone(&self) -> ArcadeEnv<A, F> {
    let mut inner = self.inner.borrow_mut();
    //CONTEXT.with(|ctx| {
      //let mut ctx = ctx.borrow_mut();
      /*let mut new_history = VecDeque::with_capacity(4);
      for h in inner.history.iter_mut() {
        ctx.load_system_state(&mut h.0);
        new_history.push_back((ctx.save_system_state(), h.1));
      }*/
      /*let new_state = if let Some(ref mut state) = inner.state {
        ctx.load_system_state(state);
        Some(ctx.save_system_state())
      } else {
        None
      };*/
      ArcadeEnv{
        inner:  RefCell::new(ArcadeEnvInner{
          cfg:      inner.cfg.clone(),
          ctx:      ArcadeContext::new(),
          rom_path: inner.rom_path.clone(),
          //history:  new_history,
          //state:    new_state,
          features: inner.features.clone(),
          _marker:  PhantomData,
        }),
      }
    //})
  }
}*/

impl<A, F> ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  pub fn _state_size(&self) -> usize {
    let mut inner = self.inner.borrow_mut();
    //inner.state.as_mut().unwrap().encoded_size()
    unimplemented!();
  }

  pub fn num_minimal_actions(&self) -> usize {
    let mut inner = self.inner.borrow_mut();
    //CONTEXT.with(|ctx| {
      //let mut ctx = ctx.borrow_mut();
      inner.ctx.num_minimal_actions()
    //})
  }

  pub fn extract_minimal_action_set(&self, actions: &mut [i32]) -> usize {
    let mut inner = self.inner.borrow_mut();
    //CONTEXT.with(|ctx| {
      //let mut ctx = ctx.borrow_mut();
      inner.ctx.extract_minimal_action_set(actions)
    //})
  }
}

impl<A, F> Env for ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  type Init = ArcadeConfig;
  type Action = A;
  type Response = f32;

  fn reset<R>(&self, init: &ArcadeConfig, rng: &mut R) where R: Rng {
  //fn reset(&self, init: &ArcadeConfig, rng: &mut Xorshiftplus128Rng) {
    let mut inner = self.inner.borrow_mut();
    inner.reset(init, rng);
  }

  fn is_terminal(&self) -> bool {
    let mut inner = self.inner.borrow_mut();
    inner.is_terminal()
  }

  fn is_legal_action(&self, action: &A) -> bool {
    true
  }

  fn step(&self, action: &A) -> Result<Option<f32>, ()> {
    let mut inner = self.inner.borrow_mut();
    inner.step(action)
  }

  fn _save_png(&self, path: &Path) {
    let mut inner = self.inner.borrow_mut();
    inner.save_png(path);
  }
}

#[derive(Clone)]
pub struct ArcadeGrayObs {
  pub dim:  (usize, usize, usize),
  pub buf:  Vec<u8>,
}

impl Deref for ArcadeGrayObs {
  type Target = [u8];

  fn deref(&self) -> &[u8] {
    &self.buf
  }
}

impl<A, F> EnvObsRepr<ArcadeGrayObs> for ArcadeEnv<A, F> where A: GenericArcadeAction, F: ArcadeFeatures {
  fn _obs_shape3d() -> (usize, usize, usize) {
    (160, 160, 1)
    //(160, 210, 1)
  }

  fn observe(&self) -> ArcadeGrayObs {
    let mut inner = self.inner.borrow_mut();
    //CONTEXT.with(|ctx| {
      //let mut ctx = ctx.borrow_mut();
      let mut buf = Vec::with_capacity(160 * 210);
      buf.resize(160 * 210, 0);
      inner.ctx.extract_screen_grayscale(&mut buf);
      if inner.cfg.crop_screen.is_none() && inner.cfg.resize_screen.is_none() {
        ArcadeGrayObs{
          dim:    (160, 210, 1),
          buf:    buf,
        }
      } else if inner.cfg.crop_screen.is_some() && inner.cfg.resize_screen.is_none() {
        let (crop_w, crop_h, offset_x, offset_y) = inner.cfg.crop_screen.unwrap();
        let mut cropped_buf = Vec::with_capacity(crop_w * crop_h);
        for v in 0 .. crop_h {
          for u in 0 .. crop_w {
            let x = (u as isize + offset_x) as usize;
            let y = (v as isize + offset_y) as usize;
            if x < 160 && y < 210 {
              cropped_buf.push(buf[x + 160 * y]);
            } else {
              cropped_buf.push(0);
            }
          }
        }
        assert_eq!(crop_w * crop_h, cropped_buf.len());
        ArcadeGrayObs{
          dim:    (crop_w, crop_h, 1),
          buf:    cropped_buf,
        }
      } else if inner.cfg.crop_screen.is_none() && inner.cfg.resize_screen.is_some() {
        unimplemented!();
      } else if inner.cfg.crop_screen.is_some() && inner.cfg.resize_screen.is_some() {
        unimplemented!();
      } else {
        unreachable!();
      }
    //})
  }
}

impl SampleExtractInput<[f32]> for ArcadeGrayObs {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    assert!(160 * 160 <= output.len());
    for p in 0 .. 160 * 160 {
      output[p] = self.buf[p] as f32;
    }
    Ok(160 * 160)
    /*assert!(160 * 210 <= output.len());
    for p in 0 .. 160 * 210 {
      output[p] = self.buf[p] as f32;
    }
    Ok(160 * 210)*/
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
    let inner = self.inner.borrow();
    (inner.features.width, inner.features.height, inner.features.window)
  }
}

impl<A> SampleExtractInput<[f32]> for ArcadeEnv<A, GrayArcadeFeatures> where A: GenericArcadeAction {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let inner = self.inner.borrow();
    let frame_sz = inner.features.width * inner.features.height * inner.features.window;
    for p in 0 .. frame_sz {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(frame_sz)
  }
}

impl<A> EnvInputRepr<[f32]> for ArcadeEnv<A, RevGrayArcadeFeatures> where A: GenericArcadeAction {
  fn _shape3d(&self) -> (usize, usize, usize) {
    let inner = self.inner.borrow();
    (inner.features.width, inner.features.height, inner.features.window)
  }
}

impl<A> SampleExtractInput<[f32]> for ArcadeEnv<A, RevGrayArcadeFeatures> where A: GenericArcadeAction {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let inner = self.inner.borrow();
    let frame_sz = inner.features.width * inner.features.height * inner.features.window;
    for p in 0 .. frame_sz {
      output[p] = inner.features.obs_buf[p] as f32;
    }
    Ok(frame_sz)
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
