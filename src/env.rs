use super::{ArcadeContext, ArcadeSavedState};

use genrl::env::{Env, Action, DiscreteAction, EnvRepr};

use rand::{Rng};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::rc::{Rc};

const NUM_ACTIONS:  usize = 18;
const MAX_ACTION:   u32   = 18;

#[derive(Clone)]
pub struct ArcadeConfig {
  pub rom_path: PathBuf,
  pub context:  Option<Rc<RefCell<ArcadeContext>>>,
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

#[derive(Clone, Copy, Debug)]
pub struct ArcadeRamFeatures {
}

#[derive(Clone, Copy, Debug)]
pub struct ArcadeGrayscaleFeatures {
}

#[derive(Clone, Copy, Debug)]
pub struct ArcadeRgbFeatures {
}

pub struct ArcadeEnv<F> {
  cfg:      ArcadeConfig,
  context:  Rc<RefCell<ArcadeContext>>,
  state:    Option<ArcadeSavedState>,
  _marker:  PhantomData<F>,
}

impl<F> Default for ArcadeEnv<F> {
  fn default() -> ArcadeEnv<F> {
    ArcadeEnv{
      cfg:      ArcadeConfig{
        rom_path:   PathBuf::from(""),
        context:    None,
      },
      context:  Rc::new(RefCell::new(ArcadeContext::new())),
      state:    None,
      _marker:  PhantomData,
    }
  }
}

impl<F> Env for ArcadeEnv<F> {
  type Init = ArcadeConfig;
  type Action = ArcadeAction;
  type Response = f32;

  fn reset<R>(&mut self, init: &ArcadeConfig, rng: &mut R) where R: Rng {
    self.cfg = init.clone();
    assert!(self.context.borrow_mut().open_rom(&self.cfg.rom_path).is_ok());
    self.context.borrow_mut().reset();
    self.state = Some(self.context.borrow_mut().save_system_state());
  }

  fn is_terminal(&mut self) -> bool {
    self.context.borrow_mut().is_game_over()
  }

  fn is_legal_action(&mut self, action: &ArcadeAction) -> bool {
    true
  }

  fn step(&mut self, action: &ArcadeAction) -> Result<Option<f32>, ()> {
    if let Some(ref state) = self.state {
      self.context.borrow_mut().load_system_state(state);
    }
    let res = self.context.borrow_mut().act(action.id);
    self.state = Some(self.context.borrow_mut().save_system_state());
    Ok(Some(res as f32))
  }
}

impl EnvRepr<u8> for ArcadeEnv<ArcadeGrayscaleFeatures> {
  fn observable_len(&mut self) -> usize {
    self.context.borrow_mut().screen_width() * self.context.borrow_mut().screen_height()
  }

  fn extract_observable(&mut self, obs_buf: &mut [u8]) {
    self.context.borrow_mut().extract_screen_grayscale(obs_buf);
  }
}

impl EnvRepr<u8> for ArcadeEnv<ArcadeRgbFeatures> {
  fn observable_len(&mut self) -> usize {
    self.context.borrow_mut().screen_width() * self.context.borrow_mut().screen_height() * 3
  }

  fn extract_observable(&mut self, obs_buf: &mut [u8]) {
    self.context.borrow_mut().extract_screen_rgb(obs_buf);
  }
}
