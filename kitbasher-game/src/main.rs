use bevy::prelude::*;
use configs::ReleaseCfgPlugin;

mod cartpole;
mod configs;
mod engine;

/// Main entry point for our game.
fn main() {
    App::new().add_plugins(ReleaseCfgPlugin).run();
}
