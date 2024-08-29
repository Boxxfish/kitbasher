use bevy::prelude::*;
use configs::ReleaseCfgPlugin;

mod configs;
mod engine;
mod viewer;
mod ui;

/// Main entry point for our game.
fn main() {
    App::new().add_plugins(ReleaseCfgPlugin).run();
}
