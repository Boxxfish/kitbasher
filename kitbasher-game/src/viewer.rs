use bevy::{prelude::*, utils::HashMap};
use bevy_common_assets::ron::RonAssetPlugin;

use crate::engine::{KBEngine, PartData};

pub struct ViewerPlugin;

impl Plugin for ViewerPlugin {
    fn build(&self, app: &mut App) {
        app
        .add_plugins(RonAssetPlugin::<PartData>::new(&["ron"]))
        .add_systems(Startup, setup)
        .add_systems(Update, handle_part_loaded);
    }
}

const BLOCK_PART_NAMES: [&str; 8] = [
    "1x1",
    "2x1_slanted",
    "2x1",
    "2x2_axle",
    "2x2_slanted",
    "2x2",
    "4x1",
    "wheel",
];

#[derive(Resource)]
struct PartModels(pub HashMap<String, Handle<Scene>>);

#[derive(Component)]
struct ProcessPart(pub Handle<PartData>);

#[derive(Resource)]
struct KBEngineWrapper(pub KBEngine);


fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Load part models and data
    let mut part_models = HashMap::new();
    for part_name in BLOCK_PART_NAMES {
        let handle = asset_server.load(format!("models/{}.glb#Scene0", part_name));
        part_models.insert(part_name.into(), handle);
        commands.spawn(ProcessPart(asset_server.load(format!("models/{}.ron", part_name))));
    }
    commands.insert_resource(PartModels(part_models));
    commands.insert_resource(KBEngineWrapper(KBEngine::new(&[], &[[0, 0]])))
}

fn handle_part_loaded(parts: Res<Assets<PartData>>, parts_query: Query<&ProcessPart>, mut engine: ResMut<KBEngineWrapper>) {
    for part_handle in parts_query.iter() {
        if let Some(part) = parts.get(&part_handle.0) {
            engine.0.add_part(part);
        }
    }
}

