use bevy::{color::palettes::css, prelude::*, utils::HashMap};
use bevy_common_assets::ron::RonAssetPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use rand::Rng;

use crate::{
    engine::{KBEngine, PartData, PlacedConfig},
    ui::menu_button::{MenuButtonBundle, MenuButtonPressedEvent},
};

pub struct ViewerPlugin;

impl Plugin for ViewerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            RonAssetPlugin::<PartData>::new(&["ron"]),
            PanOrbitCameraPlugin,
        ))
        .add_systems(Startup, (setup_parts, setup_ui))
        .add_systems(
            Update,
            (
                handle_part_loaded,
                handle_buttons,
                update_model,
                render_placed,
                move_camera,
            ),
        );
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

#[derive(Component)]
struct ModelRoot;

fn setup_parts(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Load part models and data
    let mut part_models = HashMap::new();
    for part_name in BLOCK_PART_NAMES {
        let handle = asset_server.load(format!("models/{}.glb#Scene0", part_name));
        part_models.insert(part_name.into(), handle);
        commands.spawn(ProcessPart(
            asset_server.load(format!("models/{}.ron", part_name)),
        ));
    }
    commands.insert_resource(PartModels(part_models));
    commands.insert_resource(KBEngineWrapper(KBEngine::new(&[], &[[0, 0]])));
    commands.spawn((ModelRoot, SpatialBundle::default()));
}

fn handle_part_loaded(
    parts: Res<Assets<PartData>>,
    parts_query: Query<(Entity, &ProcessPart)>,
    mut engine: ResMut<KBEngineWrapper>,
    mut commands: Commands,
) {
    for (e, part_handle) in parts_query.iter() {
        if let Some(part) = parts.get(&part_handle.0) {
            engine.0.add_part(part);
            commands.entity(e).despawn();
        }
    }
}

#[derive(Component)]
enum UIAction {
    Step,
    Generate,
    Reset,
}

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            Camera3dBundle { ..default() },
            PanOrbitCamera {
                radius: Some(100.),
                ..default()
            },
        ))
        .with_children(|p| {
            p.spawn(DirectionalLightBundle {
                directional_light: DirectionalLight {
                    color: Color::WHITE,
                    illuminance: 1000.,
                    ..default()
                },
                ..default()
            });
        });
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                display: Display::Flex,
                ..default()
            },
            ..default()
        })
        .with_children(|p| {
            p.spawn(NodeBundle {
                style: Style {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(16.)),
                    row_gap: Val::Px(8.),
                    ..default()
                },
                background_color: Color::BLACK.with_alpha(0.5).into(),
                ..default()
            })
            .with_children(|p| {
                p.spawn((UIAction::Step, MenuButtonBundle::from_label("Step")));
                p.spawn((UIAction::Generate, MenuButtonBundle::from_label("Generate")));
                p.spawn((UIAction::Reset, MenuButtonBundle::from_label("Reset")));
            });
        });
}

fn handle_buttons(
    mut ev_menu_pressed: EventReader<MenuButtonPressedEvent>,
    action_query: Query<&UIAction>,
    mut engine: ResMut<KBEngineWrapper>,
) {
    for ev in ev_menu_pressed.read() {
        match action_query.get(ev.sender).unwrap() {
            UIAction::Step => {
                let engine = &mut engine.0;
                if engine.get_model().is_empty() {
                    let part = engine.get_part(0);
                    let placed = PlacedConfig {
                        position: Vec3::ZERO,
                        part_id: 0,
                        rotation: Quat::IDENTITY,
                        connectors: part.connectors.clone(),
                        bboxes: part.bboxes.clone(),
                        connections: vec![None; part.connectors.len()],
                    };
                    engine.place_part(&placed);
                } else {
                    let candidates = engine.gen_candidates();
                    let mut rng = rand::thread_rng();
                    let placed = &candidates[rng.gen_range(0..candidates.len())];
                    engine.place_part(placed);
                }
            }
            UIAction::Generate => {
                let engine = &mut engine.0;

                let part = engine.get_part(0);
                let placed = PlacedConfig {
                    position: Vec3::ZERO,
                    part_id: 0,
                    rotation: Quat::IDENTITY,
                    connectors: part.connectors.clone(),
                    bboxes: part.bboxes.clone(),
                    connections: vec![None; part.connectors.len()],
                };
                engine.place_part(&placed);

                for _ in 0..64 {
                    let candidates = engine.gen_candidates();
                    let mut rng = rand::thread_rng();
                    let placed = &candidates[rng.gen_range(0..candidates.len())];
                    engine.place_part(placed);
                }
            }
            UIAction::Reset => {
                let engine = &mut engine.0;
                engine.clear_model();
            }
        }
    }
}

#[derive(Component)]
struct PlacedWrapper(pub PlacedConfig);

fn update_model(
    mut commands: Commands,
    engine: Res<KBEngineWrapper>,
    model_query: Query<Entity, With<ModelRoot>>,
    part_models: Res<PartModels>,
) {
    if engine.is_changed() {
        if let Ok(root) = model_query.get_single() {
            commands.entity(root).despawn_recursive();
            commands
                .spawn((ModelRoot, SpatialBundle::default()))
                .with_children(|p| {
                    for placed in engine.0.get_model() {
                        let part = engine.0.get_part(placed.part_id);
                        let part_name = part
                            .model_path
                            .split('/')
                            .last()
                            .unwrap()
                            .split('.')
                            .next()
                            .unwrap();
                        p.spawn((
                            PlacedWrapper(placed.clone()),
                            SceneBundle {
                                scene: part_models.0.get(part_name).unwrap().clone(),
                                transform: Transform::from_rotation(placed.rotation)
                                    .with_translation(placed.position),
                                ..default()
                            },
                        ));
                    }
                });
        }
    }
}

fn render_placed(placed_query: Query<&PlacedWrapper>, mut gizmo: Gizmos) {
    for placed in placed_query.iter() {
        // Bounding boxes
        for bbox in &placed.0.bboxes {
            gizmo.cuboid(
                Transform {
                    translation: placed.0.position + bbox.center,
                    rotation: Quat::IDENTITY,
                    scale: bbox.half_sizes * 2.,
                },
                css::GREEN,
            );
        }
        // Connectors
        for connector in &placed.0.connectors {
            gizmo.sphere(
                placed.0.position + connector.position,
                Quat::IDENTITY,
                1.,
                if connector.side_a {
                    css::YELLOW
                } else {
                    css::PURPLE
                },
            );
        }
    }
}

fn move_camera(
    input: Res<ButtonInput<KeyCode>>,
    mut orbit_query: Query<&mut PanOrbitCamera>,
    time: Res<Time>,
) {
    if let Ok(mut orbit) = orbit_query.get_single_mut() {
        let speed = 10. * time.delta_seconds();
        if input.pressed(KeyCode::ArrowLeft) {
            orbit.target_yaw += speed;
        }
        if input.pressed(KeyCode::ArrowRight) {
            orbit.target_yaw -= speed;
        }
        if input.pressed(KeyCode::ArrowUp) {
            orbit.target_pitch -= speed;
        }
        if input.pressed(KeyCode::ArrowDown) {
            orbit.target_pitch += speed;
        }
    }
}
