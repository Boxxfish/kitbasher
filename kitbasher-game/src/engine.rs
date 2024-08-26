use bevy::math::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// An instance of the kitbasher engine.
pub struct KBEngine {
    parts: Vec<PartData>,
    connect_rules: Vec<[usize; 2]>,
    model: Vec<PlacedConfig>,
}

impl KBEngine {
    /// Creates a new instance of the engine.
    pub fn new(parts: &[PartData], connect_rules: &[[usize; 2]]) -> Self {
        Self {
            parts: parts.into(),
            connect_rules: connect_rules.into(),
            model: Vec::new(),
        }
    }

    /// Returns the set of valid next placements.
    pub fn gen_candidates(&self) -> Vec<PlacedConfig> {
        let mut configs = Vec::new();
        for (part_id, part) in self.parts.iter().enumerate() {
            let z_rot = 0;
            for x_rot in 0..4 {
                for y_rot in 0..4 {
                    // Rotate part
                    let mut connectors = Vec::new();
                    for connector in &part.connectors {
                        let mut axis = connector.axis;
                        let mut side_a = connector.side_a;
                        let mut position = connector.position;
                        for (axis_type, num_turns) in
                            [(Axis::X, x_rot), (Axis::Z, z_rot), (Axis::Y, y_rot)]
                        {
                            let normal = match axis_type {
                                Axis::X => Vec3::X,
                                Axis::Y => Vec3::Y,
                                Axis::Z => Vec3::Z,
                            };
                            let rot = Quat::from_axis_angle(normal, std::f32::consts::PI / 2.);
                            let (axis1, axis2) = match axis_type {
                                Axis::X => (Axis::Z, Axis::Y),
                                Axis::Y => (Axis::X, Axis::Z),
                                Axis::Z => (Axis::X, Axis::Y),
                            };
                            for _ in 0..num_turns {
                                if axis == axis1 && connector.side_a {
                                    side_a = false;
                                } else if connector.axis == axis2 && !connector.side_a {
                                    side_a = true;
                                }
                                // Swap axis to other plane axis
                                axis = if axis == axis1 {
                                    axis2
                                } else if axis == axis2 {
                                    axis1
                                } else {
                                    axis
                                };
                                position = rot.mul_vec3(position);
                            }
                        }
                        let new_connector = Connector {
                            side_a,
                            axis,
                            connect_type: connector.connect_type,
                            position,
                        };
                        connectors.push(new_connector);
                    }
                    let mut bboxes = Vec::new();
                    for bbox in &part.bboxes {
                        let mut center = bbox.center;
                        let mut half_sizes = bbox.half_sizes;
                        for (axis_type, num_turns) in
                            [(Axis::X, x_rot), (Axis::Z, z_rot), (Axis::Y, y_rot)]
                        {
                            let normal = match axis_type {
                                Axis::X => Vec3::X,
                                Axis::Y => Vec3::Y,
                                Axis::Z => Vec3::Z,
                            };
                            let rot = Quat::from_axis_angle(normal, std::f32::consts::PI / 2.);
                            for _ in 0..num_turns {
                                center = rot.mul_vec3(center);
                                half_sizes = match axis_type {
                                    Axis::X => Vec3::new(half_sizes.x, half_sizes.z, half_sizes.y),
                                    Axis::Y => Vec3::new(half_sizes.z, half_sizes.y, half_sizes.x),
                                    Axis::Z => Vec3::new(half_sizes.y, half_sizes.x, half_sizes.z),
                                }
                            }
                        }
                        let new_bbox = AABB { center, half_sizes };
                        bboxes.push(new_bbox);
                    }
                    let new_part = PartData {
                        bboxes,
                        model_path: part.model_path.clone(),
                        connectors,
                    };
                }
            }
            for z_rot in [1, 3] {
                for y_rot in 0..4 {}
            }
        }
        configs
    }

    /// Places a part on the model.
    ///
    /// This doesn't check whether the config is valid, only use configs generated by `gen_candidates`.
    pub fn place_part(&mut self, _placement: &PlacedConfig) {}

    /// Returns the current model.
    pub fn get_model(&self) -> &[PlacedConfig] {
        &self.model
    }

    /// Clears the current model.
    pub fn clear_model(&mut self) {
        self.model = Vec::new();
    }
}

/// AABB bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AABB {
    pub center: Vec3,
    pub half_sizes: Vec3,
}

/// Denotes an axis.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Information on how parts can snap to each other.
/// Connectors are locked to an X, Y, and Z axis, and can be on side A or B
/// Two connectors with the same axis, different sides, and compatible connector types can be joined.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connector {
    pub side_a: bool,
    pub axis: Axis,
    pub connect_type: usize,
    pub position: Vec3,
}

/// Basic data for a part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartData {
    pub bboxes: Vec<AABB>,
    pub model_path: String,
    pub connectors: Vec<Connector>,
}

/// A part's configuration after being placed.
pub struct PlacedConfig {
    pub position: Vec3,
    pub part_id: usize,
    pub rotation: Quat,
    pub connections: Vec<Connection>,
}

/// Describes another part's connector.
pub struct Connection {
    pub placed_id: usize,
    pub connector_id: usize,
}
