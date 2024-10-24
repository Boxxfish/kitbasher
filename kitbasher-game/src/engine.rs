use bevy::{
    math::{Quat, Vec3},
    reflect::TypePath,
};
use serde::{Deserialize, Serialize};

/// An instance of the kitbasher engine.
pub struct KBEngine {
    parts: Vec<PartData>,
    connect_rules: Vec<[usize; 2]>,
    model: Vec<PlacedConfig>,
    use_mirror: bool,
}

impl KBEngine {
    /// Creates a new instance of the engine.
    pub fn new(parts: &[PartData], connect_rules: &[[usize; 2]], use_mirror: bool) -> Self {
        Self {
            parts: parts.into(),
            connect_rules: connect_rules.into(),
            model: Vec::new(),
            use_mirror,
        }
    }

    /// Adds a new part.
    pub fn add_part(&mut self, part: &PartData) {
        self.parts.push(part.clone());
    }

    /// Returns a part by its ID.
    pub fn get_part(&self, id: usize) -> &PartData {
        &self.parts[id]
    }

    /// Returns the set of valid next placements.
    pub fn gen_candidates(&self) -> Vec<PlacedConfig> {
        let mut configs = Vec::new();
        for (part_id, part) in self.parts.iter().enumerate() {
            for x_rot in 0..4 {
                if let Some(invar_x) = part.invar_x {
                    if x_rot > invar_x - 1 {
                        continue;
                    }
                }
                for y_rot in 0..4 {
                    if let Some(invar_y) = part.invar_y {
                        if y_rot > invar_y - 1 {
                            continue;
                        }
                    }
                    let new_configs = self.rotate_and_gen_next(part_id, part, x_rot, y_rot, 0);
                    configs.extend(new_configs);
                }
            }
            for z_rot in [1, 3] {
                if let Some(invar_z) = part.invar_z {
                    if z_rot > invar_z - 1 {
                        continue;
                    }
                }
                for y_rot in 0..4 {
                    if let Some(invar_y) = part.invar_y {
                        if y_rot > invar_y - 1 {
                            continue;
                        }
                    }
                    let new_configs = self.rotate_and_gen_next(part_id, part, 0, y_rot, z_rot);
                    configs.extend(new_configs);
                }
            }
        }
        configs
    }

    /// Rotates a part and returns all valid configurations for the part.
    fn rotate_and_gen_next(
        &self,
        part_id: usize,
        part: &PartData,
        x_rot: i32,
        y_rot: i32,
        z_rot: i32,
    ) -> Vec<PlacedConfig> {
        let mut configs = Vec::new();
        let mut connectors = Vec::new();
        for connector in &part.connectors {
            let mut axis = connector.axis;
            let mut position = connector.position;
            for (axis_type, num_turns) in [(Axis::X, x_rot), (Axis::Z, z_rot), (Axis::Y, y_rot)] {
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
                side_a: connector.side_a,
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
            for (axis_type, num_turns) in [(Axis::X, x_rot), (Axis::Z, z_rot), (Axis::Y, y_rot)] {
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
            invar_x: part.invar_x,
            invar_y: part.invar_y,
            invar_z: part.invar_z,
        };

        // Check if part can be attached to existing parts
        for (placed_id, placed) in self.model.iter().enumerate() {
            for (placed_connector_id, placed_connector) in placed.connectors.iter().enumerate() {
                for (part_connector_id, part_connector) in new_part.connectors.iter().enumerate() {
                    if part_connector.axis == placed_connector.axis
                        && part_connector.side_a != placed_connector.side_a
                        && (self.connect_rules.contains(&[
                            part_connector.connect_type,
                            placed_connector.connect_type,
                        ]) || self.connect_rules.contains(&[
                            placed_connector.connect_type,
                            part_connector.connect_type,
                        ]))
                    {
                        let conn_world_pos = placed.position + placed_connector.position;
                        let part_world_pos = conn_world_pos - part_connector.position;
                        // Check that new part doesn't intersect with any other parts
                        let mut intersected = false;
                        'check_bbox: for placed in &self.model {
                            for placed_bbox in &placed.bboxes {
                                let mut placed_bbox = *placed_bbox;
                                placed_bbox.center += placed.position;
                                for part_bbox in &new_part.bboxes {
                                    let mut part_bbox = *part_bbox;
                                    part_bbox.center += part_world_pos;
                                    if placed_bbox.intersects(&part_bbox) {
                                        intersected = true;
                                        break 'check_bbox;
                                    }
                                }
                            }
                        }
                        if intersected {
                            continue;
                        }
                        let mut connections = vec![None; new_part.connectors.len()];
                        connections[part_connector_id] = Some(Connection {
                            placed_id,
                            connector_id: placed_connector_id,
                        });
                        let new_placed = PlacedConfig {
                            position: part_world_pos,
                            part_id,
                            rotation: Quat::from_axis_angle(
                                Vec3::Y,
                                y_rot as f32 * (std::f32::consts::PI / 2.),
                            ) * Quat::from_axis_angle(
                                Vec3::X,
                                x_rot as f32 * (std::f32::consts::PI / 2.),
                            ) * Quat::from_axis_angle(
                                Vec3::Z,
                                z_rot as f32 * (std::f32::consts::PI / 2.),
                            ),
                            connectors: new_part.connectors.clone(),
                            bboxes: new_part.bboxes.clone(),
                            connections,
                        };
                        configs.push(new_placed);
                    }
                }
            }
        }

        // If mirroring is enabled and a configuration crosses the x axis, remove it
        if self.use_mirror {
            configs.retain(|config| {
                let mut crossed = false;
                for bbox in &config.bboxes {
                    let x = bbox.center.x + config.position.x - bbox.half_sizes.x;
                    if x < -0.001 {
                        crossed = true;
                        break;
                    }
                }
                !crossed
            });
        }

        // If any configurations share the same position and rotation, merge them
        let mut merged_configs: Vec<PlacedConfig> = Vec::new();
        for config in configs {
            let mut merged = false;
            for merged_config in &mut merged_configs {
                if config.position.distance_squared(merged_config.position) < 0.01
                    && config.rotation.angle_between(merged_config.rotation) < 0.01
                {
                    merged = true;
                    for (i, connection) in config.connections.iter().copied().enumerate() {
                        if let Some(connection) = connection {
                            merged_config.connections[i] = Some(connection);
                        }
                    }
                    break;
                }
            }
            if !merged {
                merged_configs.push(config);
            }
        }

        merged_configs
    }

    /// Places a part on the model.
    ///
    /// This doesn't check whether the config is valid, only use configs generated by `gen_candidates`.
    pub fn place_part(&mut self, placement: &PlacedConfig) {
        let placed_id = self.model.len();
        // Connect other parts to new parts
        for (connector_id, connection) in placement.connections.iter().enumerate() {
            if let Some(connection) = connection {
                self.model[connection.placed_id].connections[connection.connector_id] =
                    Some(Connection {
                        placed_id,
                        connector_id,
                    });
            }
        }
        self.model.push(placement.clone());
    }

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
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct AABB {
    pub center: Vec3,
    pub half_sizes: Vec3,
}

impl AABB {
    pub fn intersects(&self, other: &Self) -> bool {
        let diff = (self.center - other.center).abs();
        let max_diffs = self.half_sizes + other.half_sizes;
        diff.x < max_diffs.x && diff.y < max_diffs.y && diff.z < max_diffs.z
    }
    pub fn union(&self, other: &Self) -> Self {
        let min = (self.center - self.half_sizes).min(other.center - other.half_sizes);
        let max = (self.center + self.half_sizes).max(other.center + other.half_sizes);
        let half_sizes = (max - min) / 2.;
        Self {
            half_sizes,
            center: min + half_sizes,
        }
    }
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
#[derive(Debug, Clone, Serialize, Deserialize, bevy::asset::Asset, TypePath)]
pub struct PartData {
    pub bboxes: Vec<AABB>,
    pub model_path: String,
    pub connectors: Vec<Connector>,
    pub invar_x: Option<i32>,
    pub invar_y: Option<i32>,
    pub invar_z: Option<i32>,
}

/// A part's configuration after being placed.
#[derive(Debug, Clone)]
pub struct PlacedConfig {
    pub position: Vec3,
    pub part_id: usize,
    pub rotation: Quat,
    /// Rotated connectors, cached for efficiency.
    pub connectors: Vec<Connector>,
    /// Rotated bounding boxes, cached for efficiency.
    pub bboxes: Vec<AABB>,
    pub connections: Vec<Option<Connection>>,
}

/// Describes another part's connector.
#[derive(Debug, Copy, Clone)]
pub struct Connection {
    pub placed_id: usize,
    pub connector_id: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that a part can connect to itself.
    #[test]
    fn conn_self() {
        let parts = [PartData {
            bboxes: vec![AABB {
                center: Vec3::ZERO,
                half_sizes: Vec3::splat(0.49),
            }],
            model_path: "".into(),
            connectors: vec![
                Connector {
                    side_a: true,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(0., 0.5, 0.),
                },
                Connector {
                    side_a: false,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(0., -0.5, 0.),
                },
            ],
            invar_x: None,
            invar_y: Some(1),
            invar_z: None,
        }];
        let mut engine = KBEngine::new(&parts, &[[0, 0]], false);
        engine.place_part(&PlacedConfig {
            position: Vec3::ZERO,
            part_id: 0,
            rotation: Quat::IDENTITY,
            connectors: parts[0].connectors.clone(),
            bboxes: parts[0].bboxes.clone(),
            connections: vec![None; parts[0].connectors.len()],
        });
        let candidates = engine.gen_candidates();
        assert_eq!(candidates.len(), 2);
    }

    /// Tests that configurations with the same position and rotation get merged.
    #[test]
    fn merging() {
        let parts = [PartData {
            bboxes: vec![AABB {
                center: Vec3::ZERO,
                half_sizes: Vec3::splat(0.49),
            }],
            model_path: "".into(),
            connectors: vec![
                Connector {
                    side_a: true,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(0.4, 0.5, 0.4),
                },
                Connector {
                    side_a: true,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(0.4, 0.5, -0.4),
                },
                Connector {
                    side_a: true,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(-0.4, 0.5, 0.4),
                },
                Connector {
                    side_a: true,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(-0.4, 0.5, -0.4),
                },
                Connector {
                    side_a: false,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(0.4, -0.5, 0.4),
                },
                Connector {
                    side_a: false,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(0.4, -0.5, -0.4),
                },
                Connector {
                    side_a: false,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(-0.4, -0.5, 0.4),
                },
                Connector {
                    side_a: false,
                    axis: Axis::Y,
                    connect_type: 0,
                    position: Vec3::new(-0.4, -0.5, -0.4),
                },
            ],
            invar_x: None,
            invar_y: Some(1),
            invar_z: None,
        }];
        let mut engine = KBEngine::new(&parts, &[[0, 0]], false);
        engine.place_part(&PlacedConfig {
            position: Vec3::ZERO,
            part_id: 0,
            rotation: Quat::IDENTITY,
            connectors: parts[0].connectors.clone(),
            bboxes: parts[0].bboxes.clone(),
            connections: vec![None; parts[0].connectors.len()],
        });
        let candidates = engine.gen_candidates();
        assert_eq!(candidates.len(), 18);
    }
}
