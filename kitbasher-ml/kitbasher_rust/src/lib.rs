use std::{
    collections::{HashMap, VecDeque},
    f32::consts::PI,
};

use bevy::math::{Mat3, Quat, Vec3};
use kitbasher_game::engine::{Axis, Connection, Connector, KBEngine, PartData, PlacedConfig, AABB};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use three_d::*;

#[pyclass]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct PyVec3 {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
    #[pyo3(get)]
    pub z: f32,
}

impl From<Vec3> for PyVec3 {
    fn from(value: Vec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<PyVec3> for Vec3 {
    fn from(val: PyVec3) -> Self {
        Vec3::new(val.x, val.y, val.z)
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct PyQuat {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
    #[pyo3(get)]
    pub z: f32,
    #[pyo3(get)]
    pub w: f32,
}

impl From<Quat> for PyQuat {
    fn from(value: Quat) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: value.w,
        }
    }
}

impl From<PyQuat> for Quat {
    fn from(val: PyQuat) -> Self {
        Quat::from_xyzw(val.x, val.y, val.z, val.w)
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PyAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl From<Axis> for PyAxis {
    fn from(value: Axis) -> Self {
        match value {
            Axis::X => Self::X,
            Axis::Y => Self::Y,
            Axis::Z => Self::Z,
        }
    }
}

impl From<PyAxis> for Axis {
    fn from(val: PyAxis) -> Self {
        match val {
            PyAxis::X => Axis::X,
            PyAxis::Y => Axis::Y,
            PyAxis::Z => Axis::Z,
        }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct PyConnector {
    #[pyo3(get)]
    pub side_a: bool,
    #[pyo3(get)]
    pub axis: PyAxis,
    #[pyo3(get)]
    pub connect_type: usize,
    #[pyo3(get)]
    pub position: PyVec3,
}

impl From<Connector> for PyConnector {
    fn from(value: Connector) -> Self {
        Self {
            side_a: value.side_a,
            axis: value.axis.into(),
            connect_type: value.connect_type,
            position: value.position.into(),
        }
    }
}

impl From<PyConnector> for Connector {
    fn from(val: PyConnector) -> Self {
        Connector {
            side_a: val.side_a,
            axis: val.axis.into(),
            connect_type: val.connect_type,
            position: val.position.into(),
        }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct PyAABB {
    #[pyo3(get)]
    pub center: PyVec3,
    #[pyo3(get)]
    pub half_sizes: PyVec3,
}

impl From<AABB> for PyAABB {
    fn from(value: AABB) -> Self {
        Self {
            center: value.center.into(),
            half_sizes: value.half_sizes.into(),
        }
    }
}

impl From<PyAABB> for AABB {
    fn from(val: PyAABB) -> Self {
        AABB {
            center: val.center.into(),
            half_sizes: val.half_sizes.into(),
        }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct PyConnection {
    #[pyo3(get)]
    pub placed_id: usize,
    #[pyo3(get)]
    pub connector_id: usize,
}

impl From<Connection> for PyConnection {
    fn from(value: Connection) -> Self {
        Self {
            placed_id: value.placed_id,
            connector_id: value.connector_id,
        }
    }
}

impl From<PyConnection> for Connection {
    fn from(val: PyConnection) -> Self {
        Connection {
            placed_id: val.placed_id,
            connector_id: val.connector_id,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyPlacedConfig {
    #[pyo3(get)]
    pub position: PyVec3,
    #[pyo3(get)]
    pub part_id: usize,
    #[pyo3(get)]
    pub rotation: PyQuat,
    #[pyo3(get)]
    pub connectors: Vec<PyConnector>,
    #[pyo3(get)]
    pub bboxes: Vec<PyAABB>,
    #[pyo3(get, set)]
    pub connections: Vec<Option<PyConnection>>,
}

#[pymethods]
impl PyPlacedConfig {
    pub fn to_json(&self) -> String {
        serde_json::ser::to_string(self).unwrap()
    }

    #[staticmethod]
    pub fn from_json(s: &str) -> PyPlacedConfig {
        serde_json::de::from_str(s).unwrap()
    }
}

impl From<PlacedConfig> for PyPlacedConfig {
    fn from(value: PlacedConfig) -> Self {
        Self {
            position: value.position.into(),
            part_id: value.part_id,
            rotation: value.rotation.into(),
            connectors: value
                .connectors
                .iter()
                .map(|c| c.to_owned().into())
                .collect(),
            bboxes: value.bboxes.iter().map(|b| b.to_owned().into()).collect(),
            connections: value
                .connections
                .iter()
                .map(|c| c.to_owned().map(|c| c.into()))
                .collect(),
        }
    }
}

impl From<PyPlacedConfig> for PlacedConfig {
    fn from(val: PyPlacedConfig) -> Self {
        PlacedConfig {
            position: val.position.into(),
            part_id: val.part_id,
            rotation: val.rotation.into(),
            connectors: val.connectors.iter().map(|c| c.to_owned().into()).collect(),
            bboxes: val.bboxes.iter().map(|b| b.to_owned().into()).collect(),
            connections: val
                .connections
                .iter()
                .map(|c| c.to_owned().map(|c| c.into()))
                .collect(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PartReference {
    part_id: usize,
    pos_offset_x: f32,
    pos_offset_y: f32,
    pos_offset_z: f32,
    rot_offset_x: u8,
    rot_offset_y: u8,
    rot_offset_z: u8,
}

#[pymethods]
impl PartReference {
    #[new]
    fn new(
        part_id: usize,
        pos_offset_x: f32,
        pos_offset_y: f32,
        pos_offset_z: f32,
        rot_offset_x: u8,
        rot_offset_y: u8,
        rot_offset_z: u8,
    ) -> PartReference {
        PartReference {
            part_id,
            pos_offset_x,
            pos_offset_y,
            pos_offset_z,
            rot_offset_x,
            rot_offset_y,
            rot_offset_z,
        }
    }
}

#[pyclass]
pub struct EngineWrapper {
    engine: KBEngine,
}

#[pymethods]
impl EngineWrapper {
    #[new]
    pub fn new(
        part_paths: Vec<String>,
        connect_rules: Vec<[usize; 2]>,
        use_mirror: bool,
    ) -> EngineWrapper {
        let mut parts = Vec::new();
        for path in &part_paths {
            let content = std::fs::read_to_string(path).unwrap();
            let part = ron::from_str(&content).unwrap();
            parts.push(part);
        }
        let engine = KBEngine::new(&parts, &connect_rules, use_mirror);
        EngineWrapper { engine }
    }

    pub fn clear_model(&mut self) {
        self.engine.clear_model();
    }

    pub fn gen_candidates(&self) -> Vec<PyPlacedConfig> {
        self.engine
            .gen_candidates()
            .iter()
            .map(|p| p.to_owned().into())
            .collect()
    }

    pub fn get_model(&self) -> Vec<PyPlacedConfig> {
        self.engine
            .get_model()
            .iter()
            .map(|p| p.to_owned().into())
            .collect()
    }

    pub fn set_model(&mut self, model: Vec<PyPlacedConfig>) {
        self.engine
            .set_model(&model.iter().cloned().map(|p| p.into()).collect::<Vec<_>>());
    }

    pub fn place_part(&mut self, placement: PyPlacedConfig) {
        self.engine.place_part(&placement.into());
    }

    pub fn create_config(&self, part_id: usize, x: f32, y: f32, z: f32) -> PyPlacedConfig {
        let part = self.engine.get_part(part_id);
        PyPlacedConfig {
            position: PyVec3 { x, y, z },
            part_id,
            rotation: Quat::IDENTITY.into(),
            connectors: part
                .connectors
                .clone()
                .iter()
                .map(|x| x.clone().into())
                .collect(),
            bboxes: part.bboxes.clone().iter().map(|x| (*x).into()).collect(),
            connections: vec![None; part.connectors.len()],
        }
    }

    // Constructs the ldraw model in the engine.
    pub fn load_ldraw(
        &mut self,
        path: &str,
        ref_map: HashMap<String, PartReference>,
        use_mirror: bool,
    ) {
        let pos_scale = 0.25;
        let file_bytes = std::fs::read(path).unwrap();
        let commands = weldr::parse_raw(&file_bytes).unwrap();
        for cmd in &commands {
            if let weldr::Command::SubFileRef(cmd) = cmd {
                if let weldr::SubFileRef::UnresolvedRef(subref) = &cmd.file {
                    let subref = subref.replace(".dat", "");
                    let part_ref = ref_map
                        .get(&subref)
                        .unwrap_or_else(|| panic!("\"{subref}\" was not present in `ref_map`."));
                    let part_id = part_ref.part_id;
                    let part = self.engine.get_part(part_id);

                    // Transform our part to ldraw part rotation + position
                    let pos_offset = Vec3::new(
                        part_ref.pos_offset_x,
                        part_ref.pos_offset_y,
                        part_ref.pos_offset_z,
                    );
                    let (new_part, new_rotation) = self.engine.rotate_part(
                        part,
                        part_ref.rot_offset_x as i32,
                        part_ref.rot_offset_y as i32,
                        part_ref.rot_offset_z as i32,
                    );

                    // Place part on model
                    let ldraw_pos = Vec3::new(cmd.pos.x, -cmd.pos.y, cmd.pos.z) * pos_scale; // ldraw uses -Y
                    let xform = Mat3::from_cols(
                        Vec3::new(cmd.row0.x, cmd.row0.y, cmd.row0.z),
                        Vec3::new(cmd.row1.x, cmd.row1.y, cmd.row1.z),
                        Vec3::new(cmd.row2.x, cmd.row2.y, cmd.row2.z),
                    )
                    .transpose();
                    let ldraw_rot = Quat::from_mat3(&xform);
                    let xformed_pos_offset = ldraw_rot.mul_vec3(pos_offset);
                    let (x_rot, y_rot, z_rot) = ldraw_rot.to_euler(bevy::math::EulerRot::XYZ);
                    let x_rot = ((x_rot / (PI / 2.)).round() as i32) % 4;
                    let y_rot = ((y_rot / (PI / 2.)).round() as i32) % 4;
                    let z_rot = ((z_rot / (PI / 2.)).round() as i32) % 4;
                    let (new_part, rotation) =
                        self.engine.rotate_part(&new_part, x_rot, y_rot, z_rot);
                    let new_rotation = rotation.mul_quat(new_rotation);
                    let new_position = ldraw_pos - xformed_pos_offset;

                    // If we're mirroring and the part crosses the X axis, skip adding it
                    if use_mirror && !self.engine.get_model().is_empty() {
                        let x_origin = self.engine.get_model()[0].bboxes[0].center.x;
                        let mut crossed = false;
                        for bbox in &new_part.bboxes {
                            let x = bbox.center.x + new_position.x - bbox.half_sizes.x;
                            if x < x_origin + -0.001 {
                                crossed = true;
                                break;
                            }
                        }
                        if crossed {
                            continue;
                        }
                    }

                    let config = PlacedConfig {
                        bboxes: new_part.bboxes.clone(),
                        connections: new_part.connectors.iter().map(|_| None).collect(),
                        connectors: new_part.connectors.clone(),
                        part_id,
                        position: new_position,
                        rotation: new_rotation,
                    };
                    self.engine.place_part(&config);
                }
            }
        }

        // Connect parts together
        // Assumption: Only two connectors ever overlap each other
        let model = &mut self.engine.model;
        for part1_idx in 0..model.len() {
            for c1_idx in 0..model[part1_idx].connectors.len() {
                if model[part1_idx].connections[c1_idx].is_some() {
                    continue;
                }
                'find_conn: for part2_idx in 0..model.len() {
                    if part1_idx == part2_idx {
                        continue;
                    }
                    let part2 = &model[part2_idx];
                    for c2_idx in 0..part2.connectors.len() {
                        if part2.connections[c2_idx].is_some() {
                            continue;
                        }
                        let c1 = &model[part1_idx].connectors[c1_idx];
                        let c2 = &part2.connectors[c2_idx];
                        let c1_world_pos = model[part1_idx].position + c1.position;
                        let c2_world_pos = part2.position + c2.position;
                        if c1_world_pos.abs_diff_eq(c2_world_pos, 2.)
                            && c1.axis == c2.axis
                            && c1.side_a != c2.side_a
                            && (self
                                .engine
                                .connect_rules
                                .contains(&[c1.connect_type, c2.connect_type])
                                || self
                                    .engine
                                    .connect_rules
                                    .contains(&[c1.connect_type, c2.connect_type]))
                        {
                            model[part1_idx].connections[c1_idx] = Some(Connection {
                                placed_id: part2_idx,
                                connector_id: c2_idx,
                            });
                            model[part2_idx].connections[c2_idx] = Some(Connection {
                                placed_id: part1_idx,
                                connector_id: c1_idx,
                            });
                            println!("Connected {part1_idx} to {part2_idx}");
                            break 'find_conn;
                        }
                    }
                }
            }
        }
    }

    /// Shuffles the indices of the parts of the model.
    /// This is useful when popping off the last pieces of the model to get a sequence of partial build states.
    /// The new model is guaranteed to be buildable (e.g. all parts after the first part connect to some part of the model).
    /// Additionally, the first part will be unchanged.
    pub fn shuffle_model_parts(&mut self) {
        // Determine new indices
        let mut rng = rand::thread_rng();
        let mut new_to_old: Vec<_> = vec![0]; // the new `i` is the old `new_idxs[i]`
        for _ in 1..self.engine.model.len() {
            let mut candidates = Vec::new();
            for old_idx in 0..self.engine.model.len() {
                // Skip if this part is already in the new model
                if new_to_old.contains(&old_idx) {
                    continue;
                }
                // Add to candidates if this part connects to any parts in the new model
                for conn in self.engine.model[old_idx].connections.iter().flatten() {
                    if new_to_old.contains(&conn.placed_id) {
                        candidates.push(old_idx);
                        break;
                    }
                }
            }
            new_to_old.push(*candidates.choose(&mut rng).unwrap());
        }
        let mut old_to_new: Vec<_> = (0..self.engine.get_model().len()).map(|_| 0).collect();
        for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
            old_to_new[old_idx] = new_idx;
        }

        // Create new model
        let old_model = self.engine.get_model();
        let mut new_model = Vec::new();
        for old_idx in &new_to_old {
            let mut new_part = old_model[*old_idx].clone();
            for conn_idx in 0..new_part.connections.len() {
                if let Some(conn) = &mut new_part.connections[conn_idx] {
                    conn.placed_id = old_to_new[conn.placed_id];
                }
            }
            new_model.push(new_part);
        }

        self.engine.set_model(&new_model);
    }

    /// Pops off the last part of the model and returns it as a part config.
    /// This can be directly added to the model again via `place_part`.
    /// Connections to the popped part will be removed from the model, but the
    /// returned config will retain these connections (the same as any other part candidate).
    pub fn pop_part(&mut self) -> PyPlacedConfig {
        let popped_idx = self.engine.model.len() - 1;
        let popped_cfg = self
            .engine
            .model
            .pop()
            .expect("Must be at least one part on model.");
        for part in &mut self.engine.model {
            for conn_idx in 0..part.connections.len() {
                if let Some(conn) = &part.connections[conn_idx] {
                    if conn.placed_id == popped_idx {
                        part.connections[conn_idx] = None;
                    }
                }
            }
        }
        popped_cfg.into()
    }
}

#[pyclass(unsendable)]
pub struct Renderer {
    part_models: Vec<(CpuMesh, CpuMaterial)>,
    use_mirror: bool,
    pub event_loop: winit::event_loop::EventLoop<()>,
}

#[pymethods]
impl Renderer {
    #[new]
    pub fn new(part_paths: Vec<String>, use_mirror: bool) -> Self {
        let mut part_models = Vec::new();
        for path in &part_paths {
            let (document, buffers, _) = gltf::import(path).unwrap();
            for scene in document.scenes() {
                for node in scene.nodes() {
                    if node.name().is_some() && node.name().unwrap().contains("bbox") {
                        continue;
                    }
                    if let Some(mesh) = node.mesh() {
                        let xform = node.transform().matrix();
                        let cols = xform.as_slice();
                        let xform = Matrix4::from_cols(
                            cols[0].into(),
                            cols[1].into(),
                            cols[2].into(),
                            cols[3].into(),
                        );
                        // let norm_xform = xform.try_inverse().unwrap().transpose();
                        for prim in mesh.primitives() {
                            let reader = prim.reader(|buffer| {
                                buffers.get(buffer.index()).map(|data| &data.0[..])
                            });
                            let color =
                                prim.material().pbr_metallic_roughness().base_color_factor();
                            let color = Vec3::new(color[0], color[1], color[2]);
                            let positions: Vec<_> = reader
                                .read_positions()
                                .unwrap()
                                .map(|x| xform.transform_point(x.into()).to_vec())
                                .collect();
                            let indices: Vec<_> = reader
                                .read_indices()
                                .unwrap()
                                .into_u32()
                                .map(|x| x as u16)
                                .collect::<Vec<_>>();
                            let mut part_mesh = CpuMesh {
                                positions: Positions::F32(positions.clone()),
                                indices: Indices::U16(indices.clone()),
                                colors: None,
                                normals: None,
                                tangents: None,
                                uvs: None,
                            };
                            part_mesh.compute_normals();

                            part_models.push((
                                part_mesh,
                                CpuMaterial {
                                    albedo: Srgba::new(
                                        (color.x * 255.) as u8,
                                        (color.y * 255.) as u8,
                                        (color.z * 255.) as u8,
                                        255,
                                    ),
                                    ..Default::default()
                                },
                            ));
                        }
                    }
                }
            }
        }

        let event_loop = winit::event_loop::EventLoopBuilder::new().build();

        Self {
            part_models,
            use_mirror,
            event_loop,
        }
    }

    /// Renders the model to an image and returns a byte array.
    pub fn render_model(&mut self, model: Vec<PyPlacedConfig>) -> (Vec<u8>, Vec<u8>) {
        let viewport = Viewport::new_at_origo(512, 512);
        // let context = HeadlessContext::new().unwrap();
        let window = winit::window::WindowBuilder::new()
            .build(&self.event_loop)
            .unwrap();
        let context =
            WindowedContext::from_winit_window(&window, SurfaceSettings::default()).unwrap();

        self.render_model_with_window(&context, viewport, model)
    }
}

impl Renderer {
    /// Same as `render_model`, but it expects a window.
    pub fn render_model_with_window(
        &mut self,
        context: &WindowedContext,
        viewport: Viewport,
        model: Vec<PyPlacedConfig>,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut render_tex = Texture2D::new_empty::<[u8; 4]>(
            context,
            viewport.width,
            viewport.height,
            Interpolation::Nearest,
            Interpolation::Nearest,
            None,
            Wrapping::ClampToEdge,
            Wrapping::ClampToEdge,
        );
        let mut depth_tex = DepthTexture2D::new::<f32>(
            context,
            viewport.width,
            viewport.height,
            Wrapping::ClampToEdge,
            Wrapping::ClampToEdge,
        );
        let mut model_bbox: AABB = model[0]
            .bboxes
            .iter()
            .copied()
            .reduce(|a, b| AABB::from(a).union(&b.into()).into())
            .unwrap()
            .into();
        let mut models = Vec::new();
        let outline_material = CpuMaterial {
            albedo: Srgba::new(0, 0, 0, 255),
            ..Default::default()
        };
        for placed in &model {
            let (part_model, material) = &self.part_models[placed.part_id];
            // Render both original part and outline
            let part_xform = Matrix4::from_translation(Vector3::new(
                placed.position.x,
                placed.position.y,
                placed.position.z,
            )) * Matrix4::from(Quaternion::new(
                placed.rotation.w,
                placed.rotation.x,
                placed.rotation.y,
                placed.rotation.z,
            ));

            // Part model
            let mut model = Gm::new(
                Mesh::new(context, part_model),
                PhysicalMaterial::new_opaque(context, material),
            );
            model.set_transformation(part_xform);
            models.push(model);

            // Outline
            let mut outline_mat = PhysicalMaterial::new_opaque(context, &outline_material);
            outline_mat.render_states.cull = Cull::Front;
            let mut model = Gm::new(Mesh::new(context, part_model), outline_mat);
            model.set_transformation(part_xform * Matrix4::from_scale(1.1));
            models.push(model);

            // Render a flipped version of the part if mirroring
            if self.use_mirror {
                let mirrored_xform = Matrix4::from_nonuniform_scale(-1., 1., 1.) * part_xform;
                let mut model = Gm::new(
                    Mesh::new(context, part_model),
                    PhysicalMaterial::new_opaque(context, material),
                );
                model.material.render_states.cull = Cull::None;
                model.update_positions(
                    &part_model
                        .positions
                        .to_f32()
                        .iter()
                        .map(|p| {
                            mirrored_xform
                                .transform_point(Point3::from_vec(*p))
                                .to_vec()
                        })
                        .collect::<Vec<_>>(),
                );
                model.update_normals(
                    &part_model
                        .normals
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|n| {
                            -mirrored_xform
                                .invert()
                                .unwrap()
                                .transpose()
                                .transform_vector(*n)
                                .normalize()
                        })
                        .collect::<Vec<_>>(),
                );
                models.push(model);

                let mut outline_mat = PhysicalMaterial::new_opaque(context, &outline_material);
                outline_mat.render_states.cull = Cull::Back;
                let mut model = Gm::new(Mesh::new(context, part_model), outline_mat);
                model.set_transformation(mirrored_xform * Matrix4::from_scale(1.1));
                models.push(model);
            }

            // Update model bbox
            model_bbox = model_bbox.union(
                &placed
                    .bboxes
                    .iter()
                    .map(|bbox| AABB {
                        center: Vec3::from(bbox.center) + Vec3::from(placed.position),
                        half_sizes: bbox.half_sizes.into(),
                    })
                    .reduce(|a, b| a.union(&b))
                    .unwrap(),
            );
        }

        // Update bbox if mirroring
        if self.use_mirror {
            let mut flipped_bbox = model_bbox;
            flipped_bbox.center.x = -flipped_bbox.center.x;
            model_bbox = model_bbox.union(&flipped_bbox);
        }

        let model_center = Vector3::from(model_bbox.center.to_array());
        let at = model_center;

        // Render both front and back
        let mut buffers = Vec::new();
        for offset in [vec3(80., 50., 100.), vec3(-100., 50., -80.)] {
            let eye = offset + model_center;
            let directional = DirectionalLight::new(context, 2.0, Srgba::WHITE, &(at - eye));
            let camera = Camera::new_perspective(
                viewport,
                eye,
                at,
                -Vector3::unit_y(),
                degrees(60.0),
                0.1,
                1000.0,
            );

            // Debug render the bounding box
            // let aabb = AxisAlignedBoundingBox::new_with_positions(&[
            //     (model_bbox.center + model_bbox.half_sizes)
            //         .to_array()
            //         .into(),
            //     (model_bbox.center - model_bbox.half_sizes)
            //         .to_array()
            //         .into(),
            // ]);
            // let bounding_box_cube = Gm::new(
            //     BoundingBox::new(context, aabb),
            //     PhysicalMaterial::new_opaque(context, &outline_material),
            // );

            let buffer = RenderTarget::new(
                render_tex.as_color_target(None),
                depth_tex.as_depth_target(),
            )
            .clear(ClearState::color_and_depth(1., 1., 1., 1., 1.))
            .render(&camera, &models, &[&directional])
            // .render(&camera, vec![bounding_box_cube], &[&directional])
            .read_color::<[u8; 4]>();
            buffers.push(buffer.into_flattened());
        }

        (buffers[0].clone(), buffers[1].clone())
    }
}

#[pymodule]
fn kitbasher_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EngineWrapper>()?;
    m.add_class::<PyPlacedConfig>()?;
    m.add_class::<PyVec3>()?;
    m.add_class::<PyQuat>()?;
    m.add_class::<PyAABB>()?;
    m.add_class::<PyAxis>()?;
    m.add_class::<PyConnection>()?;
    m.add_class::<PyConnector>()?;
    m.add_class::<Renderer>()?;
    m.add_class::<PartReference>()?;
    Ok(())
}
