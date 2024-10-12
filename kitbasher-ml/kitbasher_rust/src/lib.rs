use std::{cell::RefCell, rc::Rc};

use bevy::math::{Quat, Vec3};
use kiss3d::{
    camera::{Camera, FirstPerson},
    nalgebra::{self, Quaternion, UnitQuaternion, Vector3},
    resource::Mesh,
    window::Window,
};
use kitbasher_game::engine::{Axis, Connection, Connector, KBEngine, PlacedConfig, AABB};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Clone)]
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
pub struct EngineWrapper {
    engine: KBEngine,
}

#[pymethods]
impl EngineWrapper {
    #[new]
    pub fn new(part_paths: Vec<String>, connect_rules: Vec<[usize; 2]>) -> EngineWrapper {
        let mut parts = Vec::new();
        for path in &part_paths {
            let content = std::fs::read_to_string(path).unwrap();
            let part = ron::from_str(&content).unwrap();
            parts.push(part);
        }
        let engine = KBEngine::new(&parts, &connect_rules);
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
}

#[pyclass]
struct Renderer {
    part_models: Vec<kiss3d::resource::Mesh>,
    part_model_outlines: Vec<kiss3d::resource::Mesh>,
}

#[pymethods]
impl Renderer {
    #[new]
    pub fn new(part_paths: Vec<String>) -> Self {
        let mut part_models = Vec::new();
        let mut part_model_outlines = Vec::new();
        for path in &part_paths {
            let (document, buffers, _) = gltf::import(path).unwrap();
            for scene in document.scenes() {
                for node in scene.nodes() {
                    if node.name().is_some() && node.name().unwrap().contains("bbox") {
                        continue;
                    }
                    if let Some(mesh) = node.mesh() {
                        let xform = nalgebra::Matrix4::from_column_slice(
                            node.transform().matrix().as_slice().as_flattened(),
                        );
                        let norm_xform = xform.try_inverse().unwrap().transpose();
                        for prim in mesh.primitives() {
                            let reader = prim.reader(|buffer| {
                                buffers.get(buffer.index()).map(|data| &data.0[..])
                            });
                            let positions: Vec<_> = reader
                                .read_positions()
                                .unwrap()
                                .map(|x| xform.transform_point(&nalgebra::Point3::from(x)))
                                .collect();
                            // let normals = reader
                            //     .read_normals()
                            //     .unwrap()
                            //     .map(|x| norm_xform.transform_vector(&nalgebra::Point3::from(x)))
                            //     .collect();
                            let indices: Vec<_> = reader
                                .read_indices()
                                .unwrap()
                                .into_u32()
                                .map(|x| x as u16)
                                .collect::<Vec<_>>()
                                .chunks_exact(3)
                                .map(|x| kiss3d::nalgebra::Point3::new(x[0], x[1], x[2]))
                                .collect();
                            let part_mesh = kiss3d::resource::Mesh::new(
                                positions.clone(),
                                indices.clone(),
                                None,
                                None,
                                false,
                            );
                            let part_mesh_outline = kiss3d::resource::Mesh::new(
                                positions,
                                indices.iter().map(|x| x.yxz()).collect(),
                                None,
                                None,
                                false,
                            );
                            part_models.push(part_mesh);
                            part_model_outlines.push(part_mesh_outline);
                        }
                    }
                }
            }
        }

        Self {
            part_models,
            part_model_outlines,
        }
    }

    /// Renders the model to an image and returns a byte array.
    pub fn render_model(&self, model: Vec<PyPlacedConfig>) -> (Vec<u8>, Vec<u8>) {
        let mut window = Window::new_with_size("Model Renderer", 512, 512);
        window.set_light(kiss3d::light::Light::StickToCamera);
        window.set_background_color(1., 1., 1.);
        window.render();

        let part_models: Vec<_> = self
            .part_models
            .iter()
            .map(|m| {
                Rc::new(RefCell::new(Mesh::new(
                    m.coords().read().unwrap().data().as_ref().unwrap().clone(),
                    m.faces().read().unwrap().data().as_ref().unwrap().clone(),
                    m.normals().read().unwrap().data().clone(),
                    m.uvs().read().unwrap().data().clone(),
                    false,
                )))
            })
            .collect();
        let part_model_outlines: Vec<_> = self
            .part_model_outlines
            .iter()
            .map(|m| {
                Rc::new(RefCell::new(Mesh::new(
                    m.coords().read().unwrap().data().as_ref().unwrap().clone(),
                    m.faces().read().unwrap().data().as_ref().unwrap().clone(),
                    m.normals().read().unwrap().data().clone(),
                    m.uvs().read().unwrap().data().clone(),
                    false,
                )))
            })
            .collect();
        let mut model_bbox: AABB = model[0]
            .bboxes
            .iter()
            .copied()
            .reduce(|a, b| AABB::from(a).union(&b.into()).into())
            .unwrap()
            .into();
        let mut root = window.add_group();
        for placed in &model {
            let part_model = part_models[placed.part_id].clone();
            let part_model_outline = part_model_outlines[placed.part_id].clone();
            // Render both original part and outline
            let part_xform = kiss3d::nalgebra::Isometry3::from_parts(
                kiss3d::nalgebra::Translation3::new(
                    placed.position.x,
                    placed.position.y,
                    placed.position.z,
                ),
                UnitQuaternion::from_quaternion(Quaternion::new(
                    placed.rotation.w,
                    placed.rotation.x,
                    placed.rotation.y,
                    placed.rotation.z,
                )),
            );
            let mut c1 = root.add_mesh(part_model, Vector3::new(1., 1., 1.));
            c1.set_color(1., 0., 0.);
            c1.prepend_to_local_transformation(&part_xform);

            let mut c2 = root.add_mesh(part_model_outline, Vector3::new(1.05, 1.05, 1.05));
            c2.set_color(0., 0., 0.);
            c2.enable_backface_culling(true);
            c2.prepend_to_local_transformation(&part_xform);

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

        let model_center = nalgebra::Vector3::from(model_bbox.center.to_array());
        let at = model_center;

        // Render both front and back
        let eye = nalgebra::Vector3::new(100., 50., 100.) + model_center;
        let mut fp = FirstPerson::new(eye.into(), at.into());
        fp.set_up_axis(-nalgebra::Vector3::y());
        window.render_with_camera(&mut fp);
        window.render_with_camera(&mut fp);

        let mut buffer1 = Vec::new();
        window.snap(&mut buffer1);
        
        let eye = nalgebra::Vector3::new(-100., 50., -100.) + model_center;
        let mut fp = FirstPerson::new(eye.into(), at.into());
        fp.set_up_axis(-nalgebra::Vector3::y());
        window.render_with_camera(&mut fp);
        window.render_with_camera(&mut fp);

        let mut buffer2 = Vec::new();
        window.snap(&mut buffer2);

        (buffer1, buffer2)
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
    Ok(())
}
