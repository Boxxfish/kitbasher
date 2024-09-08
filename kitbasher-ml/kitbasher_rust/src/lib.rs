use bevy::math::{Quat, Vec3};
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
    #[pyo3(get)]
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

#[pymodule]
fn kitbasher_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EngineWrapper>()?;
    m.add_class::<PyPlacedConfig>()?;
    m.add_class::<PyQuat>()?;
    m.add_class::<PyVec3>()?;
    m.add_class::<PyAABB>()?;
    m.add_class::<PyAxis>()?;
    m.add_class::<PyConnection>()?;
    m.add_class::<PyConnector>()?;
    Ok(())
}
