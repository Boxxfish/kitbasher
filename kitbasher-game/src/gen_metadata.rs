use std::io::Write;

use bevy::math::{Mat4, Vec3};
use engine::{Axis, Connector, PartData, AABB};
use ron::ser::PrettyConfig;

mod engine;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 1 {
        println!("Usage: ./gen_metadata FILE_0 ... FILE_N");
        return;
    }
    for path in &args[1..] {
        let mut connectors = Vec::new();
        let mut bboxes = Vec::new();

        let (document, _, _) = gltf::import(path).unwrap();
        for scene in document.scenes() {
            for node in scene.nodes() {
                if let Some(name) = node.name() {
                    // Connectors
                    if name.starts_with('C') {
                        let connect_type = name.split('_').collect::<Vec<_>>()[1].parse().unwrap();
                        let matrix = Mat4::from_cols_array_2d(&node.transform().matrix());
                        let (_, rot, position) = matrix.to_scale_rotation_translation();
                        let rotated_dir = rot.mul_vec3(Vec3::Y);
                        for (axis, dir, side_a) in [
                            (Axis::X, Vec3::X, true),
                            (Axis::X, -Vec3::X, false),
                            (Axis::Y, Vec3::Y, true),
                            (Axis::Y, -Vec3::Y, false),
                            (Axis::Z, Vec3::Z, true),
                            (Axis::Z, -Vec3::Z, false),
                        ] {
                            let sim = dir.dot(rotated_dir);
                            if sim > 0.9 {
                                let connector = Connector {
                                    side_a,
                                    axis,
                                    connect_type,
                                    position,
                                };
                                connectors.push(connector);
                                break;
                            }
                        }
                    }
                    // Bounding boxes
                    if name.starts_with("bbox") {
                        let mesh = node
                            .mesh()
                            .expect("Bounding boxes must have a mesh attached.");
                        for prim in mesh.primitives() {
                            let bbox = prim.bounding_box();
                            let min = Vec3::from_array(bbox.min);
                            let max = Vec3::from_array(bbox.max);
                            let half_sizes = (max - min) / 2.;
                            let aabb = AABB {
                                half_sizes,
                                center: min + half_sizes,
                            };
                            bboxes.push(aabb);
                        }
                    }
                }
            }
        }

        let part = PartData {
            bboxes,
            model_path: path.to_owned(),
            connectors,
        };
        let out_path = format!("{}.ron", path.split_at(path.len() - 4).0);
        let mut out_file = std::fs::File::create(out_path).unwrap();
        let out_str = ron::ser::to_string_pretty(&part, PrettyConfig::default()).unwrap();
        out_file.write_all(out_str.as_bytes()).unwrap();
    }
}
