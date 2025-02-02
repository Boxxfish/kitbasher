use base64::prelude::*;
use clap::Parser;
use kitbasher_rust::{PyPlacedConfig, Renderer};
use serde::{Deserialize, Serialize};
use three_d::{SurfaceSettings, Viewport, WindowedContext};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    part_paths: Vec<String>,
    #[arg(short, long)]
    use_mirror: bool,
    #[arg(short, long)]
    in_socket_addr: u32,
    #[arg(short, long)]
    out_socket_addr: u32,
}

#[derive(Deserialize)]
struct RenderMessage {
    buffer_idx: u32,
    label_idx: u32,
    traj_id: u32,
    prompts: Vec<String>,
    part_configs: Vec<String>,
    scorer_fn: String,
}

#[derive(Serialize)]
struct ScorerMessage {
    buffer_idx: u32,
    label_idx: u32,
    traj_id: u32,
    prompts: Vec<String>,
    images: Vec<String>,
    scorer_fn: String,
}

fn main() {
    let args = Args::parse();

    // Set up sockets
    let ctx = zmq::Context::new();

    let receiver = ctx.socket(zmq::PULL).unwrap();
    receiver
        .connect(&format!("tcp://localhost:{}", args.in_socket_addr))
        .unwrap();

    let sender = ctx.socket(zmq::PUSH).unwrap();
    sender
        .connect(&format!("tcp://localhost:{}", args.out_socket_addr))
        .unwrap();

    // Set up rendering
    let mut renderer = Renderer::new(args.part_paths.clone(), args.use_mirror);
    let viewport = Viewport::new_at_origo(512, 512);
    let window = winit::window::WindowBuilder::new()
        .build(&renderer.event_loop)
        .unwrap();
    let context = WindowedContext::from_winit_window(&window, SurfaceSettings::default()).unwrap();

    let mut msg = zmq::Message::new();
    loop {
        receiver.recv(&mut msg, 0).unwrap();
        let render_msg: RenderMessage = serde_json::from_str(msg.as_str().unwrap()).unwrap();
        let img_buffers = renderer.render_model_with_window(
            &context,
            viewport,
            render_msg
                .part_configs
                .iter()
                .map(|s| PyPlacedConfig::from_json(s))
                .collect(),
        );
        let scorer_msg = ScorerMessage {
            buffer_idx: render_msg.buffer_idx,
            label_idx: render_msg.label_idx,
            traj_id: render_msg.traj_id,
            prompts: render_msg.prompts,
            images: vec![
                BASE64_STANDARD.encode(img_buffers.0),
                BASE64_STANDARD.encode(img_buffers.1),
            ],
            scorer_fn: render_msg.scorer_fn,
        };
        let out_str = serde_json::to_string(&scorer_msg).unwrap();
        sender.send(out_str.as_bytes(), 0).unwrap();
    }
}
