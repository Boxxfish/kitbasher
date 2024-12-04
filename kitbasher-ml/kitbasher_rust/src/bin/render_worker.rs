use base64::prelude::*;
use clap::Parser;
use kitbasher_rust::{PyPlacedConfig, Renderer};
use serde::{Deserialize, Serialize};
use three_d::{SurfaceSettings, Viewport, WindowedContext};

#[derive(Parser)]
struct Args {
    part_paths: Vec<String>,
    use_mirror: bool,
    in_socket_addr: String,
    out_socket_addr: String,
}

#[derive(Deserialize)]
struct RenderMessage {
    buffer_idx: u32,
    label_idx: u32,
    prompts: Vec<String>,
    part_configs: Vec<PyPlacedConfig>,
    scorer_fn: String,
}

#[derive(Serialize)]
struct ScorerMessage {
    buffer_idx: u32,
    label_idx: u32,
    prompts: Vec<String>,
    images: Vec<String>,
    scorer_fn: String,
}

fn main() {
    let args = Args::parse();

    // Set up sockets
    let ctx = zmq::Context::new();

    let receiver = ctx.socket(zmq::PULL).unwrap();
    receiver.bind(&args.in_socket_addr).unwrap();

    let sender = ctx.socket(zmq::PUSH).unwrap();
    sender.bind(&args.out_socket_addr).unwrap();

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
        let img_buffers =
            renderer.render_model_with_window(&context, viewport, render_msg.part_configs);
        let scorer_msg = ScorerMessage {
            buffer_idx: render_msg.buffer_idx,
            label_idx: render_msg.label_idx,
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
