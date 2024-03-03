use std::time::Instant;

pub fn str_print(msg: &str, instant: Instant) {
    println!("[{:.2}s] {}", instant.elapsed().as_secs_f32(), msg);
}
