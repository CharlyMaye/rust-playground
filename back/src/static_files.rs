use actix_web::web;
use actix_files as fs;

pub fn static_files_config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        fs::Files::new(
            "/",
             "./www/"
        )
        .index_file("index.html")
    );
}
