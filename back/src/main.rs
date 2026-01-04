// https://actix.rs/docs/application
// https://redis.io/docs/latest/

use std::sync::Mutex;

use actix_session::{SessionMiddleware, storage::RedisSessionStore};
use actix_web::{
    App, HttpServer, middleware, web::{self}
};

mod authentication;
mod protected_routes;
mod shared;
mod static_files;

struct AppState {
    _app_name: String,
}
struct AppStareWithCounter {
    _counter: Mutex<i32>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(
        env_logger::Env::new().default_filter_or("info")
    );

    log::info!("Starting server at http://localhost:8080");

    // Generate a random 32 byte key. Note that it is important to use a unique
    // private key for every project. Anyone with access to the key can generate
    // authentication cookies for any user!
    let private_key = actix_web::cookie::Key::generate();

    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let store = RedisSessionStore::new(&redis_url)
        .await
        .unwrap();

    let counter = web::Data::new(AppStareWithCounter {
        _counter: Mutex::new(0),
    });
    HttpServer::new(move || {
        App::new()
            .wrap(
                SessionMiddleware::builder(store.clone(), private_key.clone())
                .cookie_secure(false)
                .build()
            )
            // création et ajout d'un état par worker (clone par thread)
            .app_data(web::Data::new(AppState {
                _app_name: String::from("My Actix Web App"),
            }))
            // ajout d'un état partagé contenant un compteur accessible à tous les workers (Arc + Mutex)
            .app_data(counter.clone())
            // Exemple avec middleware : toutes les routes sont protégées
            .configure(authentication::authentication_config)

            // Exemple avec extractor : route individuelle protégée
            .route("/protected", web::get().to(protected_routes::protected_route))
            
            // Exemple avec middleware : tout le scope /api est protégé
            .service(
                web::scope("/api")
                .wrap(authentication::AuthenticationMiddleware)
                .route("/data", web::get().to(protected_routes::api_data))
                .route("/use", web::get().to(protected_routes::api_user_info))
            )
            // Fichiers static à la fin pour ne pas interférer avec les autres routes
            .configure(static_files::static_files_config)
            .wrap(middleware::Logger::default())
    })
    .workers(2)
    .keep_alive(std::time::Duration::from_secs(75))
    .shutdown_timeout(30)
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
