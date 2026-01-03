// https://actix.rs/docs/application
// https://redis.io/docs/latest/

use std::sync::Mutex;

use actix_session::{Session, SessionMiddleware, storage::RedisSessionStore};
use actix_web::{
    App, HttpResponse, HttpServer, Responder, Result, middleware, web::{self, get, post, resource}
};
use serde::{Deserialize, Serialize};

use crate::shared::IndexResponse;

mod authentication;
mod shared;

struct AppState {
    app_name: String,
}
struct AppStareWithCounter {
    counter: Mutex<i32>,
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
        counter: Mutex::new(0),
    });
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            // création et ajout d'un état par worker (clone par thread)
            .app_data(web::Data::new(AppState {
                app_name: String::from("My Actix Web App"),
            }))
            // ajout d'un état partagé contenant un compteur accessible à tous les workers (Arc + Mutex)
            .app_data(counter.clone())
            .route("/", web::get().to(index))
            .configure(authentication::authentication_config)
            // TODO - à remplacer par un vrai scope
            .service(
                web::scope("/do_something")
                .route("/", web::get().to(do_something))
            )

    })
    .workers(2)
    .keep_alive(None)
    .shutdown_timeout(30)
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}




async fn index(
    session: Session,
    app_state: web::Data<AppState>,
    app_state_with_counter: web::Data<AppStareWithCounter>) -> impl Responder {
    let user_id: Option<String> = session.get::<String>("user_id").unwrap();
    let session_counter: i32 = session
        .get::<i32>("counter")
        .unwrap_or(Some(0))
        .unwrap_or(0);
    let app_name = &app_state.app_name;
    let mut call_counter = app_state_with_counter.counter.lock().unwrap();
    *call_counter += 1;
    
    HttpResponse::Ok().json(shared::IndexResponse {
        user_id,
        session_counter,
    })
}

async fn do_something(session: Session) -> Result<HttpResponse> {
    let user_id: Option<String> = session.get::<String>("user_id").unwrap();
    let counter: i32 = session
        .get::<i32>("counter")
        .unwrap_or(Some(0))
        .map_or(1, |inner| inner + 1);
    session.insert("counter", counter)?;

    Ok(HttpResponse::Ok().json(shared::IndexResponse { user_id, session_counter: counter }))
}