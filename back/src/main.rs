use std::sync::Mutex;

use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let counter = web::Data::new(AppStareWithCounter {
        counter: Mutex::new(0),
    });
    HttpServer::new(move || {
        App::new()
            // création et ajout d'un état par worker (clone par thread)
            .app_data(web::Data::new(AppState {
                app_name: String::from("My Actix Web App"),
            }))
            // ajout d'un état partagé contenant un compteur accessible à tous les workers (Arc + Mutex)
            .app_data(counter.clone())
            .service(hello)
            .service(echo)
            .service(
                web::scope("/app")
                    .route("/", web::get().to(index))
            )
            .configure(config)
            .route("/hey", web::get().to(manual_hello))

    })
    .workers(2)
    .keep_alive(None)
    .shutdown_timeout(30)
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

// https://actix.rs/docs/application
// this function could be located in a different module
fn scoped_config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/test")
            .route(web::get().to(|| async { HttpResponse::Ok().body("test") }))
            .route(web::head().to(HttpResponse::MethodNotAllowed)),
    );
}

// this function could be located in a different module
fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/app")
            .route(web::get().to(|| async { HttpResponse::Ok().body("app") }))
            .route(web::head().to(HttpResponse::MethodNotAllowed)),
    );
}
async fn index(app_state: web::Data<AppState>, app_state_with_counter: web::Data<AppStareWithCounter>) -> impl Responder {
    let app_name = &app_state.app_name;
    let mut counter = app_state_with_counter.counter.lock().unwrap();
    *counter += 1;
    HttpResponse::Ok().body(format!("Welcome to {}! Request number: {}", app_name, counter))
}

struct AppState {
    app_name: String,
}
struct AppStareWithCounter {
    counter: Mutex<i32>,
}

// https://actix.rs/docs/getting-started
#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello world!")
}

#[post("/echo")]
async fn echo(req_body: String) -> impl Responder {
    HttpResponse::Ok().body(req_body)
}

async fn manual_hello() -> impl Responder {
    HttpResponse::Ok().body("Hey there!")
}